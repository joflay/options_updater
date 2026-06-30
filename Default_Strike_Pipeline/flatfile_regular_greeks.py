"""Standalone OPRA pipeline that saves contract-strike Greeks.

This mirrors flatfile_iv.py's IV, filtering, checkpoint, daily-feature, and
final-feature flow, but computes Greeks at each option's actual strike instead
of normalizing every contract to hypothetical ATM (K = S) Greeks.
"""

import argparse
import concurrent.futures
import multiprocessing as mp
import os
import shutil
import sys
from concurrent.futures.process import BrokenProcessPool
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
UPDATER_ROOT = Path(
    os.environ.get("OPTIONS_UPDATER_ROOT", str(Path(__file__).resolve().parent.parent))
)
sys.path.insert(0, str(UPDATER_ROOT / "Data_Pipeline"))

import flatfile_iv
from constituency import active_symbols_on_date, load_constituency_snapshots
from config import DATA_DIR
from dataset_paths import CONSTITUENCY_ROOT, DEFAULT_STRIKE_CONTRACT_ROOT
from utils import american_option_price_crr

OUTPUT_ROOT = DEFAULT_STRIKE_CONTRACT_ROOT
SOURCE_IV_ROOT = flatfile_iv.OUTPUT_ROOT
DAILY_FEATURES_ROOT = Path(DATA_DIR) / "features_regular_greeks" / "day_aggs_v1"
CACHE_OUTPUT = Path(DATA_DIR) / "features_regular_greeks" / "underlying_close_cache.csv"
FINAL_FEATURES_DIR = Path(__file__).resolve().parent.parent / "final_features_regular_greeks"

GreekTask = Tuple[
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    str,
]

REGULAR_GREEK_COLS = ["delta", "gamma", "theta", "vega"]
LEGACY_DTE_MIN_DAYS = 30
LEGACY_DTE_MAX_DAYS = 90
REGULAR_FEATURE_MAP = {
    "avg_call_iv": "implied_volatility_call",
    "avg_put_iv": "implied_volatility_put",
    "avg_call_delta": "delta_call",
    "avg_put_delta": "delta_put",
    "avg_call_gamma": "gamma_call",
    "avg_put_gamma": "gamma_put",
    "avg_call_theta": "theta_call",
    "avg_put_theta": "theta_put",
    "avg_call_vega": "vega_call",
    "avg_put_vega": "vega_put",
}
DAILY_FEATURE_COLUMNS = ["trade_date", "underlying", *REGULAR_FEATURE_MAP.keys()]


def is_missing_value(value: Optional[float]) -> bool:
    return value is None or bool(pd.isna(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute implied volatilities and regular contract-strike Greeks for OPRA flat files."
    )
    parser.add_argument("--input-root", type=Path, default=flatfile_iv.INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--source-iv-root",
        type=Path,
        default=SOURCE_IV_ROOT,
        help="Existing ATM pipeline IV output root to reuse instead of recalculating IV.",
    )
    parser.add_argument("--daily-features-root", type=Path, default=DAILY_FEATURES_ROOT)
    parser.add_argument("--cache-output", type=Path, default=CACHE_OUTPUT)
    parser.add_argument("--final-features-dir", type=Path, default=FINAL_FEATURES_DIR)
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--start-date", type=lambda value: date.fromisoformat(value), default=None)
    parser.add_argument("--end-date", type=lambda value: date.fromisoformat(value), default=None)
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--test-stock", default="AAPL")
    parser.add_argument("--underlyings", nargs="+", default=None)
    parser.add_argument("--constituency-root", type=Path, default=CONSTITUENCY_ROOT)
    parser.add_argument(
        "--all-underlyings",
        action="store_true",
        help="Disable the default date-specific S&P 500 constituency filter.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Worker processes; file-level when multiple files are selected, row-level for one file.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--features-only", action="store_true")
    parser.add_argument("--overwrite-features", action="store_true")
    parser.add_argument(
        "--recalculate-missing-iv",
        action="store_true",
        help="Recalculate IV from raw flatfiles only when a matching source IV output is missing.",
    )
    return parser.parse_args()


def compute_row_regular_greeks_task(task: GreekTask) -> dict:
    implied_volatility, underlying_close, strike, time_to_expiry_years, risk_free_rate, dividend_yield, option_type = task
    if any(is_missing_value(value) for value in (implied_volatility, underlying_close, strike)):
        return {"delta": None, "gamma": None, "theta": None, "vega": None}
    if (
        is_missing_value(time_to_expiry_years)
        or time_to_expiry_years <= 0
        or is_missing_value(risk_free_rate)
    ):
        return {"delta": None, "gamma": None, "theta": None, "vega": None}

    sigma = float(implied_volatility)
    S = float(underlying_close)
    K = float(strike)
    T = float(time_to_expiry_years)
    r = float(risk_free_rate)
    q = 0.0 if is_missing_value(dividend_yield) else max(0.0, float(dividend_yield))
    if sigma <= 0 or S <= 0 or K <= 0 or T <= 0:
        return {"delta": None, "gamma": None, "theta": None, "vega": None}

    dS = max(0.01, 0.01 * S)
    d_sigma = max(1e-4, 0.01 * sigma)
    dT = min(1.0 / 365.25, max(T / 10.0, 1.0 / 365.25))

    try:
        base = american_option_price_crr(S, K, T, r, sigma, option_type, q)
        up = american_option_price_crr(S + dS, K, T, r, sigma, option_type, q)
        down = american_option_price_crr(max(1e-8, S - dS), K, T, r, sigma, option_type, q)
        vega_up = american_option_price_crr(S, K, T, r, sigma + d_sigma, option_type, q)
        vega_down = american_option_price_crr(S, K, T, r, max(1e-6, sigma - d_sigma), option_type, q)
        shorter = american_option_price_crr(S, K, max(1e-6, T - dT), r, sigma, option_type, q)
    except Exception:
        return {"delta": None, "gamma": None, "theta": None, "vega": None}

    return {
        "delta": float((up - down) / (2.0 * dS)),
        "gamma": float((up - 2.0 * base + down) / (dS ** 2)),
        "theta": float((shorter - base) / dT),
        "vega": float((vega_up - vega_down) / (2.0 * d_sigma)),
    }


def required_regular_output_cols() -> set[str]:
    return (flatfile_iv.REQUIRED_IV_OUTPUT_COLS - {"atm_delta", "atm_gamma", "atm_theta", "atm_vega"}) | set(
        REGULAR_GREEK_COLS
    )


def existing_regular_output_is_valid(output_path: Path) -> bool:
    try:
        columns = pd.read_csv(output_path, nrows=0).columns
    except Exception:
        return False
    return required_regular_output_cols().issubset(columns)


def csv_dte_bounds(path: Path) -> Optional[Tuple[int, int]]:
    try:
        dte = pd.read_csv(path, usecols=["dte_days"])["dte_days"]
    except Exception:
        return None
    dte = pd.to_numeric(dte, errors="coerce").dropna()
    if dte.empty:
        return None
    return int(dte.min()), int(dte.max())


def existing_regular_output_covers_target(output_path: Path) -> bool:
    if not output_path.exists() or not existing_regular_output_is_valid(output_path):
        return False
    bounds = csv_dte_bounds(output_path)
    if bounds is None:
        return False
    min_dte, max_dte = bounds
    if flatfile_iv.DTE_MIN_DAYS < LEGACY_DTE_MIN_DAYS:
        lower_covered = min_dte < LEGACY_DTE_MIN_DAYS
        upper_covered = flatfile_iv.DTE_MAX_DAYS <= LEGACY_DTE_MAX_DAYS or max_dte >= flatfile_iv.DTE_MAX_DAYS
        return lower_covered and upper_covered
    return min_dte <= flatfile_iv.DTE_MIN_DAYS and max_dte >= flatfile_iv.DTE_MAX_DAYS


def required_source_iv_cols() -> set[str]:
    return {
        "trade_date",
        "ticker",
        "underlying",
        "expiration_date",
        "option_type",
        "strike",
        "close",
        "underlying_close",
        "dte_days",
        "time_to_expiry_years",
        "risk_free_rate",
        flatfile_iv.DIVIDEND_YIELD_FIELD,
        "implied_volatility",
        "volume",
        "transactions",
    }


def source_iv_output_is_valid(source_iv_path: Path) -> bool:
    try:
        columns = pd.read_csv(source_iv_path, nrows=0).columns
    except Exception:
        return False
    return required_source_iv_cols().issubset(columns)


def source_iv_output_covers_target(source_iv_path: Path) -> bool:
    if not source_iv_path.exists() or not source_iv_output_is_valid(source_iv_path):
        return False
    bounds = csv_dte_bounds(source_iv_path)
    if bounds is None:
        return False
    min_dte, max_dte = bounds
    return min_dte <= flatfile_iv.DTE_MIN_DAYS and max_dte >= flatfile_iv.DTE_MAX_DAYS


def process_file_output_path(path: Path, input_root: Path, output_root: Path) -> Path:
    return output_root / path.relative_to(input_root).with_suffix("")


def process_file_will_write_output(path: Path, input_root: Path, output_root: Path, overwrite: bool) -> tuple[bool, Path]:
    output_path = process_file_output_path(path, input_root, output_root)
    if not overwrite and existing_regular_output_covers_target(output_path):
        return False, output_path
    return True, output_path


def preflight_process_file_task_outputs(tasks: List[Tuple]) -> None:
    for task in tasks:
        path, input_root, output_root, overwrite = task[:4]
        will_write, output_path = process_file_will_write_output(path, input_root, output_root, overwrite)
        if will_write:
            flatfile_iv.ensure_csv_output_path_writable(output_path)


def process_file_task(args: Tuple) -> tuple[Path, pd.DataFrame, Dict[str, Optional[float]]]:
    return process_file(*args)


def add_regular_greeks(df: pd.DataFrame, row_workers: int, trade_date: date) -> pd.DataFrame:
    enriched = df.drop(columns=["atm_delta", "atm_gamma", "atm_theta", "atm_vega"], errors="ignore").copy()
    for col in REGULAR_GREEK_COLS:
        if col in enriched.columns:
            enriched = enriched.drop(columns=[col])

    if enriched.empty:
        for col in REGULAR_GREEK_COLS:
            enriched[col] = pd.Series(dtype=float)
        return enriched

    greek_tasks: List[GreekTask] = list(
        zip(
            enriched["implied_volatility"].where(pd.notna(enriched["implied_volatility"]), None).tolist(),
            enriched["underlying_close"].where(pd.notna(enriched["underlying_close"]), None).tolist(),
            enriched["strike"].where(pd.notna(enriched["strike"]), None).tolist(),
            enriched["time_to_expiry_years"].where(pd.notna(enriched["time_to_expiry_years"]), None).tolist(),
            enriched["risk_free_rate"].where(pd.notna(enriched["risk_free_rate"]), None).tolist(),
            enriched[flatfile_iv.DIVIDEND_YIELD_FIELD].where(pd.notna(enriched[flatfile_iv.DIVIDEND_YIELD_FIELD]), None).tolist(),
            enriched["option_type"].tolist(),
        )
    )
    greek_values = flatfile_iv.run_batched_tasks(
        greek_tasks,
        row_workers,
        5000,
        "regular Greeks",
        trade_date,
        compute_row_regular_greeks_task,
    )
    enriched[REGULAR_GREEK_COLS] = pd.DataFrame(greek_values, index=enriched.index)[REGULAR_GREEK_COLS]
    return enriched


def feature_columns() -> List[str]:
    return [
        "trade_date",
        "ticker",
        "underlying",
        "expiration_date",
        "option_type",
        "strike",
        "close",
        "underlying_close",
        "dte_days",
        "time_to_expiry_years",
        "risk_free_rate",
        flatfile_iv.DIVIDEND_YIELD_FIELD,
        "implied_volatility",
        *REGULAR_GREEK_COLS,
        "volume",
        "transactions",
    ]


def load_existing_regular_output(output_path: Path, overwrite: bool) -> Optional[pd.DataFrame]:
    if overwrite or not output_path.exists() or not existing_regular_output_is_valid(output_path):
        return None
    return pd.read_csv(output_path)


def restrict_to_missing_dte_rows(enriched: pd.DataFrame, existing: Optional[pd.DataFrame]) -> pd.DataFrame:
    enriched = enriched[enriched["dte_days"].between(flatfile_iv.DTE_MIN_DAYS, flatfile_iv.DTE_MAX_DAYS)].copy()
    if existing is None or existing.empty or "dte_days" not in existing.columns:
        return enriched

    existing_dte = pd.to_numeric(existing["dte_days"], errors="coerce").dropna()
    if existing_dte.empty:
        return enriched

    lower_missing_cutoff = int(existing_dte.min())
    if flatfile_iv.DTE_MIN_DAYS < LEGACY_DTE_MIN_DAYS:
        lower_missing_cutoff = LEGACY_DTE_MIN_DAYS
    missing_mask = enriched["dte_days"] < lower_missing_cutoff
    if flatfile_iv.DTE_MAX_DAYS > LEGACY_DTE_MAX_DAYS:
        max_existing_dte = int(existing_dte.max())
        missing_mask = missing_mask | (enriched["dte_days"] > max_existing_dte)
    return enriched[missing_mask].copy()


def merge_existing_and_new_rows(existing: Optional[pd.DataFrame], new_rows: pd.DataFrame) -> pd.DataFrame:
    if existing is None:
        return new_rows
    if new_rows.empty:
        return existing

    combined = pd.concat([existing, new_rows], ignore_index=True, sort=False)
    dedupe_cols = [col for col in ["trade_date", "ticker"] if col in combined.columns]
    if len(dedupe_cols) == 2:
        combined = combined.drop_duplicates(subset=dedupe_cols, keep="last")
    sort_cols = [col for col in ["trade_date", "underlying", "expiration_date", "option_type", "strike"] if col in combined.columns]
    if sort_cols:
        combined = combined.sort_values(sort_cols).reset_index(drop=True)
    return combined


def process_existing_iv_file(
    source_iv_path: Path,
    output_path: Path,
    existing: Optional[pd.DataFrame],
    allowed_underlyings: Optional[set[str]],
    testing: bool,
    test_stock: str,
    row_workers: int,
    trade_date: date,
) -> pd.DataFrame:
    print(f"  reusing IV from {source_iv_path}")
    enriched = pd.read_csv(source_iv_path)
    if allowed_underlyings is not None:
        enriched["underlying"] = enriched["underlying"].map(flatfile_iv.normalize_underlying_symbol)
        enriched = enriched[enriched["underlying"].isin(allowed_underlyings)].copy()
    if testing:
        enriched = enriched[enriched["underlying"] == test_stock].copy()

    enriched = restrict_to_missing_dte_rows(enriched, existing)
    required_numeric = [
        "close",
        "strike",
        "underlying_close",
        "time_to_expiry_years",
        "risk_free_rate",
        "implied_volatility",
    ]
    for column in required_numeric:
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce")
    valid = enriched[required_numeric].notna().all(axis=1)
    valid &= enriched[["close", "strike", "underlying_close", "time_to_expiry_years", "implied_volatility"]].gt(0).all(axis=1)
    if "iv_failure_reason" in enriched.columns:
        valid &= enriched["iv_failure_reason"].astype(str).str.strip().str.lower().eq("ok")
    enriched = enriched[valid].copy()
    enriched = add_regular_greeks(enriched, row_workers, trade_date)
    enriched = enriched.dropna(subset=REGULAR_GREEK_COLS).copy()
    combined = merge_existing_and_new_rows(existing, enriched)
    flatfile_iv.write_output_csv_atomic(combined, output_path)
    return combined[feature_columns()].copy()


def process_file(
    path: Path,
    input_root: Path,
    output_root: Path,
    overwrite: bool,
    close_cache: Dict[str, Optional[float]],
    testing: bool,
    test_stock: str,
    allowed_underlyings: Optional[set[str]],
    row_workers: int,
    source_iv_root: Path,
    recalculate_missing_iv: bool,
    risk_free_rates: Optional[Dict[date, float]],
) -> tuple[Path, pd.DataFrame, Dict[str, Optional[float]]]:
    trade_date = flatfile_iv.trade_date_from_path(path)
    rel_path = path.relative_to(input_root)
    output_path = process_file_output_path(path, input_root, output_root)
    stock_cache: flatfile_iv.StockHistoryCache = {}
    local_close_cache = dict(close_cache)
    close_cache_updates: Dict[str, Optional[float]] = {}

    if not overwrite and existing_regular_output_covers_target(output_path):
        return output_path, pd.DataFrame(), close_cache_updates

    flatfile_iv.ensure_csv_output_path_writable(output_path)
    existing = load_existing_regular_output(output_path, overwrite)

    source_iv_path = process_file_output_path(path, input_root, source_iv_root)
    if source_iv_output_covers_target(source_iv_path):
        features = process_existing_iv_file(
            source_iv_path,
            output_path,
            existing,
            allowed_underlyings,
            testing,
            test_stock,
            row_workers,
            trade_date,
        )
        return output_path, features, close_cache_updates

    source_iv_valid = source_iv_output_is_valid(source_iv_path)
    if not source_iv_valid and not recalculate_missing_iv:
        raise RuntimeError(
            f"Missing reusable IV source {source_iv_path}. "
            "Run the original IV pipeline first or pass --recalculate-missing-iv."
        )
    if risk_free_rates is None:
        raise RuntimeError("Cannot recalculate missing IV without risk-free rates.")

    print(f"  processing {rel_path}")

    df = pd.read_csv(path, compression="gzip")
    if df.empty:
        combined = merge_existing_and_new_rows(existing, df)
        flatfile_iv.write_output_csv_atomic(combined, output_path)
        return output_path, combined[feature_columns()].copy() if not combined.empty else pd.DataFrame(), close_cache_updates

    parsed = df["ticker"].apply(flatfile_iv.parse_opra_option_ticker)
    meta = pd.DataFrame(
        [
            item if isinstance(item, dict) else {
                "underlying": None,
                "expiration_date": None,
                "option_type": None,
                "strike": None,
            }
            for item in parsed
        ]
    )

    enriched = pd.concat([df, meta], axis=1)
    enriched = enriched[enriched["option_type"].isin(["call", "put"])].copy()
    if enriched.empty:
        combined = merge_existing_and_new_rows(existing, enriched)
        flatfile_iv.write_output_csv_atomic(combined, output_path)
        return output_path, combined[feature_columns()].copy() if not combined.empty else pd.DataFrame(), close_cache_updates

    enriched["trade_date"] = pd.Timestamp(trade_date)
    enriched["expiration_date"] = pd.to_datetime(enriched["expiration_date"], errors="coerce")
    enriched["close"] = pd.to_numeric(enriched["close"], errors="coerce")
    enriched["strike"] = pd.to_numeric(enriched["strike"], errors="coerce")
    enriched["dte_days"] = (enriched["expiration_date"] - enriched["trade_date"]).dt.days
    enriched["time_to_expiry_years"] = enriched["dte_days"] / 365.25
    enriched = restrict_to_missing_dte_rows(enriched, existing)
    enriched["underlying"] = enriched["underlying"].map(flatfile_iv.normalize_underlying_symbol)
    enriched = enriched[enriched["underlying"].map(flatfile_iv.is_supported_underlying)].copy()
    if allowed_underlyings is not None:
        enriched = enriched[enriched["underlying"].isin(allowed_underlyings)].copy()
    enriched = enriched[enriched["volume"].fillna(0) > 0].copy()
    enriched = enriched[enriched["close"].fillna(0) > 0].copy()
    if testing:
        enriched = enriched[enriched["underlying"] == test_stock].copy()
    if enriched.empty:
        combined = merge_existing_and_new_rows(existing, enriched)
        flatfile_iv.write_output_csv_atomic(combined, output_path)
        return output_path, combined[feature_columns()].copy() if not combined.empty else pd.DataFrame(), close_cache_updates

    underlying_closes = flatfile_iv.fetch_underlying_closes(
        enriched["underlying"].dropna().unique(),
        trade_date,
        local_close_cache,
        stock_cache,
        testing=testing,
        test_stock=test_stock,
        cache_updates=close_cache_updates,
    )
    print(f"  loaded underlying closes for {len(underlying_closes)} symbols")
    enriched["underlying_close"] = enriched["underlying"].map(underlying_closes)
    dividend_yields = flatfile_iv.fetch_underlying_dividend_yields(
        enriched["underlying"].dropna().unique(),
        trade_date,
        stock_cache,
        testing=testing,
        test_stock=test_stock,
    )
    enriched[flatfile_iv.DIVIDEND_YIELD_FIELD] = enriched["underlying"].map(dividend_yields).fillna(0.0)
    risk_free_rate = risk_free_rates.get(trade_date)
    if risk_free_rate is None:
        raise ValueError(f"Missing risk-free rate for {trade_date.isoformat()}")
    enriched["risk_free_rate"] = risk_free_rate
    rows_before_iv = len(enriched)

    iv_tasks: List[flatfile_iv.IvTask] = list(
        zip(
            enriched["underlying_close"].where(pd.notna(enriched["underlying_close"]), None).tolist(),
            enriched["strike"].where(pd.notna(enriched["strike"]), None).tolist(),
            enriched["close"].where(pd.notna(enriched["close"]), None).tolist(),
            enriched["time_to_expiry_years"].where(pd.notna(enriched["time_to_expiry_years"]), None).tolist(),
            enriched["risk_free_rate"].where(pd.notna(enriched["risk_free_rate"]), None).tolist(),
            enriched[flatfile_iv.DIVIDEND_YIELD_FIELD].where(pd.notna(enriched[flatfile_iv.DIVIDEND_YIELD_FIELD]), None).tolist(),
            enriched["option_type"].tolist(),
        )
    )
    iv_results = flatfile_iv.run_batched_tasks(iv_tasks, row_workers, 5000, "IV", trade_date, flatfile_iv.compute_row_iv_task)
    enriched["implied_volatility"] = [result[0] for result in iv_results]
    enriched["iv_failure_reason"] = [result[1] for result in iv_results]
    missing_underlying = int(enriched["underlying_close"].isna().sum())
    invalid_expiry = int(
        enriched["time_to_expiry_years"].isna().sum()
        + (enriched["time_to_expiry_years"] <= 0).fillna(False).sum()
    )
    iv_ready_rows = int(enriched["implied_volatility"].notna().sum())
    iv_failed_rows = rows_before_iv - iv_ready_rows
    n_calls = int((enriched["option_type"] == "call").sum())
    n_puts = int((enriched["option_type"] == "put").sum())
    iv_failure_counts = (
        enriched.loc[enriched["iv_failure_reason"] != "ok", "iv_failure_reason"]
        .value_counts()
        .to_dict()
    )

    enriched = add_regular_greeks(enriched.dropna(subset=["implied_volatility"]).copy(), row_workers, trade_date)
    enriched = enriched.dropna(subset=REGULAR_GREEK_COLS).copy()

    combined = merge_existing_and_new_rows(existing, enriched)
    flatfile_iv.write_output_csv_atomic(combined, output_path)
    features = combined[feature_columns()].copy()
    print(
        f"  diagnostics total_options={rows_before_iv} calls={n_calls} puts={n_puts} "
        f"missing_underlying={missing_underlying} invalid_expiry={invalid_expiry} "
        f"iv_written={iv_ready_rows} iv_failed={iv_failed_rows}"
    )
    if iv_failure_counts:
        print(f"  iv_failure_breakdown={iv_failure_counts}")
    return output_path, features, close_cache_updates


def build_daily_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=DAILY_FEATURE_COLUMNS)
    if "option_type" not in df.columns and "ticker" in df.columns:
        parsed = df["ticker"].apply(flatfile_iv.parse_opra_option_ticker)
        df["option_type"] = [item.get("option_type") if isinstance(item, dict) else None for item in parsed]

    required = {"trade_date", "underlying", "option_type", "implied_volatility", *REGULAR_GREEK_COLS}
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Cannot build regular Greek daily features; missing columns {missing}")

    df["trade_date"] = flatfile_iv.parse_mixed_datetimes(df["trade_date"]).dt.date
    df["underlying"] = df["underlying"].map(flatfile_iv.normalize_underlying_symbol)
    df = df[df["option_type"].isin(["call", "put"])].copy()
    df = df[df["underlying"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=DAILY_FEATURE_COLUMNS)

    aggregated = (
        df.groupby(["trade_date", "underlying", "option_type"])[["implied_volatility", *REGULAR_GREEK_COLS]]
        .mean()
        .reset_index()
    )
    feature_df = (
        aggregated[["trade_date", "underlying"]]
        .drop_duplicates()
        .sort_values(["trade_date", "underlying"])
        .reset_index(drop=True)
    )

    for option_type, suffix in [("call", "call"), ("put", "put")]:
        side = aggregated[aggregated["option_type"] == option_type].copy()
        side = side.rename(
            columns={
                "implied_volatility": f"implied_volatility_{suffix}",
                "delta": f"delta_{suffix}",
                "gamma": f"gamma_{suffix}",
                "theta": f"theta_{suffix}",
                "vega": f"vega_{suffix}",
            }
        )
        side_cols = ["trade_date", "underlying", *[col for col in REGULAR_FEATURE_MAP.values() if col.endswith(f"_{suffix}")]]
        feature_df = feature_df.merge(side[side_cols], on=["trade_date", "underlying"], how="left")

    for feature_name, source_col in REGULAR_FEATURE_MAP.items():
        feature_df[feature_name] = feature_df[source_col] if source_col in feature_df.columns else pd.NA
    return feature_df[DAILY_FEATURE_COLUMNS]


def daily_feature_output_path(greek_output_path: Path, greek_root: Path, daily_features_root: Path) -> Path:
    return daily_features_root / greek_output_path.relative_to(greek_root)


def daily_feature_file_is_valid(path: Path, source_path: Optional[Path] = None) -> bool:
    try:
        if source_path is not None and path.stat().st_mtime < source_path.stat().st_mtime:
            return False
        columns = pd.read_csv(path, nrows=0).columns
    except (Exception, OSError):
        return False
    return set(DAILY_FEATURE_COLUMNS).issubset(columns)


def export_daily_feature_file(
    greek_output_path: Path,
    daily_features_root: Path,
    greek_root: Path = OUTPUT_ROOT,
    reuse_existing: bool = True,
) -> tuple[Path, int]:
    output_path = daily_feature_output_path(greek_output_path, greek_root, daily_features_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if reuse_existing and output_path.exists() and daily_feature_file_is_valid(output_path, greek_output_path):
        return output_path, flatfile_iv.count_csv_data_rows(output_path)

    feature_df = build_daily_feature_frame(pd.read_csv(greek_output_path))
    flatfile_iv.write_output_csv_atomic(feature_df, output_path)
    return output_path, len(feature_df)


def export_final_feature_files(daily_features_root: Path, output_dir: Path) -> None:
    daily_feature_files = flatfile_iv.iter_daily_feature_files(daily_features_root)
    if not daily_feature_files:
        print(f"Skipping final feature export; no daily feature files under {daily_features_root}")
        return

    daily_frames = [pd.read_csv(path) for path in daily_feature_files]
    daily_frames = [df for df in daily_frames if not df.empty]
    if not daily_frames:
        print(f"Skipping final feature export; daily feature files under {daily_features_root} are empty")
        return

    all_daily_features = pd.concat(daily_frames, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for feature_name in REGULAR_FEATURE_MAP:
        feature_df = all_daily_features[["trade_date", "underlying", feature_name]].copy()
        feature_df.to_csv(output_dir / f"{feature_name}.csv", index=False)
    print(f"Wrote regular Greek final feature CSVs to {output_dir}")


def rebuild_feature_files_from_outputs(
    greek_root: Path,
    daily_features_root: Path,
    final_features_dir: Path,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    limit_files: Optional[int] = None,
    overwrite_daily_features: bool = False,
) -> int:
    output_files = flatfile_iv.iter_daily_feature_files(greek_root)
    output_files = flatfile_iv.filter_files_by_trade_date(output_files, start_date, end_date)
    if limit_files is not None:
        output_files = output_files[:limit_files]
    if not output_files:
        print(f"No regular Greek output files found under {greek_root}")
        return 0

    feature_rows = 0
    for completed, output_path in enumerate(output_files, start=1):
        daily_features_path, daily_feature_rows = export_daily_feature_file(
            output_path,
            daily_features_root,
            greek_root,
            reuse_existing=not overwrite_daily_features,
        )
        feature_rows += daily_feature_rows
        print(f"[{completed}/{len(output_files)}] features={daily_features_path} feature_rows={daily_feature_rows}")

    temp_final_features_dir = final_features_dir.with_name(f".{final_features_dir.name}.tmp")
    if temp_final_features_dir.exists():
        shutil.rmtree(temp_final_features_dir)
    export_final_feature_files(daily_features_root, temp_final_features_dir)
    if temp_final_features_dir.exists():
        if final_features_dir.exists():
            shutil.rmtree(final_features_dir)
        temp_final_features_dir.replace(final_features_dir)
    return feature_rows


def main() -> None:
    args = parse_args()
    if args.start_date is not None and args.end_date is not None and args.start_date > args.end_date:
        raise ValueError("--start-date cannot be later than --end-date")
    if args.features_only:
        rebuild_feature_files_from_outputs(
            args.output_root,
            args.daily_features_root,
            args.final_features_dir,
            args.start_date,
            args.end_date,
            args.limit_files,
            overwrite_daily_features=args.overwrite_features,
        )
        return

    test_stock = flatfile_iv.normalize_underlying_symbol(args.test_stock)
    requested_underlyings = (
        {flatfile_iv.normalize_underlying_symbol(symbol) for symbol in args.underlyings if flatfile_iv.normalize_underlying_symbol(symbol)}
        if args.underlyings
        else None
    )
    files = flatfile_iv.iter_input_files(args.input_root)
    files = flatfile_iv.filter_files_by_trade_date(files, args.start_date, args.end_date)
    if args.limit_files is not None:
        files = files[:args.limit_files]
    if not files:
        print(f"No input files found under {args.input_root}")
        return

    print(f"Processing {len(files)} flat files from {args.input_root}")
    print(f"Reusing existing IV outputs from {args.source_iv_root}")
    print(f"Saving regular Greek-enriched option data under {args.output_root}")
    print(f"Writing daily stock/date feature CSVs under {args.daily_features_root}")
    print(f"Writing final feature CSVs to {args.final_features_dir}")
    if args.start_date is not None or args.end_date is not None:
        print(f"trade date filter: start={args.start_date} end={args.end_date}")
    snapshots = None if args.all_underlyings else load_constituency_snapshots(args.constituency_root)
    if snapshots is None:
        print("constituency filter: disabled by --all-underlyings")
    else:
        print(
            f"constituency filter: date-specific S&P 500 snapshots from "
            f"{args.constituency_root} ({len(snapshots)} snapshots)"
        )
    if requested_underlyings is not None:
        print(f"additional underlying filter: {sorted(requested_underlyings)}")

    files_with_missing_source = [
        path
        for path in files
        if not existing_regular_output_covers_target(process_file_output_path(path, args.input_root, args.output_root))
        and not source_iv_output_is_valid(process_file_output_path(path, args.input_root, args.source_iv_root))
    ]
    files_requiring_raw_iv = [
        path
        for path in files
        if not existing_regular_output_covers_target(process_file_output_path(path, args.input_root, args.output_root))
        and not source_iv_output_covers_target(process_file_output_path(path, args.input_root, args.source_iv_root))
    ]
    if files_with_missing_source and not args.recalculate_missing_iv:
        sample = ", ".join(str(path) for path in files_with_missing_source[:5])
        if len(files_with_missing_source) > 5:
            sample = f"{sample}, ..."
        raise RuntimeError(
            f"{len(files_with_missing_source)} file(s) do not have reusable IV outputs under "
            f"{args.source_iv_root}: {sample}. Run the original IV pipeline first, or pass "
            "--recalculate-missing-iv to compute only the missing IV files."
        )

    if files_requiring_raw_iv:
        print(
            f"Calculating IV from raw flatfiles for {len(files_requiring_raw_iv)} file(s) whose reusable IV "
            f"does not cover {flatfile_iv.DTE_MIN_DAYS}-{flatfile_iv.DTE_MAX_DAYS} DTE; "
            "already-saved regular Greeks rows will be preserved."
        )
        risk_free_rates: Optional[Dict[date, float]] = flatfile_iv.load_risk_free_rates_for_files(files_requiring_raw_iv)
    else:
        print("All selected files have reusable IV output covering the target DTE range; skipping IV recalculation and FRED rate loading.")
        risk_free_rates = None

    close_cache = flatfile_iv.load_close_cache(args.cache_output)
    feature_rows = 0
    temp_final_features_dir = args.final_features_dir.with_name(f".{args.final_features_dir.name}.tmp")
    if temp_final_features_dir.exists():
        shutil.rmtree(temp_final_features_dir)

    row_workers = args.workers if len(files) == 1 else 1
    file_tasks = []
    for path in files:
        if snapshots is None:
            allowed_underlyings = requested_underlyings
        else:
            allowed_underlyings = active_symbols_on_date(
                snapshots,
                flatfile_iv.trade_date_from_path(path),
            )
            if requested_underlyings is not None:
                allowed_underlyings &= requested_underlyings
        file_tasks.append(
            (
                path,
                args.input_root,
                args.output_root,
                args.overwrite,
                close_cache,
                args.testing,
                test_stock,
                allowed_underlyings,
                row_workers,
                args.source_iv_root,
                args.recalculate_missing_iv,
                risk_free_rates,
            )
        )
    preflight_process_file_task_outputs(file_tasks)

    completed = 0
    if len(files) > 1 and args.workers > 1:
        try:
            mp_context = mp.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp_context) as executor:
                results = executor.map(process_file_task, file_tasks)
                for output_path, features, close_cache_updates in results:
                    close_cache.update(close_cache_updates)
                    daily_features_path, daily_feature_rows = export_daily_feature_file(
                        output_path,
                        args.daily_features_root,
                        args.output_root,
                        reuse_existing=not (args.overwrite or args.overwrite_features),
                    )
                    feature_rows += daily_feature_rows
                    flatfile_iv.save_close_cache(args.cache_output, close_cache)
                    completed += 1
                    print(
                        f"[{completed}/{len(files)}] wrote {output_path} rows={len(features)} "
                        f"features={daily_features_path} feature_rows={daily_feature_rows}"
                    )
        except BrokenProcessPool as exc:
            print(f"Regular Greek worker process pool failed; retrying sequentially. Original error: {exc}")

    for output_path, features, close_cache_updates in map(process_file_task, file_tasks[completed:]):
        close_cache.update(close_cache_updates)
        daily_features_path, daily_feature_rows = export_daily_feature_file(
            output_path,
            args.daily_features_root,
            args.output_root,
            reuse_existing=not (args.overwrite or args.overwrite_features),
        )
        feature_rows += daily_feature_rows
        flatfile_iv.save_close_cache(args.cache_output, close_cache)
        completed += 1
        print(
            f"[{completed}/{len(files)}] wrote {output_path} rows={len(features)} "
            f"features={daily_features_path} feature_rows={daily_feature_rows}"
        )

    export_final_feature_files(args.daily_features_root, temp_final_features_dir)
    if temp_final_features_dir.exists():
        if args.final_features_dir.exists():
            shutil.rmtree(args.final_features_dir)
        temp_final_features_dir.replace(args.final_features_dir)
    print(f"Wrote {feature_rows} stock/date feature rows under {args.daily_features_root}")


if __name__ == "__main__":
    main()
