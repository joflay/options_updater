import argparse
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import flatfile_iv
import main as pipeline

try:
    import yfinance as yf
except Exception as exc:
    yf = None
    YFINANCE_IMPORT_ERROR: Optional[Exception] = exc
else:
    YFINANCE_IMPORT_ERROR = None


RAW_STOCK_ROOT = pipeline.STOCK_OUTPUT_ROOT
SPLIT_MISMATCH_REPORT = pipeline.PROJECT_ROOT / "split_mismatch_report.csv"
REPAIR_RESULTS_PATH = pipeline.PROJECT_ROOT / "split_adjustment_repair_results.csv"
PRICE_CLOSE_FIELD = pipeline.PRICE_CLOSE_FIELD
DIVIDEND_YIELD_FIELD = pipeline.DIVIDEND_YIELD_FIELD
DIVIDENDS_FIELD = "Dividends"
MIN_CLOSE_PRICE = pipeline.MIN_CLOSE_PRICE
RAW_SOURCE = "yfinance-raw-split-safe"
COMMON_SPLIT_FACTORS = (2.0, 3.0, 4.0, 5.0, 10.0, 20.0)
SPLIT_FACTOR_TOLERANCE = 0.08
TERMINAL_REPAIR_STATUSES = {
    "repaired",
    "missing_input",
    "missing_risk_free_rate",
    "no_repaired_rows",
}

OPTION_OUTPUT_COLUMNS = [
    "ticker",
    "volume",
    "open",
    "close",
    "high",
    "low",
    "window_start",
    "transactions",
    "underlying",
    "expiration_date",
    "option_type",
    "strike",
    "trade_date",
    "dte_days",
    "time_to_expiry_years",
    "underlying_close",
    DIVIDEND_YIELD_FIELD,
    "risk_free_rate",
    "implied_volatility",
    "iv_failure_reason",
    "atm_delta",
    "atm_gamma",
    "atm_theta",
    "atm_vega",
]


def find_field_column(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    normalized = {
        str(col).lower().replace(" ", "").replace(".", "").replace("_", ""): col
        for col in df.columns
    }
    for name in names:
        key = name.lower().replace(" ", "").replace(".", "").replace("_", "")
        if key in normalized:
            return normalized[key]
    return None


def symbols_from_stock_root(root: Path) -> List[str]:
    return sorted(
        path.name.replace("_stock_data.csv", "").upper()
        for path in root.glob("*_stock_data.csv")
    )


def fetch_yfinance_raw_history(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError(f"yfinance import failed: {YFINANCE_IMPORT_ERROR}")

    raw = yf.download(
        tickers=pipeline.yf_symbol_for_ticker(symbol),
        start=start_date.isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),
        group_by="ticker",
        auto_adjust=False,
        actions=True,
        progress=False,
        threads=False,
    )
    raw = pipeline.yfinance_symbol_frame(raw, pipeline.yf_symbol_for_ticker(symbol), 1)
    if raw.empty:
        return pd.DataFrame()

    out = raw.copy().reset_index()
    if "Date" not in out.columns:
        out = out.rename(columns={out.columns[0]: "Date"})
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).sort_values("Date").copy()

    close_col = find_field_column(out, ["Close"])
    if close_col is None:
        return pd.DataFrame()
    open_col = find_field_column(out, ["Open"])
    high_col = find_field_column(out, ["High"])
    low_col = find_field_column(out, ["Low"])
    volume_col = find_field_column(out, ["Volume"])
    split_col = find_field_column(out, ["Stock Splits", "StockSplits"])
    dividend_col = find_field_column(out, [DIVIDENDS_FIELD])

    splits = (
        pd.to_numeric(out[split_col], errors="coerce").fillna(0.0)
        if split_col is not None
        else pd.Series(0.0, index=out.index)
    )
    split_multiplier = splits.where(splits > 0.0, 1.0)
    future_split_factor = split_multiplier.iloc[::-1].cumprod().iloc[::-1] / split_multiplier

    result = pd.DataFrame()
    result["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    for output_col, source_col in [
        ("Open", open_col),
        ("High", high_col),
        ("Low", low_col),
        (PRICE_CLOSE_FIELD, close_col),
    ]:
        if source_col is None:
            result[output_col] = pd.NA
        else:
            result[output_col] = pd.to_numeric(out[source_col], errors="coerce") * future_split_factor
    result["Adj Close"] = pd.to_numeric(out[close_col], errors="coerce")
    result[DIVIDENDS_FIELD] = (
        pd.to_numeric(out[dividend_col], errors="coerce").fillna(0.0)
        if dividend_col is not None
        else 0.0
    )
    result[DIVIDEND_YIELD_FIELD] = (
        pd.to_numeric(result[DIVIDENDS_FIELD], errors="coerce").fillna(0.0)
        / pd.to_numeric(result[PRICE_CLOSE_FIELD], errors="coerce")
    ).replace([float("inf"), float("-inf")], 0.0).fillna(0.0).clip(lower=0.0)
    result["Volume"] = (
        pd.to_numeric(out[volume_col], errors="coerce")
        if volume_col is not None
        else pd.NA
    )
    result["ticker"] = symbol
    result["lseg_universe"] = RAW_SOURCE
    result["lseg_ric"] = ""
    result["Stock Splits"] = splits
    result = result.dropna(subset=["Date", PRICE_CLOSE_FIELD])
    result = result[result[PRICE_CLOSE_FIELD] > MIN_CLOSE_PRICE].copy()
    return result.reset_index(drop=True)


def write_raw_stock_files(output_root: Path, overwrite: bool, dry_run: bool) -> List[Dict[str, object]]:
    symbols = symbols_from_stock_root(pipeline.STOCK_OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, object]] = []
    print(
        f"Building raw split-safe prices for {len(symbols)} symbols "
        f"{pipeline.STOCK_START_DATE} -> {pipeline.STOCK_END_DATE}"
    )
    for symbol in symbols:
        output_path = output_root / f"{symbol}_stock_data.csv"
        if output_path.exists() and not overwrite:
            results.append({"symbol": symbol, "status": "exists", "path": str(output_path)})
            continue
        if dry_run:
            print(f"  would fetch {symbol}")
            results.append({"symbol": symbol, "status": "dry_run", "path": str(output_path)})
            continue
        try:
            df = fetch_yfinance_raw_history(
                symbol,
                pipeline.STOCK_START_DATE,
                pipeline.STOCK_END_DATE,
            )
        except Exception as exc:
            print(f"  failed {symbol}: {exc}")
            results.append({"symbol": symbol, "status": "failed", "error": str(exc)})
            continue
        if df.empty:
            print(f"  empty {symbol}")
            results.append({"symbol": symbol, "status": "empty", "path": str(output_path)})
            continue
        df.to_csv(output_path, index=False)
        print(f"  saved {symbol} rows={len(df)} -> {output_path}")
        results.append({"symbol": symbol, "status": "saved", "path": str(output_path), "rows": len(df)})
    return results


def read_stock_close_map(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["Date", PRICE_CLOSE_FIELD])
    try:
        df = pd.read_csv(path, usecols=lambda col: str(col) in {"Date", PRICE_CLOSE_FIELD})
    except Exception:
        return pd.DataFrame(columns=["Date", PRICE_CLOSE_FIELD])
    if "Date" not in df.columns or PRICE_CLOSE_FIELD not in df.columns:
        return pd.DataFrame(columns=["Date", PRICE_CLOSE_FIELD])
    out = df[["Date", PRICE_CLOSE_FIELD]].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.date
    out[PRICE_CLOSE_FIELD] = pd.to_numeric(out[PRICE_CLOSE_FIELD], errors="coerce")
    return out.dropna(subset=["Date", PRICE_CLOSE_FIELD])


def nearest_common_split_factor(ratio: float) -> Optional[float]:
    if ratio <= 0:
        return None
    candidates = list(COMMON_SPLIT_FACTORS)
    if ratio < 1.0:
        candidates.extend(1.0 / factor for factor in COMMON_SPLIT_FACTORS)
    best = min(candidates, key=lambda factor: abs(ratio / factor - 1.0))
    if abs(ratio / best - 1.0) <= SPLIT_FACTOR_TOLERANCE:
        return best
    return None


def scan_stock_close_differences(raw_root: Path, report_path: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    symbols = symbols_from_stock_root(raw_root)
    print(f"Scanning clean-vs-raw closes for {len(symbols)} symbols")
    for symbol in symbols:
        clean = read_stock_close_map(pipeline.STOCK_OUTPUT_ROOT / f"{symbol}_stock_data.csv").rename(
            columns={PRICE_CLOSE_FIELD: "clean_close"}
        )
        raw_prices = read_stock_close_map(raw_root / f"{symbol}_stock_data.csv").rename(
            columns={PRICE_CLOSE_FIELD: "raw_close"}
        )
        if clean.empty or raw_prices.empty:
            continue
        merged = clean.merge(raw_prices, on="Date", how="inner")
        merged = merged[
            (merged["Date"] >= pipeline.STOCK_START_DATE)
            & (merged["Date"] <= pipeline.STOCK_END_DATE)
        ].copy()
        if merged.empty:
            continue
        merged["ratio"] = merged["raw_close"] / merged["clean_close"]
        for _, row in merged.iterrows():
            factor = nearest_common_split_factor(float(row["ratio"]))
            if factor is None or factor == 1.0:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": row["Date"].isoformat(),
                    "clean_close": row["clean_close"],
                    "raw_close": row["raw_close"],
                    "ratio": row["ratio"],
                    "split_factor": factor,
                }
            )
    report = pd.DataFrame(rows)
    report.to_csv(report_path, index=False)
    print(f"Wrote {len(report)} split mismatch rows to {report_path}")
    if not report.empty:
        print(report.groupby("symbol")["trade_date"].agg(["min", "max", "count"]).to_string())
    return report


def affected_dates_from_report(report: pd.DataFrame) -> Dict[date, set[str]]:
    affected: Dict[date, set[str]] = {}
    if report.empty:
        return affected
    for _, row in report.iterrows():
        symbol = str(row["symbol"]).strip().upper()
        trade_date = date.fromisoformat(str(row["trade_date"]))
        affected.setdefault(trade_date, set()).add(symbol)
    return affected


def parse_symbols_field(value: object) -> List[str]:
    if pd.isna(value):
        return []
    return [part.strip().upper() for part in str(value).split(";") if part.strip()]


def load_completed_symbol_days(results_path: Path, retry_failed: bool) -> set[Tuple[date, str]]:
    if not results_path.exists():
        return set()
    try:
        df = pd.read_csv(results_path)
    except Exception:
        return set()
    required = {"trade_date", "symbols", "status"}
    if df.empty or not required.issubset(df.columns):
        return set()

    skip_statuses = {"repaired"} if retry_failed else TERMINAL_REPAIR_STATUSES
    completed: set[Tuple[date, str]] = set()
    for _, row in df.iterrows():
        if str(row["status"]) not in skip_statuses:
            continue
        try:
            trade_date = date.fromisoformat(str(row["trade_date"]))
        except ValueError:
            continue
        for symbol in parse_symbols_field(row["symbols"]):
            completed.add((trade_date, symbol))
    return completed


def filter_completed_affected(
    affected: Dict[date, set[str]],
    completed: set[Tuple[date, str]],
) -> Dict[date, set[str]]:
    remaining: Dict[date, set[str]] = {}
    for trade_date, symbols in affected.items():
        pending = {symbol for symbol in symbols if (trade_date, symbol) not in completed}
        if pending:
            remaining[trade_date] = pending
    return remaining


def append_repair_result(results_path: Path, row: Dict[str, object]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        results_path,
        mode="a",
        header=not results_path.exists(),
        index=False,
    )


def filter_symbols_already_repaired_on_disk(
    trade_date: date,
    symbols: set[str],
    close_lookup: Dict[Tuple[str, date], float],
    iv_path: Path,
) -> Tuple[set[str], set[str]]:
    if not symbols or not iv_path.exists():
        return symbols, set()
    try:
        df = pd.read_csv(iv_path, usecols=["underlying", "underlying_close"])
    except Exception:
        return symbols, set()
    if df.empty:
        return symbols, set()

    pending: set[str] = set()
    already_repaired: set[str] = set()
    for symbol in symbols:
        expected = close_lookup.get((symbol, trade_date))
        if expected is None:
            pending.add(symbol)
            continue
        rows = df[df["underlying"] == symbol]
        if rows.empty:
            pending.add(symbol)
            continue
        closes = pd.to_numeric(rows["underlying_close"], errors="coerce").dropna()
        if closes.empty:
            pending.add(symbol)
            continue
        relative_error = (closes - expected).abs() / abs(expected)
        if (relative_error <= 1e-6).all():
            already_repaired.add(symbol)
        else:
            pending.add(symbol)
    return pending, already_repaired


def raw_close_lookup(raw_root: Path, symbols: Iterable[str]) -> Dict[Tuple[str, date], float]:
    lookup: Dict[Tuple[str, date], float] = {}
    for symbol in sorted(set(symbols)):
        df = read_stock_close_map(raw_root / f"{symbol}_stock_data.csv")
        for _, row in df.iterrows():
            lookup[(symbol, row["Date"])] = float(row[PRICE_CLOSE_FIELD])
    return lookup


def risk_free_rate_from_iv_file(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, usecols=["risk_free_rate"])
    except Exception:
        return None
    rates = pd.to_numeric(df["risk_free_rate"], errors="coerce").dropna()
    if rates.empty:
        return None
    return float(rates.iloc[-1])


def recompute_option_rows_for_symbols(
    raw_path: Path,
    symbols: set[str],
    close_lookup: Dict[Tuple[str, date], float],
    risk_free_rate: float,
    workers: int,
) -> pd.DataFrame:
    trade_date = flatfile_iv.trade_date_from_path(raw_path)
    df = pd.read_csv(raw_path, compression="gzip")
    if df.empty:
        return pd.DataFrame()

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
    enriched["underlying"] = enriched["underlying"].map(flatfile_iv.normalize_underlying_symbol)
    enriched = enriched[
        enriched["underlying"].isin(symbols)
        & enriched["option_type"].isin(["call", "put"])
    ].copy()
    if enriched.empty:
        return pd.DataFrame()

    enriched["trade_date"] = pd.Timestamp(trade_date)
    enriched["expiration_date"] = pd.to_datetime(enriched["expiration_date"], errors="coerce")
    enriched["close"] = pd.to_numeric(enriched["close"], errors="coerce")
    enriched["strike"] = pd.to_numeric(enriched["strike"], errors="coerce")
    enriched["volume"] = pd.to_numeric(enriched["volume"], errors="coerce")
    enriched["dte_days"] = (enriched["expiration_date"] - enriched["trade_date"]).dt.days
    enriched["time_to_expiry_years"] = enriched["dte_days"] / 365.25
    enriched = enriched[enriched["dte_days"].between(flatfile_iv.DTE_MIN_DAYS, flatfile_iv.DTE_MAX_DAYS)].copy()
    enriched = enriched[enriched["volume"].fillna(0) > 0].copy()
    enriched = enriched[enriched["close"].fillna(0) > 0].copy()
    if enriched.empty:
        return pd.DataFrame()

    stock_cache: flatfile_iv.StockHistoryCache = {}
    enriched["underlying_close"] = [
        close_lookup.get((symbol, trade_date))
        for symbol in enriched["underlying"].tolist()
    ]
    enriched[DIVIDEND_YIELD_FIELD] = [
        flatfile_iv.load_local_dividend_yield(symbol, trade_date, stock_cache)
        for symbol in enriched["underlying"].tolist()
    ]
    enriched["risk_free_rate"] = risk_free_rate

    iv_tasks = list(
        zip(
            enriched["underlying_close"].where(pd.notna(enriched["underlying_close"]), None).tolist(),
            enriched["strike"].where(pd.notna(enriched["strike"]), None).tolist(),
            enriched["close"].where(pd.notna(enriched["close"]), None).tolist(),
            enriched["time_to_expiry_years"].where(pd.notna(enriched["time_to_expiry_years"]), None).tolist(),
            enriched["risk_free_rate"].where(pd.notna(enriched["risk_free_rate"]), None).tolist(),
            enriched[DIVIDEND_YIELD_FIELD].where(pd.notna(enriched[DIVIDEND_YIELD_FIELD]), None).tolist(),
            enriched["option_type"].tolist(),
        )
    )
    iv_results = flatfile_iv.run_batched_tasks(
        tasks=iv_tasks,
        workers=workers,
        batch_size=5000,
        task_label="split repair IV",
        trade_date=trade_date,
        worker_fn=flatfile_iv.compute_row_iv_task,
    )
    enriched["implied_volatility"] = [result[0] for result in iv_results]
    enriched["iv_failure_reason"] = [result[1] for result in iv_results]
    enriched = enriched.dropna(subset=["implied_volatility"]).copy()
    if enriched.empty:
        return pd.DataFrame()

    greek_tasks = list(
        zip(
            enriched["implied_volatility"].tolist(),
            enriched["underlying_close"].tolist(),
            enriched["time_to_expiry_years"].tolist(),
            enriched["risk_free_rate"].tolist(),
            enriched[DIVIDEND_YIELD_FIELD].tolist(),
            enriched["option_type"].tolist(),
        )
    )
    greek_values = flatfile_iv.run_batched_tasks(
        tasks=greek_tasks,
        workers=workers,
        batch_size=5000,
        task_label="split repair ATM Greeks",
        trade_date=trade_date,
        worker_fn=flatfile_iv.compute_row_atm_greeks_task,
    )
    greeks = pd.DataFrame(greek_values, index=enriched.index)
    enriched[["atm_delta", "atm_gamma", "atm_theta", "atm_vega"]] = greeks[
        ["delta", "gamma", "theta", "vega"]
    ]
    return enriched[[col for col in OPTION_OUTPUT_COLUMNS if col in enriched.columns]].copy()


def repair_option_files(
    affected: Dict[date, set[str]],
    raw_root: Path,
    results_path: Path,
    dry_run: bool,
    workers: int,
    retry_failed: bool,
) -> List[Dict[str, object]]:
    completed = load_completed_symbol_days(results_path, retry_failed)
    if completed:
        before = sum(len(values) for values in affected.values())
        affected = filter_completed_affected(affected, completed)
        after = sum(len(values) for values in affected.values())
        print(f"Skipping {before - after} symbol-days already recorded in {results_path}")
    if not affected:
        print("No pending symbol-days remain after applying repair ledger.")
        return []

    symbols = {symbol for day_symbols in affected.values() for symbol in day_symbols}
    close_lookup = raw_close_lookup(raw_root, symbols)
    results: List[Dict[str, object]] = []

    for trade_date, day_symbols in sorted(affected.items()):
        raw_path = pipeline.OPTIONS_INPUT_ROOT / f"{trade_date.year:04d}" / f"{trade_date.month:02d}" / f"{trade_date.isoformat()}.csv.gz"
        iv_path = pipeline.OPTIONS_OUTPUT_ROOT / f"{trade_date.year:04d}" / f"{trade_date.month:02d}" / f"{trade_date.isoformat()}.csv"
        if not raw_path.exists() or not iv_path.exists():
            row = {"trade_date": trade_date.isoformat(), "symbols": ";".join(sorted(day_symbols)), "status": "missing_input"}
            results.append(row)
            if not dry_run:
                append_repair_result(results_path, row)
            continue
        day_symbols, already_repaired = filter_symbols_already_repaired_on_disk(
            trade_date,
            day_symbols,
            close_lookup,
            iv_path,
        )
        if already_repaired:
            row = {
                "trade_date": trade_date.isoformat(),
                "symbols": ";".join(sorted(already_repaired)),
                "status": "repaired",
                "note": "detected_existing_repair",
            }
            results.append(row)
            if not dry_run:
                append_repair_result(results_path, row)
            print(f"  already repaired {trade_date}: {sorted(already_repaired)}")
        if not day_symbols:
            continue
        if dry_run:
            print(f"  would repair {trade_date}: {sorted(day_symbols)}")
            row = {"trade_date": trade_date.isoformat(), "symbols": ";".join(sorted(day_symbols)), "status": "dry_run"}
            results.append(row)
            continue

        risk_free_rate = risk_free_rate_from_iv_file(iv_path)
        if risk_free_rate is None:
            rates = flatfile_iv.load_risk_free_rates_for_files([raw_path])
            risk_free_rate = rates.get(trade_date)
        if risk_free_rate is None:
            row = {"trade_date": trade_date.isoformat(), "symbols": ";".join(sorted(day_symbols)), "status": "missing_risk_free_rate"}
            results.append(row)
            if not dry_run:
                append_repair_result(results_path, row)
            continue

        repaired_rows = recompute_option_rows_for_symbols(
            raw_path,
            day_symbols,
            close_lookup,
            risk_free_rate,
            workers,
        )
        if repaired_rows.empty:
            row = {"trade_date": trade_date.isoformat(), "symbols": ";".join(sorted(day_symbols)), "status": "no_repaired_rows"}
            results.append(row)
            if not dry_run:
                append_repair_result(results_path, row)
            continue

        existing = pd.read_csv(iv_path)
        remaining = existing[~existing["underlying"].isin(day_symbols)].copy()
        merged = pd.concat([remaining, repaired_rows], ignore_index=True, sort=False)
        if "ticker" in merged.columns:
            merged = merged.sort_values(["underlying", "ticker"]).reset_index(drop=True)
        flatfile_iv.write_output_csv_atomic(merged, iv_path)
        feature_path, feature_rows = flatfile_iv.export_daily_feature_file(
            iv_path,
            pipeline.DAILY_FEATURES_ROOT,
            pipeline.OPTIONS_OUTPUT_ROOT,
            reuse_existing=False,
        )
        print(
            f"  repaired {trade_date}: symbols={sorted(day_symbols)} "
            f"option_rows={len(repaired_rows)} feature_rows={feature_rows}"
        )
        row = {
            "trade_date": trade_date.isoformat(),
            "symbols": ";".join(sorted(day_symbols)),
            "status": "repaired",
            "option_rows": len(repaired_rows),
            "feature_path": str(feature_path),
        }
        results.append(row)
        if not dry_run:
            append_repair_result(results_path, row)

    print(f"Appended {len(results)} repair result rows to {results_path}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair split-adjustment mismatches by rebuilding raw stock closes, "
            "detecting affected symbol-days, and recomputing option IV/Greeks."
        )
    )
    parser.add_argument("--raw-output-root", type=Path, default=RAW_STOCK_ROOT)
    parser.add_argument("--split-report", type=Path, default=SPLIT_MISMATCH_REPORT)
    parser.add_argument("--results-file", type=Path, default=REPAIR_RESULTS_PATH)
    parser.add_argument(
        "--overwrite-raw",
        action="store_true",
        help="Refetch raw stock files even if they already exist.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=pipeline.OPTIONS_WORKERS,
        help="Worker processes for IV and Greek row computations.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry prior non-repaired rows in the repair ledger instead of skipping them.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline.load_dotenv()
    raw_results = write_raw_stock_files(
        args.raw_output_root,
        args.overwrite_raw,
        args.dry_run,
    )
    if not args.dry_run and raw_results:
        pd.DataFrame(raw_results).to_csv(
            args.split_report.with_name("raw_stock_build_results.csv"),
            index=False,
        )
    report = scan_stock_close_differences(args.raw_output_root, args.split_report)
    affected = affected_dates_from_report(report)
    if not affected:
        print("No split-adjustment mismatches found; no option repair needed.")
        return
    print(
        f"Repairing {sum(len(values) for values in affected.values())} "
        f"symbol-days across {len(affected)} option files"
    )
    repair_option_files(
        affected,
        args.raw_output_root,
        args.results_file,
        args.dry_run,
        args.workers,
        args.retry_failed,
    )


if __name__ == "__main__":
    main()
