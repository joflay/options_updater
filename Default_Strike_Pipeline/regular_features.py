import argparse
import concurrent.futures
import math
import multiprocessing as mp
import os
import sys
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from dataset_paths import DEFAULT_STRIKE_CONTRACT_ROOT, DEFAULT_STRIKE_FEATURE_ROOT


INPUT_ROOT = DEFAULT_STRIKE_CONTRACT_ROOT
OUTPUT_ROOT = DEFAULT_STRIKE_FEATURE_ROOT

DTE_BUCKETS: Dict[str, Tuple[int, int]] = {
    "30d": (30, 45),
    "60d": (46, 70),
    "90d": (71, 90),
}

DELTA_TARGETS = {
    "put_10_delta": {"option_type": "put", "target": -0.10, "tolerance": 0.05},
    "put_25_delta": {"option_type": "put", "target": -0.25, "tolerance": 0.075},
    "atm_call": {"option_type": "call", "target": 0.50, "tolerance": 0.10},
    "atm_put": {"option_type": "put", "target": -0.50, "tolerance": 0.10},
    "call_25_delta": {"option_type": "call", "target": 0.25, "tolerance": 0.075},
    "call_10_delta": {"option_type": "call", "target": 0.10, "tolerance": 0.05},
}

REQUIRED_SOURCE_COLUMNS = {
    "trade_date",
    "underlying",
    "option_type",
    "strike",
    "underlying_close",
    "dte_days",
    "implied_volatility",
    "delta",
    "gamma",
    "theta",
    "vega",
}

EPSILON = 1e-12


def nanmean(values: Iterable[object]) -> float:
    numeric = [float(value) for value in values if pd.notna(value)]
    if not numeric:
        return np.nan
    return float(np.mean(numeric))


def safe_divide(numerator: object, denominator: object) -> float:
    if pd.isna(numerator) or pd.isna(denominator):
        return np.nan
    denominator = float(denominator)
    if abs(denominator) <= EPSILON:
        return np.nan
    return float(numerator) / denominator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build stock/date regular Greek delta-bucket option features."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=INPUT_ROOT,
        help="Contract CSV root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Daily feature CSV root.",
    )
    parser.add_argument(
        "--start-date",
        type=lambda value: date.fromisoformat(value),
        default=None,
        help="Optional inclusive trade-date filter in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        type=lambda value: date.fromisoformat(value),
        default=None,
        help="Optional inclusive trade-date filter in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild selected output shards even when valid resume outputs exist.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Worker processes for file-level parallelism.",
    )
    return parser.parse_args()


def trade_date_from_path(path: Path) -> Optional[date]:
    try:
        return date.fromisoformat(path.stem)
    except ValueError:
        return None


def iter_source_files(
    input_root: Path,
    start_date: Optional[date],
    end_date: Optional[date],
) -> List[Path]:
    files = sorted(input_root.rglob("*.csv"))
    filtered = []
    for path in files:
        trade_date = trade_date_from_path(path)
        if trade_date is None:
            continue
        if start_date is not None and trade_date < start_date:
            continue
        if end_date is not None and trade_date > end_date:
            continue
        filtered.append(path)
    return filtered


def output_path_for_source(
    source_path: Path,
    input_root: Path = INPUT_ROOT,
    output_root: Path = OUTPUT_ROOT,
) -> Path:
    return output_root / source_path.relative_to(input_root)


def temp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def stable_columns() -> List[str]:
    columns = [
        "trade_date",
        "underlying",
        "put_10_delta_iv",
        "put_25_delta_iv",
        "atm_call_iv",
        "atm_put_iv",
        "atm_iv",
        "call_25_delta_iv",
        "call_10_delta_iv",
        "risk_reversal_25_delta",
        "risk_reversal_10_delta",
        "put_skew_10_25",
        "call_skew_10_25",
        "curvature_25_delta",
        "curvature_10_delta",
        "skew_slope",
        "put_25_delta_gamma",
        "call_25_delta_gamma",
        "put_10_delta_gamma",
        "call_10_delta_gamma",
        "gamma_rr_25_delta",
        "gamma_wing_ratio",
        "vega_rr_25_delta",
        "theta_rr_25_delta",
        "near_atm_gamma_sum_raw",
        "put_gamma_sum_raw",
        "call_gamma_sum_raw",
        "gamma_imbalance_sum_raw",
        "near_atm_gamma_mean",
        "put_gamma_mean",
        "call_gamma_mean",
        "gamma_imbalance_mean",
    ]

    for dte_label in DTE_BUCKETS:
        columns.extend(
            [
                f"put_10_delta_iv_{dte_label}",
                f"put_25_delta_iv_{dte_label}",
                f"atm_call_iv_{dte_label}",
                f"atm_put_iv_{dte_label}",
                f"atm_iv_{dte_label}",
                f"call_25_delta_iv_{dte_label}",
                f"call_10_delta_iv_{dte_label}",
                f"risk_reversal_25_delta_{dte_label}",
                f"risk_reversal_10_delta_{dte_label}",
                f"put_skew_10_25_{dte_label}",
                f"call_skew_10_25_{dte_label}",
                f"curvature_25_delta_{dte_label}",
                f"curvature_10_delta_{dte_label}",
                f"skew_slope_{dte_label}",
                f"put_25_delta_gamma_{dte_label}",
                f"call_25_delta_gamma_{dte_label}",
                f"put_10_delta_gamma_{dte_label}",
                f"call_10_delta_gamma_{dte_label}",
                f"gamma_rr_25_delta_{dte_label}",
                f"gamma_wing_ratio_{dte_label}",
                f"vega_rr_25_delta_{dte_label}",
                f"theta_rr_25_delta_{dte_label}",
            ]
        )

    columns.extend(
        [
            "atm_term_slope",
            "put_skew_term_slope",
            "rr_term_slope",
            "n_contracts_total",
            "n_contracts_30d",
            "n_contracts_60d",
            "n_contracts_90d",
            "put_10_delta_distance",
            "put_25_delta_distance",
            "atm_call_delta_distance",
            "atm_put_delta_distance",
            "call_25_delta_distance",
            "call_10_delta_distance",
        ]
    )
    return columns


EXPECTED_COLUMNS = stable_columns()


def existing_output_is_valid(output_path: Path, source_path: Path) -> bool:
    if not output_path.exists():
        return False
    try:
        if output_path.stat().st_mtime < source_path.stat().st_mtime:
            return False
        columns = pd.read_csv(output_path, nrows=0).columns.tolist()
    except Exception:
        return False
    return columns == EXPECTED_COLUMNS


def read_existing_output(output_path: Path) -> pd.DataFrame:
    return ensure_output_columns(pd.read_csv(output_path))


def prepare_contract_frame(source_path: Path) -> pd.DataFrame:
    df = pd.read_csv(source_path)
    missing = sorted(REQUIRED_SOURCE_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{source_path} missing required column(s): {missing}")

    numeric_columns = [
        "strike",
        "underlying_close",
        "dte_days",
        "implied_volatility",
        "delta",
        "gamma",
        "theta",
        "vega",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    try:
        parsed_trade_dates = pd.to_datetime(df["trade_date"], errors="coerce", format="mixed")
    except (TypeError, ValueError):
        parsed_trade_dates = df["trade_date"].apply(
            lambda value: pd.to_datetime(value, errors="coerce")
        )
    df = df[parsed_trade_dates.notna()].copy()
    df["trade_date"] = parsed_trade_dates.loc[df.index].dt.date.astype(str)
    df["underlying"] = df["underlying"].astype(str).str.strip().str.upper()
    df["option_type"] = df["option_type"].astype(str).str.strip().str.lower()
    df = df[df["option_type"].isin(["call", "put"])].copy()

    valid = (
        (df["strike"] > 0)
        & (df["underlying_close"] > 0)
        & (df["dte_days"] > 0)
        & (df["implied_volatility"] > 0)
        & df["delta"].notna()
        & df["gamma"].notna()
    )
    df = df[valid].copy()
    if df.empty:
        return df

    df["moneyness"] = df["strike"] / df["underlying_close"]
    df = df[df["dte_days"].between(30, 90)].copy()
    return df


def dte_mask(frame: pd.DataFrame, dte_label: str) -> pd.Series:
    lower, upper = DTE_BUCKETS[dte_label]
    return frame["dte_days"].between(lower, upper)


def select_delta_contract(frame: pd.DataFrame, target_name: str) -> Tuple[Optional[pd.Series], float]:
    spec = DELTA_TARGETS[target_name]
    candidates = frame[frame["option_type"] == spec["option_type"]].copy()
    if candidates.empty:
        return None, np.nan

    distances = (candidates["delta"] - float(spec["target"])).abs()
    best_index = distances.idxmin()
    best_distance = float(distances.loc[best_index])
    if best_distance > float(spec["tolerance"]):
        return None, np.nan
    return candidates.loc[best_index], best_distance


def selected_value(selections: Dict[str, Optional[pd.Series]], name: str, column: str) -> float:
    selected = selections.get(name)
    if selected is None:
        return np.nan
    value = selected.get(column, np.nan)
    return float(value) if pd.notna(value) else np.nan


def regression_slope(points: List[Tuple[float, float]]) -> float:
    clean = [(x, y) for x, y in points if pd.notna(y)]
    if len(clean) < 2:
        return np.nan
    xs = np.array([point[0] for point in clean], dtype=float)
    ys = np.array([point[1] for point in clean], dtype=float)
    if np.allclose(xs, xs[0]):
        return np.nan
    return float(np.polyfit(xs, ys, 1)[0])


def mean_across_dte(row: dict, metric: str) -> float:
    return nanmean(row.get(f"{metric}_{dte_label}") for dte_label in DTE_BUCKETS)


def build_dte_features(row: dict, group: pd.DataFrame, dte_label: str) -> Dict[str, float]:
    dte_group = group[dte_mask(group, dte_label)].copy()
    row[f"n_contracts_{dte_label}"] = int(len(dte_group))

    selections: Dict[str, Optional[pd.Series]] = {}
    distances: Dict[str, float] = {}
    for target_name in DELTA_TARGETS:
        selected, distance = select_delta_contract(dte_group, target_name)
        selections[target_name] = selected
        distances[target_name] = distance

    for target_name in DELTA_TARGETS:
        row[f"{target_name}_iv_{dte_label}"] = selected_value(
            selections, target_name, "implied_volatility"
        )

    row[f"atm_iv_{dte_label}"] = nanmean(
        [row[f"atm_call_iv_{dte_label}"], row[f"atm_put_iv_{dte_label}"]]
    )
    row[f"risk_reversal_25_delta_{dte_label}"] = (
        row[f"call_25_delta_iv_{dte_label}"] - row[f"put_25_delta_iv_{dte_label}"]
    )
    row[f"risk_reversal_10_delta_{dte_label}"] = (
        row[f"call_10_delta_iv_{dte_label}"] - row[f"put_10_delta_iv_{dte_label}"]
    )
    row[f"put_skew_10_25_{dte_label}"] = (
        row[f"put_10_delta_iv_{dte_label}"] - row[f"put_25_delta_iv_{dte_label}"]
    )
    row[f"call_skew_10_25_{dte_label}"] = (
        row[f"call_10_delta_iv_{dte_label}"] - row[f"call_25_delta_iv_{dte_label}"]
    )
    row[f"curvature_25_delta_{dte_label}"] = (
        nanmean([row[f"put_25_delta_iv_{dte_label}"], row[f"call_25_delta_iv_{dte_label}"]])
        - row[f"atm_iv_{dte_label}"]
    )
    row[f"curvature_10_delta_{dte_label}"] = (
        nanmean([row[f"put_10_delta_iv_{dte_label}"], row[f"call_10_delta_iv_{dte_label}"]])
        - row[f"atm_iv_{dte_label}"]
    )
    row[f"skew_slope_{dte_label}"] = regression_slope(
        [
            (-0.10, row[f"put_10_delta_iv_{dte_label}"]),
            (-0.25, row[f"put_25_delta_iv_{dte_label}"]),
            (-0.50, row[f"atm_put_iv_{dte_label}"]),
            (0.50, row[f"atm_call_iv_{dte_label}"]),
            (0.25, row[f"call_25_delta_iv_{dte_label}"]),
            (0.10, row[f"call_10_delta_iv_{dte_label}"]),
        ]
    )

    for target_name in [
        "put_25_delta",
        "call_25_delta",
        "put_10_delta",
        "call_10_delta",
    ]:
        row[f"{target_name}_gamma_{dte_label}"] = selected_value(
            selections, target_name, "gamma"
        )

    row[f"gamma_rr_25_delta_{dte_label}"] = (
        row[f"call_25_delta_gamma_{dte_label}"] - row[f"put_25_delta_gamma_{dte_label}"]
    )
    row[f"gamma_wing_ratio_{dte_label}"] = safe_divide(
        row[f"put_10_delta_gamma_{dte_label}"],
        row[f"call_10_delta_gamma_{dte_label}"],
    )
    row[f"vega_rr_25_delta_{dte_label}"] = (
        selected_value(selections, "call_25_delta", "vega")
        - selected_value(selections, "put_25_delta", "vega")
    )
    row[f"theta_rr_25_delta_{dte_label}"] = (
        selected_value(selections, "call_25_delta", "theta")
        - selected_value(selections, "put_25_delta", "theta")
    )
    return distances


def add_raw_greek_summaries(row: dict, group: pd.DataFrame) -> None:
    call_mask = group["option_type"] == "call"
    put_mask = group["option_type"] == "put"
    near_atm_mask = group["moneyness"].between(0.97, 1.03)

    row["near_atm_gamma_sum_raw"] = float(group.loc[near_atm_mask, "gamma"].sum())
    row["put_gamma_sum_raw"] = float(group.loc[put_mask, "gamma"].sum())
    row["call_gamma_sum_raw"] = float(group.loc[call_mask, "gamma"].sum())
    row["gamma_imbalance_sum_raw"] = row["call_gamma_sum_raw"] - row["put_gamma_sum_raw"]

    row["near_atm_gamma_mean"] = float(group.loc[near_atm_mask, "gamma"].mean())
    row["put_gamma_mean"] = float(group.loc[put_mask, "gamma"].mean())
    row["call_gamma_mean"] = float(group.loc[call_mask, "gamma"].mean())
    row["gamma_imbalance_mean"] = row["call_gamma_mean"] - row["put_gamma_mean"]


def build_stock_date_row(group: pd.DataFrame) -> dict:
    row = {
        "trade_date": str(group["trade_date"].iloc[0]),
        "underlying": str(group["underlying"].iloc[0]),
        "n_contracts_total": int(len(group)),
    }

    distance_values: Dict[str, List[float]] = {name: [] for name in DELTA_TARGETS}
    for dte_label in DTE_BUCKETS:
        distances = build_dte_features(row, group, dte_label)
        for target_name, distance in distances.items():
            distance_values[target_name].append(distance)

    for metric in [
        "put_10_delta_iv",
        "put_25_delta_iv",
        "atm_call_iv",
        "atm_put_iv",
        "atm_iv",
        "call_25_delta_iv",
        "call_10_delta_iv",
        "risk_reversal_25_delta",
        "risk_reversal_10_delta",
        "put_skew_10_25",
        "call_skew_10_25",
        "curvature_25_delta",
        "curvature_10_delta",
        "skew_slope",
        "put_25_delta_gamma",
        "call_25_delta_gamma",
        "put_10_delta_gamma",
        "call_10_delta_gamma",
        "gamma_rr_25_delta",
        "gamma_wing_ratio",
        "vega_rr_25_delta",
        "theta_rr_25_delta",
    ]:
        row[metric] = mean_across_dte(row, metric)

    row["atm_term_slope"] = row["atm_iv_90d"] - row["atm_iv_30d"]
    row["put_skew_term_slope"] = row["put_25_delta_iv_90d"] - row["put_25_delta_iv_30d"]
    row["rr_term_slope"] = (
        row["risk_reversal_25_delta_90d"] - row["risk_reversal_25_delta_30d"]
    )

    add_raw_greek_summaries(row, group)

    distance_column_names = {
        "put_10_delta": "put_10_delta_distance",
        "put_25_delta": "put_25_delta_distance",
        "atm_call": "atm_call_delta_distance",
        "atm_put": "atm_put_delta_distance",
        "call_25_delta": "call_25_delta_distance",
        "call_10_delta": "call_10_delta_distance",
    }
    for target_name, column_name in distance_column_names.items():
        row[column_name] = nanmean(distance_values[target_name])

    return row


def build_daily_features(source_path: Path) -> pd.DataFrame:
    contracts = prepare_contract_frame(source_path)
    if contracts.empty:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    rows = [
        build_stock_date_row(group)
        for _, group in contracts.groupby(["trade_date", "underlying"], sort=True)
    ]
    return ensure_output_columns(pd.DataFrame(rows))


def build_daily_features_task(source_path: Path) -> Tuple[Path, pd.DataFrame]:
    return source_path, build_daily_features(source_path)


def drop_invalid_trade_dates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "trade_date" not in df.columns:
        return df
    parsed = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df[parsed.notna()].copy()
    if df.empty:
        return df
    df["trade_date"] = parsed.loc[df.index].dt.date.astype(str)
    return df


def ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_invalid_trade_dates(df)
    for column in EXPECTED_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    df = df[EXPECTED_COLUMNS].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def write_csv_atomic(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = temp_path(output_path)
    df.to_csv(tmp, index=False)
    tmp.replace(output_path)


def write_daily_shards(
    df: pd.DataFrame,
    source_paths: Dict[str, Path],
    input_root: Path = INPUT_ROOT,
    output_root: Path = OUTPUT_ROOT,
) -> int:
    written = 0
    if df.empty:
        return written
    for trade_date_value, day_frame in df.groupby("trade_date", sort=True):
        source_path = source_paths.get(str(trade_date_value))
        if source_path is None:
            print(f"  WARNING: skipping output shard for unknown trade_date={trade_date_value}")
            continue
        output_path = output_path_for_source(source_path, input_root, output_root)
        write_csv_atomic(ensure_output_columns(day_frame.sort_values("underlying")), output_path)
        written += 1
    return written


def missing_percent(df: pd.DataFrame, columns: List[str]) -> float:
    available = [column for column in columns if column in df.columns]
    if df.empty or not available:
        return math.nan
    return float(df[available].isna().mean().mean() * 100.0)


def print_run_summary(input_files_read: int, output_shards_written: int, features: pd.DataFrame) -> None:
    stock_date_rows = len(features)
    groups = {
        "delta_bucket_iv": [
            "put_10_delta_iv",
            "put_25_delta_iv",
            "atm_iv",
            "call_25_delta_iv",
            "call_10_delta_iv",
        ],
        "surface_shape": [
            "risk_reversal_25_delta",
            "risk_reversal_10_delta",
            "curvature_25_delta",
            "curvature_10_delta",
            "skew_slope",
        ],
        "greek_buckets": [
            "gamma_rr_25_delta",
            "gamma_wing_ratio",
            "vega_rr_25_delta",
            "theta_rr_25_delta",
        ],
        "raw_greek_summary": [
            "near_atm_gamma_sum_raw",
            "put_gamma_sum_raw",
            "call_gamma_sum_raw",
            "gamma_imbalance_sum_raw",
            "near_atm_gamma_mean",
            "put_gamma_mean",
            "call_gamma_mean",
            "gamma_imbalance_mean",
        ],
        "distances": [
            "put_10_delta_distance",
            "put_25_delta_distance",
            "atm_call_delta_distance",
            "atm_put_delta_distance",
            "call_25_delta_distance",
            "call_10_delta_distance",
        ],
    }

    print("\nRun summary")
    print(f"  input_files_read={input_files_read}")
    print(f"  output_shards_written={output_shards_written}")
    print(f"  stock_date_rows={stock_date_rows}")
    for group_name, columns in groups.items():
        value = missing_percent(features, columns)
        formatted = "nan" if math.isnan(value) else f"{value:.2f}%"
        print(f"  missing_{group_name}={formatted}")


def main() -> None:
    args = parse_args()
    if args.start_date is not None and args.end_date is not None and args.start_date > args.end_date:
        raise ValueError("--start-date cannot be later than --end-date")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    source_files = iter_source_files(args.input_root, args.start_date, args.end_date)
    if not source_files:
        print(f"No regular Greek files found under {args.input_root}")
        return

    print(f"Reading regular Greek files from {args.input_root}")
    print(f"Writing regular delta-bucket features to {args.output_root}")
    if args.start_date is not None or args.end_date is not None:
        print(f"trade date filter: start={args.start_date} end={args.end_date}")
    print(f"overwrite={args.overwrite}")
    print(f"rebuild_workers={args.workers}")

    daily_frames_by_index: Dict[int, pd.DataFrame] = {}
    rebuild_tasks: List[Tuple[int, Path]] = []
    for index, source_path in enumerate(source_files):
        output_path = output_path_for_source(source_path, args.input_root, args.output_root)

        if not args.overwrite and existing_output_is_valid(output_path, source_path):
            daily_frames_by_index[index] = read_existing_output(output_path)
            print(
                f"[{index + 1}/{len(source_files)}] reused {source_path} "
                f"rows={len(daily_frames_by_index[index])}"
            )
        else:
            rebuild_tasks.append((index, source_path))

    input_files_read = len(rebuild_tasks)
    if rebuild_tasks:
        index_by_path = {source_path: index for index, source_path in rebuild_tasks}
        mp_context = mp.get_context("fork")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=mp_context,
        ) as executor:
            future_to_path = {
                executor.submit(build_daily_features_task, source_path): source_path
                for _, source_path in rebuild_tasks
            }
            completed_rebuilds = 0
            for future in concurrent.futures.as_completed(future_to_path):
                source_path = future_to_path[future]
                _, frame = future.result()
                index = index_by_path[source_path]
                daily_frames_by_index[index] = frame
                completed_rebuilds += 1
                print(
                    f"[{index + 1}/{len(source_files)}] rebuilt {source_path} "
                    f"rows={len(frame)} rebuild_progress={completed_rebuilds}/{len(rebuild_tasks)}"
                )

    daily_frames = [
        daily_frames_by_index[index]
        for index in range(len(source_files))
        if index in daily_frames_by_index
    ]
    if daily_frames:
        features = pd.concat(daily_frames, ignore_index=True)
    else:
        features = pd.DataFrame(columns=EXPECTED_COLUMNS)
    features = ensure_output_columns(features)

    rebuilt_source_paths_by_date = {
        str(trade_date_from_path(source_path)): source_path
        for _, source_path in rebuild_tasks
    }
    output_shards_written = (
        write_daily_shards(
            features,
            rebuilt_source_paths_by_date,
            args.input_root,
            args.output_root,
        )
        if rebuilt_source_paths_by_date
        else 0
    )
    print_run_summary(input_files_read, output_shards_written, features)


if __name__ == "__main__":
    main()
