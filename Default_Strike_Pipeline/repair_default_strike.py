"""Audit or repair the canonical S&P 500 Default_Strike dataset."""

import argparse
import concurrent.futures
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Set

import numpy as np
import pandas as pd

import regular_features
from constituency import Snapshot, active_symbols_on_date, load_constituency_snapshots
from dataset_paths import (
    CLEAN_STOCK_ROOT,
    CONSTITUENCY_ROOT,
    DEFAULT_STRIKE_CONTRACT_ROOT,
    DEFAULT_STRIKE_FEATURE_ROOT,
)


REQUIRED_COLUMNS = {
    "underlying",
    "close",
    "strike",
    "dte_days",
    "time_to_expiry_years",
    "underlying_close",
    "risk_free_rate",
    "implied_volatility",
    "iv_failure_reason",
    "delta",
    "gamma",
    "theta",
    "vega",
}
POSITIVE_COLUMNS = [
    "close",
    "strike",
    "dte_days",
    "time_to_expiry_years",
    "underlying_close",
    "implied_volatility",
]
FINITE_COLUMNS = [*POSITIVE_COLUMNS, "risk_free_rate", "delta", "gamma", "theta", "vega"]


@dataclass(frozen=True)
class RepairResult:
    trade_date: str
    original_rows: int
    outside_universe_rows: int
    invalid_rows: int
    retained_rows_without_stock_file: int
    retained_symbols_without_stock_file: int
    kept_rows: int
    changed: bool
    feature_needs_rebuild: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Restrict Default_Strike contracts to the dated S&P 500 constituency, "
            "remove impossible IV/Greek rows, and rebuild daily features."
        )
    )
    parser.add_argument("--contracts-root", type=Path, default=DEFAULT_STRIKE_CONTRACT_ROOT)
    parser.add_argument("--features-root", type=Path, default=DEFAULT_STRIKE_FEATURE_ROOT)
    parser.add_argument("--constituency-root", type=Path, default=CONSTITUENCY_ROOT)
    parser.add_argument("--stock-root", type=Path, default=CLEAN_STOCK_ROOT)
    parser.add_argument("--start-date", type=date.fromisoformat, default=None)
    parser.add_argument("--end-date", type=date.fromisoformat, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write repaired contracts and rebuilt features. Without this flag the command is read-only.",
    )
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=None,
        help="Optional root for copies of changed contract and feature files before replacement.",
    )
    parser.add_argument("--report", type=Path, default=None, help="Optional CSV audit report path.")
    return parser.parse_args()


def trade_date_from_path(path: Path) -> date | None:
    try:
        return date.fromisoformat(path.stem)
    except ValueError:
        return None


def iter_contract_files(root: Path, start_date: date | None, end_date: date | None) -> List[Path]:
    files: List[Path] = []
    for path in sorted(root.glob("*/*/*.csv")):
        trade_date = trade_date_from_path(path)
        if trade_date is None:
            continue
        if start_date is not None and trade_date < start_date:
            continue
        if end_date is not None and trade_date > end_date:
            continue
        files.append(path)
    return files


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.repair.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def backup_file(path: Path, source_root: Path, backup_root: Path | None, category: str) -> None:
    if backup_root is None or not path.exists():
        return
    destination = backup_root / category / path.relative_to(source_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(path, destination)


def valid_contract_mask(frame: pd.DataFrame) -> pd.Series:
    numeric = frame[FINITE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    finite = pd.Series(np.isfinite(numeric.to_numpy()).all(axis=1), index=frame.index)
    positive = numeric[POSITIVE_COLUMNS].gt(0).all(axis=1)
    solved = frame["iv_failure_reason"].astype(str).str.strip().str.lower().eq("ok")
    return finite & positive & solved


def repair_file(
    path: Path,
    contracts_root: Path,
    features_root: Path,
    snapshots: List[Snapshot],
    stock_symbols: Set[str],
    apply: bool,
    backup_root: Path | None,
) -> RepairResult:
    trade_date = trade_date_from_path(path)
    if trade_date is None:
        raise ValueError(f"Cannot derive trade date from {path}")
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    frame["underlying"] = frame["underlying"].astype("string").str.strip().str.upper()
    frame["trade_date"] = trade_date.isoformat()
    active_symbols = active_symbols_on_date(snapshots, trade_date)
    outside_universe = ~frame["underlying"].isin(active_symbols)
    inside = frame.loc[~outside_universe].copy()
    invalid = ~valid_contract_mask(inside)
    repaired = inside.loc[~invalid].copy()

    retained_without_stock = ~repaired["underlying"].isin(stock_symbols)
    missing_stock_symbols = set(repaired.loc[retained_without_stock, "underlying"].dropna())

    changed = bool(outside_universe.any() or invalid.any())
    feature_path = features_root / path.relative_to(contracts_root)
    feature_needs_rebuild = changed or not regular_features.existing_output_is_valid(
        feature_path,
        path,
    )
    if apply and feature_needs_rebuild:
        if changed:
            backup_file(path, contracts_root, backup_root, "contracts")
            atomic_write_csv(repaired, path)
        backup_file(feature_path, features_root, backup_root, "Features")
        features = regular_features.build_daily_features(path)
        atomic_write_csv(features, feature_path)

    return RepairResult(
        trade_date=trade_date.isoformat(),
        original_rows=len(frame),
        outside_universe_rows=int(outside_universe.sum()),
        invalid_rows=int(invalid.sum()),
        retained_rows_without_stock_file=int(retained_without_stock.sum()),
        retained_symbols_without_stock_file=len(missing_stock_symbols),
        kept_rows=len(repaired),
        changed=changed,
        feature_needs_rebuild=feature_needs_rebuild,
    )


def main() -> None:
    args = parse_args()
    if args.start_date is not None and args.end_date is not None and args.start_date > args.end_date:
        raise ValueError("--start-date cannot be later than --end-date")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    snapshots = load_constituency_snapshots(args.constituency_root)
    stock_symbols = {
        path.name[: -len("_stock_data.csv")]
        for path in args.stock_root.glob("*_stock_data.csv")
    }
    files = iter_contract_files(args.contracts_root, args.start_date, args.end_date)
    if not files:
        raise ValueError(f"No contract files found under {args.contracts_root}")

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"mode={mode}")
    print(f"contracts_root={args.contracts_root}")
    print(f"features_root={args.features_root}")
    print(f"constituency_root={args.constituency_root} snapshots={len(snapshots)}")
    print(f"stock_root={args.stock_root} stock_files={len(stock_symbols)}")
    print(f"files={len(files)} workers={args.workers}")

    tasks = [
        (
            path,
            args.contracts_root,
            args.features_root,
            snapshots,
            stock_symbols,
            args.apply,
            args.backup_root,
        )
        for path in files
    ]
    if args.workers == 1:
        results = [repair_file(*task) for task in tasks]
    else:
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
                results = list(executor.map(run_repair_task, tasks))
        except (OSError, PermissionError) as exc:
            print(f"Worker processes unavailable; retrying sequentially. Reason: {exc}")
            results = [repair_file(*task) for task in tasks]

    report = pd.DataFrame([result.__dict__ for result in results]).sort_values("trade_date")
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.report, index=False)

    changed = report[report["changed"]]
    feature_rebuilds = report[report["feature_needs_rebuild"]]
    print(
        "summary "
        f"files_scanned={len(report)} files_changed={len(changed)} "
        f"features_to_rebuild={len(feature_rebuilds)} "
        f"original_rows={int(report['original_rows'].sum())} "
        f"outside_universe_rows={int(report['outside_universe_rows'].sum())} "
        f"invalid_rows={int(report['invalid_rows'].sum())} "
        f"retained_rows_without_stock_file={int(report['retained_rows_without_stock_file'].sum())} "
        f"kept_rows={int(report['kept_rows'].sum())}"
    )
    if not changed.empty:
        print(
            f"changed_date_range={changed['trade_date'].min()}..{changed['trade_date'].max()}"
        )
    if not args.apply:
        print("Dry run only; rerun with --apply after reviewing the summary/report.")


def run_repair_task(task: tuple) -> RepairResult:
    return repair_file(*task)


if __name__ == "__main__":
    main()
