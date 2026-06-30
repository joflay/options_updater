"""Update canonical Default Strike contracts and features from ATM IV outputs."""

import argparse
import os
import sys
from datetime import date
from pathlib import Path
from typing import List

import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parent
UPDATER_ROOT = Path(
    os.environ.get("OPTIONS_UPDATER_ROOT", str(PIPELINE_ROOT.parent))
).resolve()
sys.path.insert(0, str(PIPELINE_ROOT))
sys.path.insert(0, str(UPDATER_ROOT / "Data_Pipeline"))

import flatfile_iv
import flatfile_regular_greeks
import regular_features
from constituency import active_symbols_on_date, load_constituency_snapshots
from dataset_paths import (
    CONSTITUENCY_ROOT,
    DEFAULT_STRIKE_CONTRACT_ROOT,
    DEFAULT_STRIKE_FEATURE_ROOT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build missing or stale daily Default Strike contracts and features."
    )
    parser.add_argument("--source-iv-root", type=Path, default=flatfile_iv.OUTPUT_ROOT)
    parser.add_argument("--contracts-root", type=Path, default=DEFAULT_STRIKE_CONTRACT_ROOT)
    parser.add_argument("--features-root", type=Path, default=DEFAULT_STRIKE_FEATURE_ROOT)
    parser.add_argument("--constituency-root", type=Path, default=CONSTITUENCY_ROOT)
    parser.add_argument("--start-date", type=date.fromisoformat, default=None)
    parser.add_argument("--end-date", type=date.fromisoformat, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("DEFAULT_STRIKE_WORKERS", "4")),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def trade_date_from_path(path: Path) -> date | None:
    try:
        return date.fromisoformat(path.stem)
    except ValueError:
        return None


def source_files(
    root: Path,
    start_date: date | None,
    end_date: date | None,
) -> List[Path]:
    selected: List[Path] = []
    for path in sorted(root.glob("*/*/*.csv")):
        trade_date = trade_date_from_path(path)
        if trade_date is None:
            continue
        if start_date is not None and trade_date < start_date:
            continue
        if end_date is not None and trade_date > end_date:
            continue
        selected.append(path)
    return selected


def corresponding_path(path: Path, source_root: Path, output_root: Path) -> Path:
    return output_root / path.relative_to(source_root)


def contract_needs_update(source_path: Path, contract_path: Path, force: bool) -> bool:
    if force or not contract_path.exists():
        return True
    if source_path.stat().st_mtime > contract_path.stat().st_mtime:
        return True
    return not flatfile_regular_greeks.existing_regular_output_is_valid(contract_path)


def feature_needs_update(contract_path: Path, feature_path: Path, force: bool) -> bool:
    return force or not regular_features.existing_output_is_valid(feature_path, contract_path)


def verify_contract(path: Path, active_symbols: set[str], trade_date: date) -> int:
    frame = pd.read_csv(path)
    required = flatfile_regular_greeks.required_regular_output_cols()
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns after update: {missing}")
    normalized = frame["underlying"].astype("string").str.strip().str.upper()
    outside = int((~normalized.isin(active_symbols)).sum())
    invalid = int(
        frame[
            [
                "close",
                "strike",
                "underlying_close",
                "time_to_expiry_years",
                "risk_free_rate",
                "implied_volatility",
                *flatfile_regular_greeks.REGULAR_GREEK_COLS,
            ]
        ]
        .apply(pd.to_numeric, errors="coerce")
        .isna()
        .any(axis=1)
        .sum()
    )
    parsed_dates = pd.to_datetime(frame["trade_date"], errors="coerce", format="mixed").dt.date
    wrong_date = int((parsed_dates != trade_date).sum())
    duplicates = int(frame.duplicated(subset=["trade_date", "ticker"]).sum())
    if outside or invalid or wrong_date or duplicates:
        raise ValueError(
            f"{path} failed verification: outside={outside} invalid={invalid} "
            f"wrong_date={wrong_date} duplicates={duplicates}"
        )
    return len(frame)


def verify_feature(path: Path, active_symbols: set[str], trade_date: date) -> int:
    frame = pd.read_csv(path)
    if list(frame.columns) != regular_features.EXPECTED_COLUMNS:
        raise ValueError(f"{path} has an unexpected feature schema")
    outside = int(
        (~frame["underlying"].astype("string").str.strip().str.upper().isin(active_symbols)).sum()
    )
    parsed_dates = pd.to_datetime(frame["trade_date"], errors="coerce", format="mixed").dt.date
    wrong_date = int((parsed_dates != trade_date).sum())
    duplicates = int(frame.duplicated(subset=["trade_date", "underlying"]).sum())
    if outside or wrong_date or duplicates:
        raise ValueError(
            f"{path} failed verification: outside={outside} "
            f"wrong_date={wrong_date} duplicates={duplicates}"
        )
    return len(frame)


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.start_date and args.end_date and args.start_date > args.end_date:
        raise ValueError("--start-date cannot be later than --end-date")

    snapshots = load_constituency_snapshots(args.constituency_root)
    sources = source_files(args.source_iv_root, args.start_date, args.end_date)
    if not sources:
        print(f"No source IV files found under {args.source_iv_root}; nothing to update.")
        return

    contracts_written = 0
    features_written = 0
    files_skipped = 0
    contract_rows = 0
    feature_rows = 0
    for source_path in sources:
        trade_date = trade_date_from_path(source_path)
        if trade_date is None:
            continue
        active_symbols = active_symbols_on_date(snapshots, trade_date)
        contract_path = corresponding_path(source_path, args.source_iv_root, args.contracts_root)
        feature_path = corresponding_path(source_path, args.source_iv_root, args.features_root)

        rebuild_contract = contract_needs_update(source_path, contract_path, args.force)
        if rebuild_contract:
            contract_path.parent.mkdir(parents=True, exist_ok=True)
            flatfile_regular_greeks.process_existing_iv_file(
                source_path,
                contract_path,
                existing=None,
                allowed_underlyings=active_symbols,
                testing=False,
                test_stock="AAPL",
                row_workers=args.workers,
                trade_date=trade_date,
            )
            contracts_written += 1

        rebuild_feature = rebuild_contract or feature_needs_update(
            contract_path,
            feature_path,
            args.force,
        )
        if rebuild_feature:
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            features = regular_features.build_daily_features(contract_path)
            regular_features.write_csv_atomic(features, feature_path)
            features_written += 1

        if not rebuild_contract and not rebuild_feature:
            files_skipped += 1
            continue

        rows = verify_contract(contract_path, active_symbols, trade_date)
        daily_feature_rows = verify_feature(feature_path, active_symbols, trade_date)
        contract_rows += rows
        feature_rows += daily_feature_rows
        if rebuild_contract or rebuild_feature:
            print(
                f"{trade_date}: contracts={rows} contract_written={rebuild_contract} "
                f"features={daily_feature_rows} feature_written={rebuild_feature}"
            )

    print(
        "Default Strike update complete. "
        f"source_files={len(sources)} contracts_written={contracts_written} "
        f"features_written={features_written} files_skipped={files_skipped} "
        f"verified_contract_rows={contract_rows} "
        f"verified_feature_rows={feature_rows}"
    )


if __name__ == "__main__":
    main()
