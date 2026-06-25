import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import flatfile_iv
import main as pipeline

REPORT_PATH = pipeline.PROJECT_ROOT / "data_validation_report.csv"
MAX_SAMPLE_VALUES = 12
DATA_START_DATE = pipeline.STOCK_START_DATE
DATA_END_DATE = pipeline.STOCK_END_DATE
RAW_OPTION_REQUIRED_COLUMNS = {
    "ticker",
    "volume",
    "open",
    "close",
    "high",
    "low",
    "window_start",
    "transactions",
}
DIVIDEND_YIELD_FIELD = pipeline.DIVIDEND_YIELD_FIELD
SPLIT_MISMATCH_REPORT_PATH = pipeline.PROJECT_ROOT / "split_mismatch_report.csv"
SPLIT_REPAIR_RESULTS_PATH = pipeline.PROJECT_ROOT / "split_adjustment_repair_results.csv"


@dataclass
class Issue:
    severity: str
    area: str
    item: str
    message: str


def add_issue(issues: List[Issue], severity: str, area: str, item: str, message: str) -> None:
    issues.append(Issue(severity=severity, area=area, item=item, message=message))


def expected_constituency_dates(start_date: date, end_date: date) -> List[date]:
    dates = sorted({start_date, end_date})
    cursor = date(start_date.year, start_date.month, 1)
    while cursor <= end_date:
        dates.append(cursor)
        cursor = pipeline.add_months(cursor, 3)
    return sorted(set(dates))


def date_bounds_from_csv(path: Path) -> Optional[Tuple[date, date]]:
    try:
        df = pd.read_csv(path, usecols=lambda col: "date" in str(col).lower())
    except Exception:
        return None
    if df.empty:
        return None
    dates = pd.to_datetime(df[df.columns[0]], errors="coerce").dropna().dt.date
    if dates.empty:
        return None
    return dates.min(), dates.max()


def read_csv_if_present(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    except Exception:
        return None


def validate_constituency(issues: List[Issue]) -> List[Tuple[date, Set[str]]]:
    expected_dates = expected_constituency_dates(
        DATA_START_DATE,
        DATA_END_DATE,
    )
    for snapshot_date in expected_dates:
        path = pipeline.CONSTITUENCY_ROOT / f"constituency_{snapshot_date.isoformat()}.csv"
        if not path.exists():
            add_issue(
                issues,
                "error",
                "constituency",
                snapshot_date.isoformat(),
                f"Missing expected quarterly constituency snapshot: {path}",
            )
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            add_issue(issues, "error", "constituency", str(path), f"Unreadable CSV: {exc}")
            continue
        if df.empty:
            add_issue(issues, "error", "constituency", str(path), "Snapshot file is empty")
        if "ticker" not in {str(col).strip().lower() for col in df.columns}:
            add_issue(issues, "error", "constituency", str(path), "Missing ticker column")

    snapshots = pipeline.load_constituency_snapshots(pipeline.CONSTITUENCY_ROOT)
    if not snapshots:
        add_issue(
            issues,
            "error",
            "constituency",
            str(pipeline.CONSTITUENCY_ROOT),
            "No usable constituency snapshots found",
        )
    return snapshots


def validate_stocks(
    issues: List[Issue],
    membership_windows: Dict[str, List[pipeline.MembershipWindow]],
) -> None:
    for symbol, windows in sorted(membership_windows.items()):
        fetch_range = pipeline.symbol_fetch_range(windows, DATA_START_DATE, DATA_END_DATE)
        if fetch_range is None:
            continue
        expected_start, expected_end = fetch_range
        path = pipeline.STOCK_OUTPUT_ROOT / f"{symbol}_stock_data.csv"
        if not path.exists():
            add_issue(issues, "error", "stocks", symbol, f"Missing stock file: {path}")
            continue

        bounds = date_bounds_from_csv(path)
        if bounds is None:
            add_issue(issues, "error", "stocks", symbol, "Stock file has no parseable Date values")
            continue
        actual_start, actual_end = bounds
        if actual_start > expected_start:
            add_issue(
                issues,
                "error",
                "stocks",
                symbol,
                f"Stock history starts {actual_start}, expected <= {expected_start}",
            )
        if actual_end < expected_end:
            add_issue(
                issues,
                "error",
                "stocks",
                symbol,
                f"Stock history ends {actual_end}, expected >= {expected_end}",
            )

        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        close_col = next((col for col in df.columns if "close" in str(col).lower()), None)
        if close_col is None:
            add_issue(issues, "error", "stocks", symbol, "Missing close/price-close column")
        elif pd.to_numeric(df[close_col], errors="coerce").isna().any():
            add_issue(issues, "warning", "stocks", symbol, "Close column contains nonnumeric/null values")

        if "Dividends" not in df.columns:
            add_issue(
                issues,
                "warning",
                "stocks",
                symbol,
                "Missing Dividends cashflow column",
            )
        else:
            dividends = pd.to_numeric(df["Dividends"], errors="coerce")
            if dividends.isna().any():
                add_issue(
                    issues,
                    "warning",
                    "stocks",
                    symbol,
                    "Dividends column contains nonnumeric/null values",
                )
            if (dividends.dropna() < 0).any():
                add_issue(
                    issues,
                    "warning",
                    "stocks",
                    symbol,
                    "Dividends column contains negative values",
                )

        if DIVIDEND_YIELD_FIELD not in df.columns:
            add_issue(
                issues,
                "error",
                "stocks",
                symbol,
                f"Missing required {DIVIDEND_YIELD_FIELD} column",
            )
        else:
            dividend_yield = pd.to_numeric(df[DIVIDEND_YIELD_FIELD], errors="coerce")
            if dividend_yield.isna().any():
                add_issue(
                    issues,
                    "error",
                    "stocks",
                    symbol,
                    f"{DIVIDEND_YIELD_FIELD} contains nonnumeric/null values",
                )
            if (dividend_yield.dropna() < 0).any():
                add_issue(
                    issues,
                    "error",
                    "stocks",
                    symbol,
                    f"{DIVIDEND_YIELD_FIELD} contains negative values",
                )

        split_col = next((col for col in df.columns if str(col).lower() in {"stock splits", "stocksplits"}), None)
        if split_col is None:
            add_issue(
                issues,
                "warning",
                "splits",
                symbol,
                "Missing Stock Splits column; split handling could not be verified for this stock file",
            )
        else:
            splits = pd.to_numeric(df[split_col], errors="coerce")
            if splits.isna().any():
                add_issue(
                    issues,
                    "warning",
                    "splits",
                    symbol,
                    "Stock Splits column contains nonnumeric/null values",
                )
            if (splits.dropna() < 0).any():
                add_issue(
                    issues,
                    "error",
                    "splits",
                    symbol,
                    "Stock Splits column contains negative values",
                )


def validate_split_repair_artifacts(issues: List[Issue]) -> None:
    mismatch_report = read_csv_if_present(SPLIT_MISMATCH_REPORT_PATH)
    if mismatch_report is not None and not mismatch_report.empty:
        add_issue(
            issues,
            "error",
            "splits",
            str(SPLIT_MISMATCH_REPORT_PATH),
            f"Existing split mismatch report contains {len(mismatch_report)} rows",
        )

    repair_results = read_csv_if_present(SPLIT_REPAIR_RESULTS_PATH)
    if repair_results is not None and {"status", "symbols"}.issubset(repair_results.columns):
        incomplete = repair_results[repair_results["status"].astype(str).str.lower().str.strip() != "repaired"]
        if not incomplete.empty:
            status_counts = incomplete["status"].astype(str).value_counts().to_dict()
            add_issue(
                issues,
                "warning",
                "splits",
                str(SPLIT_REPAIR_RESULTS_PATH),
                f"Split repair results include non-repaired rows: {status_counts}",
            )


def validate_raw_option_flatfiles(
    issues: List[Issue],
    options_start: date,
    options_end: date,
) -> int:
    files = flatfile_iv.iter_input_files(pipeline.OPTIONS_INPUT_ROOT)
    files = flatfile_iv.filter_files_by_trade_date(files, options_start, options_end)
    if pipeline.LIMIT_OPTION_FILES is not None:
        files = files[:pipeline.LIMIT_OPTION_FILES]

    if not files:
        add_issue(
            issues,
            "error",
            "options",
            str(pipeline.OPTIONS_INPUT_ROOT),
            f"No raw option flatfiles found for {options_start} through {options_end}",
        )
        return 0

    latest_trade_date = max(flatfile_iv.trade_date_from_path(path) for path in files)
    if latest_trade_date < DATA_END_DATE:
        add_issue(
            issues,
            "error",
            "options",
            str(pipeline.OPTIONS_INPUT_ROOT),
            f"Latest raw option flatfile is {latest_trade_date}, expected >= {DATA_END_DATE}",
        )

    for input_path in files:
        try:
            df = pd.read_csv(input_path, compression="gzip", nrows=1000)
        except Exception as exc:
            add_issue(issues, "error", "options", str(input_path), f"Unreadable raw flatfile: {exc}")
            continue

        if df.empty:
            add_issue(issues, "warning", "options", str(input_path), "Raw flatfile sample is empty")
            continue

        missing_cols = sorted(RAW_OPTION_REQUIRED_COLUMNS - set(df.columns))
        if missing_cols:
            add_issue(
                issues,
                "error",
                "options",
                str(input_path),
                f"Raw flatfile missing required columns: {missing_cols}",
            )
            continue

        for numeric_col in ["volume", "open", "close", "high", "low", "transactions"]:
            if pd.to_numeric(df[numeric_col], errors="coerce").isna().any():
                add_issue(
                    issues,
                    "warning",
                    "options",
                    str(input_path),
                    f"Raw flatfile sample contains nonnumeric/null {numeric_col} values",
                )

    return len(files)


def write_report(issues: List[Issue]) -> None:
    rows = [issue.__dict__ for issue in issues]
    if rows:
        pd.DataFrame(rows).to_csv(REPORT_PATH, index=False)
    elif REPORT_PATH.exists():
        REPORT_PATH.unlink()


def main() -> None:
    issues: List[Issue] = []
    option_min_date, option_max_date = pipeline.available_option_date_range(pipeline.OPTIONS_INPUT_ROOT)
    options_start = max(option_min_date or DATA_START_DATE, DATA_START_DATE)
    options_end = min(option_max_date or DATA_END_DATE, DATA_END_DATE)

    snapshots = validate_constituency(issues)
    membership_windows = pipeline.build_membership_windows(
        snapshots,
        min(DATA_START_DATE, options_start),
        max(DATA_END_DATE, options_end),
        pipeline.LIMIT_SYMBOLS,
    ) if snapshots else {}

    if membership_windows:
        validate_stocks(issues, membership_windows)
        validate_split_repair_artifacts(issues)
    raw_option_files = validate_raw_option_flatfiles(issues, options_start, options_end)

    write_report(issues)

    error_count = sum(1 for issue in issues if issue.severity == "error")
    warning_count = sum(1 for issue in issues if issue.severity == "warning")
    print("Data validation complete.")
    print(f"  data_start={DATA_START_DATE}")
    print(f"  data_end={DATA_END_DATE}")
    print(f"  constituency_snapshots={len(snapshots)}")
    print(f"  membership_symbols={len(membership_windows)}")
    print(f"  raw_option_files={raw_option_files}")
    print(f"  errors={error_count} warnings={warning_count}")
    if issues:
        print(f"  report={REPORT_PATH}")
        for issue in issues[:20]:
            print(f"  [{issue.severity}] {issue.area} {issue.item}: {issue.message}")
        if len(issues) > 20:
            print(f"  ... {len(issues) - 20} more issues in report")
        raise SystemExit(1 if error_count else 0)


if __name__ == "__main__":
    main()
