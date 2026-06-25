import argparse
import os
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import main as pipeline

try:
    import lseg.data as ld
    LSEG_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:
    ld = None
    LSEG_IMPORT_ERROR = exc

try:
    import yfinance as yf
except Exception as exc:
    yf = None
    YFINANCE_IMPORT_ERROR: Optional[Exception] = exc
else:
    YFINANCE_IMPORT_ERROR = None

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"lseg\.data\._tools\._dataframe",
)

REPORT_PATH = pipeline.PROJECT_ROOT / "data_validation_report.csv"
FAILURES_PATH = pipeline.PROJECT_ROOT / "data_repair_failures.csv"
PRICE_CLOSE_FIELD = pipeline.PRICE_CLOSE_FIELD
DIVIDEND_YIELD_FIELD = pipeline.DIVIDEND_YIELD_FIELD
DIVIDENDS_FIELD = "Dividends"
MIN_CLOSE_PRICE = pipeline.MIN_CLOSE_PRICE
LSEG_SOURCE = "lseg-data-repair"

LSEG_HISTORY_FIELDS = [
    "TR.PriceOpen",
    "TR.PriceHigh",
    "TR.PriceLow",
    "TR.PriceClose",
    "TR.Volume",
]

OUTPUT_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    PRICE_CLOSE_FIELD,
    "Adj Close",
    DIVIDENDS_FIELD,
    DIVIDEND_YIELD_FIELD,
    "Volume",
    "ticker",
    "lseg_universe",
    "lseg_ric",
]

# Class-share tickers and other symbols where ticker-to-RIC guessing is commonly wrong.
# Constituency files remain the primary source; these aliases are extra protection.
MANUAL_RIC_ALIASES: Dict[str, List[str]] = {
    "BFB": ["BFb.N", "BFb"],
    "BRKB": ["BRKb.N", "BRKb"],
}


@dataclass(frozen=True)
class RepairTarget:
    symbol: str
    start_date: date
    end_date: date
    reason: str


def lseg_app_key() -> Optional[str]:
    return pipeline.lseg_app_key()


def open_lseg_session() -> None:
    if ld is None:
        raise RuntimeError(f"lseg.data import failed: {LSEG_IMPORT_ERROR}")
    key = lseg_app_key()
    if not key:
        raise RuntimeError(pipeline.lseg_unavailable_reason(key))
    ld.open_session(app_key=key)


def close_lseg_session() -> None:
    if ld is not None:
        try:
            ld.close_session()
        except Exception:
            pass


def issue_symbols_from_report(path: Path) -> List[str]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    required = {"severity", "area", "item"}
    if not required.issubset(df.columns):
        raise ValueError(f"Report missing required columns {sorted(required)}: {path}")
    stocks = df[(df["severity"] == "error") & (df["area"] == "stocks")].copy()
    symbols = [
        str(value).strip().upper()
        for value in stocks["item"].dropna().tolist()
        if str(value).strip()
    ]
    return sorted(set(symbols))


def load_membership_windows() -> Dict[str, List[pipeline.MembershipWindow]]:
    snapshots = pipeline.load_constituency_snapshots(pipeline.CONSTITUENCY_ROOT)
    if not snapshots:
        return {}
    return pipeline.build_membership_windows(
        snapshots,
        pipeline.STOCK_START_DATE,
        pipeline.STOCK_END_DATE,
        pipeline.LIMIT_SYMBOLS,
    )


def stock_file_bounds(path: Path) -> Optional[Tuple[date, date]]:
    return pipeline.stock_file_date_bounds(path)


def repair_range_for_symbol(
    symbol: str,
    windows: Dict[str, List[pipeline.MembershipWindow]],
    output_root: Path,
    overwrite: bool,
) -> Optional[Tuple[date, date, str]]:
    path = output_root / f"{symbol}_stock_data.csv"
    fetch_range = None
    if symbol in windows:
        fetch_range = pipeline.symbol_fetch_range(
            windows[symbol],
            pipeline.STOCK_START_DATE,
            pipeline.STOCK_END_DATE,
        )
    if fetch_range is None:
        fetch_range = (pipeline.STOCK_START_DATE, pipeline.STOCK_END_DATE)

    start_date, end_date = fetch_range
    if overwrite or not path.exists():
        return start_date, end_date, "missing_or_overwrite"

    if not pipeline.stock_file_has_dividend_yield(path):
        return start_date, end_date, "missing_dividend_yield"

    bounds = stock_file_bounds(path)
    if bounds is None:
        return start_date, end_date, "unreadable_or_empty"

    existing_start, existing_end = bounds
    if existing_start <= start_date and existing_end >= end_date:
        return None

    if existing_start <= start_date and existing_end < end_date:
        missing_start = pipeline.first_weekday_on_or_after(existing_end + timedelta(days=1))
        if missing_start <= end_date:
            return missing_start, end_date, "append_tail"
        return None

    # If the file starts too late or has a gap around the expected window, refetch the full
    # membership span for that symbol. This is safer than trying to stitch two partial slices.
    return start_date, end_date, "refetch_full_span"


def build_targets(
    symbols: Sequence[str],
    windows: Dict[str, List[pipeline.MembershipWindow]],
    output_root: Path,
    overwrite: bool,
) -> List[RepairTarget]:
    targets: List[RepairTarget] = []
    for symbol in sorted(set(s.upper() for s in symbols if s.strip())):
        repair_range = repair_range_for_symbol(symbol, windows, output_root, overwrite)
        if repair_range is None:
            continue
        start_date, end_date, reason = repair_range
        targets.append(RepairTarget(symbol, start_date, end_date, reason))
    return targets


def clean_ric(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    ric = str(value).strip()
    if not ric:
        return None
    return ric


def class_share_variants(symbol: str) -> List[str]:
    variants: List[str] = []
    if symbol in {"BFB", "BRKB"}:
        match = re.fullmatch(r"([A-Z]+)([AB])", symbol)
    else:
        match = None
    if match is not None:
        root, share_class = match.groups()
        variants.extend(
            [
                f"{root}{share_class.lower()}.N",
                f"{root}{share_class.lower()}.OQ",
                f"{root}.{share_class}.N",
                f"{root}/{share_class}.N",
            ]
        )
    return variants


def generic_ric_candidates(symbol: str) -> List[str]:
    if "." in symbol:
        root, share_class = symbol.split(".", 1)
        return [f"{root}{share_class.lower()}.N", f"{root}{share_class.lower()}.OQ"]
    return [
        f"{symbol}.N",
        f"{symbol}.OQ",
        f"{symbol}.O",
        symbol,
    ]


def constituency_ric_candidates(symbol: str) -> List[str]:
    candidates: List[str] = []
    for path in pipeline.iter_constituency_files(pipeline.CONSTITUENCY_ROOT):
        try:
            df = pd.read_csv(path, usecols=lambda col: str(col).strip().lower() in {"ticker", "original_ric"})
        except Exception:
            continue
        if df.empty or "ticker" not in {str(col).strip().lower() for col in df.columns}:
            continue
        columns = {str(col).strip().lower(): col for col in df.columns}
        ticker_col = columns.get("ticker")
        ric_col = columns.get("original_ric")
        if ticker_col is None or ric_col is None:
            continue
        rows = df[df[ticker_col].astype(str).str.upper().str.strip() == symbol]
        for value in rows[ric_col].tolist():
            ric = clean_ric(value)
            if ric:
                candidates.append(ric)
    return candidates


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def ric_candidates(symbol: str) -> List[str]:
    return unique_preserve_order(
        list(MANUAL_RIC_ALIASES.get(symbol, []))
        + constituency_ric_candidates(symbol)
        + class_share_variants(symbol)
        + generic_ric_candidates(symbol)
    )


def response_to_dataframe(response: object) -> pd.DataFrame:
    if response is None:
        return pd.DataFrame()
    if isinstance(response, pd.DataFrame):
        return response.copy()
    data = getattr(response, "data", None)
    if data is not None:
        df = getattr(data, "df", None)
        if isinstance(df, pd.DataFrame):
            return df.copy()
    return pd.DataFrame(response)


def fetch_lseg_history(ric: str, start_date: date, end_date: date) -> pd.DataFrame:
    if ld is None:
        return pd.DataFrame()
    response = ld.get_history(
        universe=ric,
        fields=LSEG_HISTORY_FIELDS,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        interval="daily",
    )
    return response_to_dataframe(response)


def fetch_yfinance_dividends(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError(f"yfinance import failed: {YFINANCE_IMPORT_ERROR}")

    fetch_start = start_date - timedelta(days=370)
    downloaded = yf.download(
        tickers=pipeline.yf_symbol_for_ticker(symbol),
        start=fetch_start.isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),
        group_by="ticker",
        auto_adjust=False,
        actions=True,
        progress=False,
        threads=False,
    )
    if downloaded.empty:
        return pd.DataFrame(columns=["Date", DIVIDENDS_FIELD])

    raw = pipeline.yfinance_symbol_frame(downloaded, pipeline.yf_symbol_for_ticker(symbol), 1)
    if raw.empty:
        return pd.DataFrame(columns=["Date", DIVIDENDS_FIELD])

    out = raw.copy().reset_index()
    if "Date" not in out.columns:
        out = out.rename(columns={out.columns[0]: "Date"})
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"])

    dividend_col = next((col for col in out.columns if str(col).lower() == "dividends"), None)
    if dividend_col is None:
        out[DIVIDENDS_FIELD] = 0.0
    else:
        out[DIVIDENDS_FIELD] = pd.to_numeric(out[dividend_col], errors="coerce").fillna(0.0)

    return out[["Date", DIVIDENDS_FIELD]].sort_values("Date").reset_index(drop=True)


def date_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if "date" in str(col).lower():
            return str(col)
    return None


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "_".join(str(part).strip() for part in col if str(part).strip())
            for col in out.columns
        ]
    else:
        out.columns = [str(col).strip() for col in out.columns]
    return out


def find_field_column(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    normalized = {
        str(col).lower().replace(" ", "").replace(".", "").replace("_", ""): col
        for col in df.columns
    }
    for name in names:
        key = name.lower().replace(" ", "").replace(".", "").replace("_", "")
        if key in normalized:
            return normalized[key]
    for col in df.columns:
        lowered = str(col).lower()
        if any(name.lower() in lowered for name in names):
            return str(col)
    return None


def normalize_lseg_history(symbol: str, ric: str, raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    out = flatten_columns(raw.reset_index())
    dcol = date_column(out)
    if dcol is None:
        dcol = str(out.columns[0])

    column_map = {
        "Open": find_field_column(out, ["TR.PriceOpen", "Price Open", "Open"]),
        "High": find_field_column(out, ["TR.PriceHigh", "Price High", "High"]),
        "Low": find_field_column(out, ["TR.PriceLow", "Price Low", "Low"]),
        PRICE_CLOSE_FIELD: find_field_column(out, ["TR.PriceClose", "Price Close", "Close"]),
        "Volume": find_field_column(out, ["TR.Volume", "Volume"]),
    }
    if column_map[PRICE_CLOSE_FIELD] is None:
        return pd.DataFrame()

    result = pd.DataFrame()
    result["Date"] = pd.to_datetime(out[dcol], errors="coerce").dt.strftime("%Y-%m-%d")
    for output_col, source_col in column_map.items():
        if source_col is None:
            result[output_col] = pd.NA
        else:
            result[output_col] = pd.to_numeric(out[source_col], errors="coerce")

    result["Adj Close"] = result[PRICE_CLOSE_FIELD]
    result[DIVIDENDS_FIELD] = 0.0
    result[DIVIDEND_YIELD_FIELD] = 0.0
    result["ticker"] = symbol
    result["lseg_universe"] = LSEG_SOURCE
    result["lseg_ric"] = ric
    result = result.dropna(subset=["Date", PRICE_CLOSE_FIELD])
    result = result[result[PRICE_CLOSE_FIELD] > MIN_CLOSE_PRICE].copy()
    result = result.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    return result[OUTPUT_COLUMNS].reset_index(drop=True)


def attach_dividend_yields(symbol: str, df: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).sort_values("Date").copy()

    try:
        dividends = fetch_yfinance_dividends(symbol, start_date, end_date)
    except Exception as exc:
        print(f"  dividend fetch failed for {symbol}; using 0.0 yields: {exc}")
        dividends = pd.DataFrame(columns=["Date", DIVIDENDS_FIELD])

    if not dividends.empty:
        dividends["Date"] = pd.to_datetime(dividends["Date"], errors="coerce")
        dividends = dividends.dropna(subset=["Date"])
        dividends = dividends.groupby("Date", as_index=False)[DIVIDENDS_FIELD].sum()
        out = out.drop(columns=[DIVIDENDS_FIELD], errors="ignore").merge(dividends, on="Date", how="left")

    if DIVIDENDS_FIELD not in out.columns:
        out[DIVIDENDS_FIELD] = 0.0
    out[DIVIDENDS_FIELD] = pd.to_numeric(out[DIVIDENDS_FIELD], errors="coerce").fillna(0.0)

    if dividends.empty:
        trailing_dividends = pd.Series(0.0, index=out.index)
    else:
        dividend_dates = dividends["Date"]
        dividend_values = pd.to_numeric(dividends[DIVIDENDS_FIELD], errors="coerce").fillna(0.0)
        trailing_dividends = out["Date"].apply(
            lambda value: dividend_values[
                (dividend_dates > value - pd.Timedelta(days=365))
                & (dividend_dates <= value)
            ].sum()
        )
    close = pd.to_numeric(out[PRICE_CLOSE_FIELD], errors="coerce")
    out[DIVIDEND_YIELD_FIELD] = trailing_dividends / close
    out[DIVIDEND_YIELD_FIELD] = (
        pd.to_numeric(out[DIVIDEND_YIELD_FIELD], errors="coerce")
        .replace([float("inf"), float("-inf")], 0.0)
        .fillna(0.0)
        .clip(lower=0.0)
    )
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    return out


def merge_with_existing(output_path: Path, new_rows: pd.DataFrame, overwrite: bool) -> pd.DataFrame:
    if overwrite or not output_path.exists():
        return new_rows
    try:
        existing = pd.read_csv(output_path)
    except Exception:
        existing = pd.DataFrame()
    if existing.empty:
        return new_rows
    merged = pd.concat([existing, new_rows], ignore_index=True, sort=False)
    if "Date" not in merged.columns:
        return new_rows
    merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce")
    merged = merged.dropna(subset=["Date"])
    merged = merged.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    merged["Date"] = merged["Date"].dt.strftime("%Y-%m-%d")
    for col in OUTPUT_COLUMNS:
        if col not in merged.columns:
            merged[col] = pd.NA
    extra_cols = [col for col in merged.columns if col not in OUTPUT_COLUMNS]
    return merged[OUTPUT_COLUMNS + extra_cols].reset_index(drop=True)


def save_stock_history(output_root: Path, symbol: str, df: pd.DataFrame, overwrite: bool) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{symbol}_stock_data.csv"
    merged = merge_with_existing(output_path, df, overwrite)
    merged_dates = pd.to_datetime(merged["Date"], errors="coerce").dropna()
    if not merged_dates.empty:
        merged = attach_dividend_yields(
            symbol,
            merged,
            merged_dates.min().date(),
            merged_dates.max().date(),
        )
    merged.to_csv(output_path, index=False)
    return output_path


def symbols_from_stock_root(root: Path) -> List[str]:
    return sorted(
        path.name.replace("_stock_data.csv", "").upper()
        for path in root.glob("*_stock_data.csv")
    )


def backfill_dividend_yields(
    output_root: Path,
    symbols: Sequence[str],
    dry_run: bool,
) -> Tuple[int, List[Dict[str, object]]]:
    repaired = 0
    failures: List[Dict[str, object]] = []
    for symbol in sorted(set(s.upper() for s in symbols if s.strip())):
        path = output_root / f"{symbol}_stock_data.csv"
        if not path.exists():
            failures.append({"symbol": symbol, "error": f"missing stock file: {path}"})
            continue

        try:
            df = pd.read_csv(path)
        except Exception as exc:
            failures.append({"symbol": symbol, "error": f"unreadable stock file: {exc}"})
            continue

        if df.empty or "Date" not in df.columns or PRICE_CLOSE_FIELD not in df.columns:
            failures.append({"symbol": symbol, "error": "missing Date or close column"})
            continue

        dates = pd.to_datetime(df["Date"], errors="coerce").dropna()
        if dates.empty:
            failures.append({"symbol": symbol, "error": "no parseable Date values"})
            continue

        if dry_run:
            print(f"  would backfill dividend_yield for {symbol} rows={len(df)}")
            continue

        try:
            repaired_df = attach_dividend_yields(
                symbol,
                df,
                dates.min().date(),
                dates.max().date(),
            )
        except Exception as exc:
            failures.append({"symbol": symbol, "error": str(exc)})
            continue

        try:
            repaired_df.to_csv(path, index=False)
        except Exception as exc:
            failures.append({"symbol": symbol, "error": f"could not write stock file: {exc}"})
            continue

        repaired += 1
        nonzero_yields = int((pd.to_numeric(repaired_df[DIVIDEND_YIELD_FIELD], errors="coerce").fillna(0.0) > 0.0).sum())
        print(f"  backfilled {symbol} rows={len(repaired_df)} nonzero_yield_rows={nonzero_yields}")

    return repaired, failures


def repair_target(target: RepairTarget, output_root: Path, overwrite: bool) -> Tuple[bool, str, str, int]:
    last_error = ""
    for ric in ric_candidates(target.symbol):
        try:
            raw = fetch_lseg_history(ric, target.start_date, target.end_date)
            normalized = normalize_lseg_history(target.symbol, ric, raw)
        except Exception as exc:
            last_error = f"{ric}: {exc}"
            continue

        if normalized.empty:
            last_error = f"{ric}: no usable rows"
            continue

        output_path = save_stock_history(output_root, target.symbol, normalized, overwrite)
        return True, ric, str(output_path), len(normalized)

    return False, "", last_error or "no RIC candidates returned usable rows", 0


def write_failures(path: Path, failures: List[Dict[str, object]]) -> None:
    if failures:
        pd.DataFrame(failures).to_csv(path, index=False)
    elif path.exists():
        path.unlink()


def parse_symbol_list(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [part.strip().upper() for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair missing or stale clean-stock CSVs with LSEG historical price data."
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORT_PATH,
        help="Validation report to read stock errors from.",
    )
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbols to repair. Defaults to stock errors in the validation report.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=pipeline.STOCK_OUTPUT_ROOT,
        help="Directory where *_stock_data.csv files are written.",
    )
    parser.add_argument(
        "--failures-file",
        type=Path,
        default=FAILURES_PATH,
        help="CSV path for symbols that could not be repaired.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace each target file instead of appending missing rows where possible.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned targets and RIC candidates without opening an LSEG session.",
    )
    parser.add_argument(
        "--backfill-dividend-yield",
        action="store_true",
        help="Rewrite Dividends and trailing 365-day dividend_yield in existing stock CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline.load_dotenv()

    symbols = parse_symbol_list(args.symbols)
    if args.backfill_dividend_yield:
        if not symbols:
            symbols = symbols_from_stock_root(args.output_root)
        if not symbols:
            print(f"No stock files found in {args.output_root}")
            return
        print(f"Backfilling dividend_yield for {len(symbols)} stock files in {args.output_root}")
        repaired, failures = backfill_dividend_yields(args.output_root, symbols, args.dry_run)
        if not args.dry_run:
            write_failures(args.failures_file, failures)
        print(f"Dividend-yield backfill complete. repaired={repaired} failed={len(failures)}")
        if failures:
            print(f"Failures written to {args.failures_file}")
        return

    if not symbols:
        symbols = issue_symbols_from_report(args.report)
    if not symbols:
        print(f"No stock errors found in {args.report}")
        return

    windows = load_membership_windows()
    targets = build_targets(symbols, windows, args.output_root, args.overwrite)
    if not targets:
        print("No stock repairs needed; target files already cover their expected ranges.")
        return

    print(f"Planned repairs: {len(targets)}")
    for target in targets:
        candidates = ", ".join(ric_candidates(target.symbol)[:6])
        print(
            f"  {target.symbol}: {target.start_date} -> {target.end_date} "
            f"({target.reason}); RIC candidates: {candidates}"
        )

    if args.dry_run:
        return

    failures: List[Dict[str, object]] = []
    repaired = 0
    open_lseg_session()
    try:
        for target in targets:
            ok, ric, detail, rows = repair_target(target, args.output_root, args.overwrite)
            if ok:
                repaired += 1
                print(f"  saved {target.symbol} rows={rows} ric={ric} -> {detail}")
            else:
                print(f"  failed {target.symbol}: {detail}")
                failures.append(
                    {
                        "symbol": target.symbol,
                        "start_date": target.start_date.isoformat(),
                        "end_date": target.end_date.isoformat(),
                        "reason": target.reason,
                        "error": detail,
                        "ric_candidates": ";".join(ric_candidates(target.symbol)),
                    }
                )
    finally:
        close_lseg_session()

    write_failures(args.failures_file, failures)
    print(f"Repair complete. repaired={repaired} failed={len(failures)}")
    if failures:
        print(f"Failures written to {args.failures_file}")


if __name__ == "__main__":
    main()
