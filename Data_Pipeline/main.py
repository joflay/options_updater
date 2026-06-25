import concurrent.futures
import math
import multiprocessing as mp
import os
import shutil
import sys
import time
import warnings
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

import flatfile_iv
import flat_file

try:
    import lseg.data as ld
    LSEG_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # LSEG is optional on local machines.
    ld = None
    LSEG_IMPORT_ERROR = exc

try:
    import yfinance as yf
except Exception as exc:  # yfinance is required for stock prices.
    raise RuntimeError("yfinance is required to run the stock pipeline") from exc

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"lseg\.data\._tools\._dataframe",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = Path(os.environ.get("OPTIONS_DATASET_ROOT", "/srv/data/options_model_features"))
SP500_INDEX_RIC = ".SPX"

def cgroup_cpu_quota_count() -> Optional[int]:
    cpu_max_path = Path("/sys/fs/cgroup/cpu.max")
    if cpu_max_path.exists():
        try:
            quota, period = cpu_max_path.read_text().strip().split()[:2]
            if quota != "max":
                quota_value = int(quota)
                period_value = int(period)
                if quota_value > 0 and period_value > 0:
                    return max(1, math.ceil(quota_value / period_value))
        except (OSError, ValueError):
            pass

    quota_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota_path.exists() and period_path.exists():
        try:
            quota_value = int(quota_path.read_text().strip())
            period_value = int(period_path.read_text().strip())
            if quota_value > 0 and period_value > 0:
                return max(1, math.ceil(quota_value / period_value))
        except (OSError, ValueError):
            pass

    return None

def available_cpu_count() -> int:
    quota_count = cgroup_cpu_quota_count()
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity_count = len(os.sched_getaffinity(0))
            return min(affinity_count, quota_count) if quota_count is not None else affinity_count
        except OSError:
            pass
    return quota_count or os.cpu_count() or 1

CONSTITUENCY_ROOT = Path(os.environ.get("OPTIONS_CONSTITUENCY_ROOT", str(PROJECT_ROOT / "Constituency")))
STOCK_OUTPUT_ROOT = Path(os.environ.get("OPTIONS_CLEAN_STOCK_ROOT", str(PROJECT_ROOT / "clean stocks")))
OPTIONS_INPUT_ROOT = flatfile_iv.INPUT_ROOT
OPTIONS_OUTPUT_ROOT = flatfile_iv.OUTPUT_ROOT
DAILY_FEATURES_ROOT = flatfile_iv.DAILY_FEATURES_ROOT
CACHE_OUTPUT = flatfile_iv.CACHE_OUTPUT
FINAL_FEATURES_DIR = flatfile_iv.FINAL_FEATURES_DIR
LSEG_DEBUG_ROOT = Path(os.environ.get("OPTIONS_LSEG_DEBUG_ROOT", str(PROJECT_ROOT / "lseg_constituency_debug")))

OVERWRITE_CONSTITUENCY = False
OVERWRITE_STOCKS = False
# Keep per-day IV outputs as restart checkpoints; flatfile_iv reprocesses stale schemas.
OVERWRITE_OPTIONS = False
REQUIRE_COMPLETE_OPTION_FLATFILES = True
DEFAULT_OPTIONS_WORKERS = 4


def positive_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return parsed


LIMIT_SYMBOLS: Optional[int] = None
LIMIT_OPTION_FILES: Optional[int] = None
OPTIONS_WORKERS = positive_int_env("OPTIONS_WORKERS", DEFAULT_OPTIONS_WORKERS)
YFINANCE_BATCH_SIZE = 10
YFINANCE_RATE_LIMIT_SLEEP_SECONDS = 5 * 60
MIN_CLOSE_PRICE = 5.0
PRICE_CLOSE_FIELD = "TR.CLOSEPRICE(Adjusted=0)"
DIVIDEND_YIELD_FIELD = "dividend_yield"
PROGRESS_BAR_WIDTH = 28
DEBUG_LSEG_CONSTITUENCY = True


@dataclass
# AGENTS: Mission-critical runtime state; only used by main() to summarize the pipeline.
class PipelineStats:
    constituency_files: int = 0
    constituency_symbols: int = 0
    stock_downloaded: int = 0
    stock_skipped_existing: int = 0
    stock_failed: int = 0
    option_flatfiles_downloaded: int = 0
    option_flatfiles_skipped_existing: int = 0
    option_flatfiles_failed: int = 0
    options_files: int = 0
    option_feature_rows: int = 0


@dataclass(frozen=True)
# AGENTS: Mission-critical runtime model for S&P membership intervals used by stock and option filtering.
class MembershipWindow:
    start_date: date
    end_date: date


# AGENTS: Runtime support for console progress only; safe to replace with logging/tqdm later.
class ProgressBar:
# AGENTS: Runtime support; constructs the lightweight progress display used in each phase.
    def __init__(self, total: int, label: str, width: int = PROGRESS_BAR_WIDTH) -> None:
        self.total = max(0, total)
        self.label = label
        self.width = max(10, width)
        self.current = 0
        self.enabled = self.total > 0
        self.started = False
        if self.enabled:
            self.render()

# AGENTS: Runtime support; called by each phase loop to update progress.
    def advance(self, detail: str = "") -> None:
        if not self.enabled:
            return
        self.current = min(self.current + 1, self.total)
        self.render(detail)

# AGENTS: Runtime support; presentation only, not part of data correctness.
    def render(self, detail: str = "") -> None:
        filled = int(self.width * self.current / self.total) if self.total else self.width
        bar = "#" * filled + "-" * (self.width - filled)
        pct = int(100 * self.current / self.total) if self.total else 100
        suffix = f" {detail}" if detail else ""
        prefix = "\r" if self.started else ""
        print(f"{prefix}{self.label} [{bar}] {self.current}/{self.total} {pct:3d}%{suffix}", end="", flush=True)
        self.started = True
        if self.current >= self.total:
            print()

# AGENTS: Runtime support for ETA/status text in long-running pipeline phases.
def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"

# AGENTS: Mission-critical startup helper; loads local secrets/config before LSEG/FRED/API code runs.
def load_dotenv() -> None:
    for path in [PROJECT_ROOT / ".env", Path(__file__).resolve().parent / ".env", PROJECT_ROOT.parent / ".env"]:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

# AGENTS: Mission-critical if pulling fresh constituency data; otherwise supports the fallback message.
def lseg_app_key() -> Optional[str]:
    return (
        os.environ.get("LSEG_APP_KEY")
        or os.environ.get("EIKON_APP_KEY")
        or os.environ.get("RDP_APP_KEY")
        or os.environ.get("APP_KEY")
    )

# AGENTS: Runtime diagnostics; explains why main() will reuse existing constituency files.
def lseg_unavailable_reason(key: Optional[str]) -> str:
    reasons: List[str] = []
    if ld is None:
        reasons.append(f"lseg.data import failed: {LSEG_IMPORT_ERROR}")
    if not key:
        reasons.append("no LSEG_APP_KEY/EIKON_APP_KEY/RDP_APP_KEY/APP_KEY found in .env or environment")
    return "; ".join(reasons) if reasons else "unknown"

# AGENTS: Mission-critical date helper; repeated concept also exists as two_years_ago() in other modules.
def subtract_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(month=2, day=28, year=value.year - years)


TODAY = date.today()
# AGENTS: Mission-critical date helper for stock/option end-date selection.
def latest_completed_weekday(value: date) -> date:
    candidate = value - timedelta(days=1)
    while not flat_file.is_business_day(candidate):
        candidate -= timedelta(days=1)
    return candidate

# AGENTS: Mission-critical date helper used when clipping stock fetch windows to trading weekdays.
def first_weekday_on_or_after(value: date) -> date:
    candidate = value
    while not flat_file.is_business_day(candidate):
        candidate += timedelta(days=1)
    return candidate

# AGENTS: Mission-critical date helper used when clipping stock fetch windows to trading weekdays.
def last_weekday_on_or_before(value: date) -> date:
    candidate = value
    while not flat_file.is_business_day(candidate):
        candidate -= timedelta(days=1)
    return candidate


CONSTITUENCY_END_DATE = TODAY
CONSTITUENCY_START_DATE = subtract_years(CONSTITUENCY_END_DATE, 3)
STOCK_END_DATE = latest_completed_weekday(TODAY)
STOCK_START_DATE = subtract_years(STOCK_END_DATE, 3)

# AGENTS: Mission-critical constituency schedule helper; duplicated in validation logic by design.
def add_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    if month == 12:
        max_day = 31
    else:
        max_day = (date(year, month + 1, 1) - timedelta(days=1)).day
    return date(year, month, min(value.day, max_day))

# AGENTS: Mission-critical constituency parser helper; normalizes RIC/ticker strings before membership windows.
def normalize_ticker(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    ticker = str(value).strip().upper()
    if not ticker:
        return None
    if "." in ticker:
        ticker = ticker.split(".", 1)[0]
    return ticker.replace("/", ".") or None

# AGENTS: Mission-critical LSEG parser helper; handles multi-value cells in constituency responses.
def cell_values(value: object) -> List[object]:
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if pd.isna(value):
        return []
    if isinstance(value, str) and (";" in value or "," in value):
        delimiter = ";" if ";" in value else ","
        return [part.strip() for part in value.split(delimiter) if part.strip()]
    return [value]

# AGENTS: Mission-critical LSEG parser adapter; probably only runs when an LSEG session is available.
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

# AGENTS: Mission-critical LSEG parser helper; probably only runs when pulling fresh constituency snapshots.
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

# AGENTS: Mission-critical LSEG parser; probably does not run if main() reuses existing constituency CSVs.
def extract_constituency_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ticker", "original_ric", "name"])

    out = flatten_columns(df.reset_index())
    ric_col = next(
        (
            col for col in out.columns
            if "constituentric" in col.replace(" ", "").lower()
            or str(col).strip().lower() in {"constituent ric", "ric"}
        ),
        None,
    )
    if ric_col is None:
        ric_col = next(
            (
                col for col in out.columns
                if str(col).strip().lower() in {"instrument", "instrument ric"}
            ),
            None,
        )
    ticker_col = next(
        (
            col for col in out.columns
            if "ticker" in str(col).lower()
            or "exchange ticker" in str(col).lower()
        ),
        None,
    )
    name_col = next((col for col in out.columns if "name" in str(col).lower()), None)
    if ticker_col is None and ric_col is None:
        return pd.DataFrame(columns=["ticker", "original_ric", "name"])

    records = []
    for row in out.to_dict("records"):
        ric_values = cell_values(row.get(ric_col)) if ric_col is not None else []
        ticker_values = cell_values(row.get(ticker_col)) if ticker_col is not None else []
        name_values = cell_values(row.get(name_col)) if name_col is not None else []
        item_count = max(len(ric_values), len(ticker_values), len(name_values), 1)

        for idx in range(item_count):
            ric = ric_values[idx] if idx < len(ric_values) else None
            ticker_source = ticker_values[idx] if idx < len(ticker_values) else ric
            ticker = normalize_ticker(ticker_source)
            if ticker is None:
                continue
            name = name_values[idx] if idx < len(name_values) else None
            records.append(
                {
                    "ticker": ticker,
                    "original_ric": "" if pd.isna(ric) else str(ric).strip(),
                    "name": "" if name is None or pd.isna(name) else str(name).strip(),
                }
            )

    if not records:
        return pd.DataFrame(columns=["ticker", "original_ric", "name"])
    result = pd.DataFrame(records).drop_duplicates(subset=["ticker"]).sort_values("ticker")
    return result.reset_index(drop=True)

# AGENTS: Mission-critical when LSEG is configured; otherwise skipped and existing constituency files are used.
def get_lseg_constituency_snapshot(index_ric: str, snapshot_date: date) -> pd.DataFrame:
    if ld is None:
        return pd.DataFrame(columns=["snapshot_date", "ticker", "original_ric", "name"])

    requests = [
        {
            "fields": ["TR.IndexConstituentRIC", "TR.IndexConstituentName", "TR.IndexConstituentTicker"],
            "parameters": {"SDate": snapshot_date.isoformat()},
        },
        {
            "fields": ["TR.IndexConstituentRIC", "TR.CommonName", "TR.ExchangeTicker"],
            "parameters": {"SDate": snapshot_date.isoformat()},
        },
        {
            "fields": ["TR.IndexConstituentRIC", "TR.IndexConstituentName"],
            "parameters": {"SDate": snapshot_date.isoformat()},
        },
        {
            "fields": [
                f"TR.IndexConstituentRIC(SDate={snapshot_date.isoformat()})",
                "TR.ExchangeTicker",
                "TR.CommonName",
            ],
            "parameters": None,
        },
    ]

    last_error: Optional[Exception] = None
    for request_index, request in enumerate(requests, start=1):
        try:
            kwargs = {
                "universe": [index_ric],
                "fields": request["fields"],
            }
            if request["parameters"] is not None:
                kwargs["parameters"] = request["parameters"]
            response = ld.get_data(**kwargs)
            raw = response_to_dataframe(response)
            rows = extract_constituency_rows(raw)
            if not rows.empty:
                rows.insert(0, "snapshot_date", snapshot_date.isoformat())
                return rows
            if DEBUG_LSEG_CONSTITUENCY and not raw.empty:
                LSEG_DEBUG_ROOT.mkdir(parents=True, exist_ok=True)
                debug_path = LSEG_DEBUG_ROOT / f"constituency_raw_{snapshot_date.isoformat()}_request_{request_index}.csv"
                raw.to_csv(debug_path, index=False)
                print(
                    "  LSEG returned data but parser found no tickers "
                    f"for {snapshot_date}; columns={list(raw.columns)} debug={debug_path}"
                )
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        print(f"  LSEG constituency snapshot failed for {snapshot_date}: {last_error}")
    return pd.DataFrame(columns=["snapshot_date", "ticker", "original_ric", "name"])

# AGENTS: Mission-critical when refreshing constituency snapshots; skipped if no LSEG session opens.
def pull_sp500_constituency(start_date: date, end_date: date) -> List[Path]:
    CONSTITUENCY_ROOT.mkdir(parents=True, exist_ok=True)
    snapshot_dates = sorted({start_date, end_date})
    cursor = date(start_date.year, start_date.month, 1)
    while cursor <= end_date:
        snapshot_dates.append(cursor)
        cursor = add_months(cursor, 3)
    snapshot_dates = sorted(set(snapshot_dates))

    written: List[Path] = []
    progress = ProgressBar(len(snapshot_dates), "Constituency")
    for snapshot_date in snapshot_dates:
        output_path = CONSTITUENCY_ROOT / f"constituency_{snapshot_date.isoformat()}.csv"
        if output_path.exists() and not OVERWRITE_CONSTITUENCY:
            written.append(output_path)
            print(f"  skip existing constituency snapshot {output_path}")
            progress.advance(snapshot_date.isoformat())
            continue
        rows = get_lseg_constituency_snapshot(SP500_INDEX_RIC, snapshot_date)
        if rows.empty:
            print(f"  no constituency rows for {snapshot_date}")
            progress.advance(snapshot_date.isoformat())
            continue
        rows.to_csv(output_path, index=False)
        written.append(output_path)
        print(f"  saved constituency {snapshot_date} rows={len(rows)} -> {output_path}")
        progress.advance(snapshot_date.isoformat())
    return written

# AGENTS: Mission-critical local-file helper; loads existing constituency snapshots for every main() run.
def iter_constituency_files(root: Path) -> List[Path]:
    return sorted(root.glob("constituency*.csv"))

# AGENTS: Mission-critical local-file helper; parses snapshot dates from constituency filenames.
def constituency_snapshot_date(path: Path) -> Optional[date]:
    try:
        return datetime.strptime(path.stem.replace("constituency_", ""), "%Y-%m-%d").date()
    except ValueError:
        return None

# AGENTS: Mission-critical local-file helper; repeated idea appears in stock_data_getter.py legacy workflow.
def extract_constituency_symbols(path: Path, seen_symbols: Set[str]) -> List[str]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if df.empty:
        return []

    ticker_col = next((c for c in df.columns if str(c).strip().lower() == "ticker"), None)
    if ticker_col is None:
        return []

    symbols: List[str] = []
    for raw_symbol in df[ticker_col]:
        symbol = normalize_ticker(raw_symbol)
        if not symbol or symbol in seen_symbols:
            continue
        seen_symbols.add(symbol)
        symbols.append(symbol)
    return symbols

# AGENTS: Mission-critical; main() cannot proceed without usable constituency snapshots from this function.
def load_constituency_snapshots(root: Path) -> List[Tuple[date, Set[str]]]:
    snapshots: List[Tuple[date, Set[str]]] = []
    for path in iter_constituency_files(root):
        snapshot_date = constituency_snapshot_date(path)
        if snapshot_date is None:
            continue
        symbols = set(extract_constituency_symbols(path, set()))
        if symbols:
            snapshots.append((snapshot_date, symbols))
    return sorted(snapshots, key=lambda item: item[0])

# AGENTS: Mission-critical; builds date-specific S&P membership filters used by stock and option phases.
def build_membership_windows(
    snapshots: List[Tuple[date, Set[str]]],
    dataset_start: date,
    dataset_end: date,
    limit_symbols: Optional[int],
) -> Dict[str, List[MembershipWindow]]:
    selected: Optional[Set[str]] = None
    if limit_symbols is not None:
        ordered = sorted({symbol for _, symbols in snapshots for symbol in symbols})
        selected = set(ordered[:limit_symbols])

    windows: Dict[str, List[MembershipWindow]] = {}
    for idx, (snapshot_date, symbols) in enumerate(snapshots):
        interval_start = max(snapshot_date, dataset_start)
        next_snapshot = snapshots[idx + 1][0] if idx + 1 < len(snapshots) else dataset_end + timedelta(days=1)
        interval_end = min(next_snapshot - timedelta(days=1), dataset_end)
        if interval_start > interval_end:
            continue
        for symbol in symbols:
            if selected is not None and symbol not in selected:
                continue
            symbol_windows = windows.setdefault(symbol, [])
            if symbol_windows and symbol_windows[-1].end_date + timedelta(days=1) >= interval_start:
                previous = symbol_windows[-1]
                symbol_windows[-1] = MembershipWindow(previous.start_date, max(previous.end_date, interval_end))
            else:
                symbol_windows.append(MembershipWindow(interval_start, interval_end))
    return windows

# AGENTS: Mission-critical option filter; determines which underlyings are valid on each trade date.
def active_symbols_on_date(membership_windows: Dict[str, List[MembershipWindow]], trade_date: date) -> Set[str]:
    return {
        symbol for symbol, windows in membership_windows.items()
        if any(window.start_date <= trade_date <= window.end_date for window in windows)
    }

# AGENTS: Mission-critical stock helper; lists all symbols requiring stock history.
def all_membership_symbols(membership_windows: Dict[str, List[MembershipWindow]]) -> List[str]:
    return sorted(membership_windows)

# AGENTS: Mission-critical stock helper; clips each symbol to its active S&P membership date range.
def symbol_fetch_range(
    windows: List[MembershipWindow],
    start_date: date,
    end_date: date,
) -> Optional[Tuple[date, date]]:
    clipped = [
        MembershipWindow(max(window.start_date, start_date), min(window.end_date, end_date))
        for window in windows
        if max(window.start_date, start_date) <= min(window.end_date, end_date)
    ]
    if not clipped:
        return None
    range_start = first_weekday_on_or_after(min(window.start_date for window in clipped))
    range_end = last_weekday_on_or_before(max(window.end_date for window in clipped))
    if range_start > range_end:
        return None
    return range_start, range_end

# AGENTS: Probably unused by main(); superseded by missing_stock_fetch_range(), but useful as a simple predicate.
def stock_file_covers_range(path: Path, start_date: date, end_date: date) -> bool:
    bounds = stock_file_date_bounds(path)
    return bounds is not None and bounds[0] <= start_date and bounds[1] >= end_date

# AGENTS: Mission-critical stock cache helper; used to avoid refetching already-covered stock files.
def stock_file_date_bounds(path: Path) -> Optional[Tuple[date, date]]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, usecols=lambda col: "date" in str(col).lower())
    except Exception:
        return None
    if df.empty:
        return None
    date_col = df.columns[0]
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna().dt.date
    if dates.empty:
        return None
    return dates.min(), dates.max()

# AGENTS: Mission-critical stock cache helper; forces legacy stock files to be refreshed with dividend yields.
def stock_file_has_dividend_yield(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path, usecols=lambda col: str(col) == DIVIDEND_YIELD_FIELD)
    except Exception:
        return False
    if DIVIDEND_YIELD_FIELD not in df.columns or df.empty:
        return False
    dividend_yield = pd.to_numeric(df[DIVIDEND_YIELD_FIELD], errors="coerce")
    return dividend_yield.notna().all() and (dividend_yield >= 0.0).all()

# AGENTS: Mission-critical stock fetch planner; decides whether to skip, append tail data, or refetch.
def missing_stock_fetch_range(path: Path, start_date: date, end_date: date) -> Optional[Tuple[date, date]]:
    start_date = first_weekday_on_or_after(start_date)
    end_date = last_weekday_on_or_before(end_date)
    if start_date > end_date:
        return None

    if OVERWRITE_STOCKS:
        return start_date, end_date

    if path.exists() and not stock_file_has_dividend_yield(path):
        return start_date, end_date

    bounds = stock_file_date_bounds(path)
    if bounds is None:
        return start_date, end_date

    existing_start, existing_end = bounds
    if existing_start <= start_date and existing_end >= end_date:
        return None

    if existing_start <= start_date and existing_end < end_date:
        tail_start = first_weekday_on_or_after(existing_end + timedelta(days=1))
        if tail_start <= end_date:
            return tail_start, end_date
        return None

    if existing_start > start_date and existing_end >= end_date:
        front_end = last_weekday_on_or_before(existing_start - timedelta(days=1))
        if start_date <= front_end:
            return start_date, front_end
        return None

    return start_date, end_date

# AGENTS: Mission-critical option date helper; discovers available raw flatfile coverage before and after downloads.
def available_option_date_range(input_root: Path) -> Tuple[Optional[date], Optional[date]]:
    dates: List[date] = []
    for path in flatfile_iv.iter_input_files(input_root):
        try:
            dates.append(flatfile_iv.trade_date_from_path(path))
        except ValueError:
            continue
    if not dates:
        return None, None
    return min(dates), max(dates)


def daily_csv_dates(root: Path) -> Set[date]:
    dates: Set[date] = set()
    if not root.exists():
        return dates
    for path in root.rglob("*.csv"):
        try:
            dates.add(datetime.strptime(path.stem, "%Y-%m-%d").date())
        except ValueError:
            continue
    return dates


def latest_atm_computed_date() -> Optional[date]:
    contract_dates = daily_csv_dates(OPTIONS_OUTPUT_ROOT)
    feature_dates = daily_csv_dates(DAILY_FEATURES_ROOT)
    completed_dates = contract_dates & feature_dates
    if not completed_dates:
        print(
            "No completed ATM_Normalized_Options dates found; "
            "option stage will only process raw dates in the active run window."
        )
        return None
    latest_date = max(completed_dates)
    print(
        f"ATM_Normalized_Options completed through {latest_date.isoformat()} "
        f"({len(completed_dates)} date(s) with contracts and features)"
    )
    return latest_date


# AGENTS: Mission-critical batching helper for yfinance downloads; repeated pattern exists in legacy stock_data_getter.py.
def chunked(values: List[str], size: int) -> Iterable[List[str]]:
    for idx in range(0, len(values), size):
        yield values[idx:idx + size]

# AGENTS: Mission-critical stock cleaning helper; enforces the minimum close price used by option feature inputs.
def apply_close_price_floor(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or PRICE_CLOSE_FIELD not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out[PRICE_CLOSE_FIELD] = pd.to_numeric(out[PRICE_CLOSE_FIELD], errors="coerce")
    out = out[out[PRICE_CLOSE_FIELD] > MIN_CLOSE_PRICE].copy()
    return out.reset_index(drop=True)

# AGENTS: Mission-critical stock writer; used by save_or_append_stock_history().
def save_stock_history(symbol: str, df: pd.DataFrame) -> Path:
    STOCK_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = STOCK_OUTPUT_ROOT / f"{symbol}_stock_data.csv"
    df.to_csv(output_path, index=False)
    return output_path

# AGENTS: Mission-critical stock writer; preserves existing files unless OVERWRITE_STOCKS is enabled.
def save_or_append_stock_history(symbol: str, df: pd.DataFrame) -> Path:
    output_path = STOCK_OUTPUT_ROOT / f"{symbol}_stock_data.csv"
    if output_path.exists() and not OVERWRITE_STOCKS:
        try:
            existing = pd.read_csv(output_path)
        except Exception:
            existing = pd.DataFrame()
        if not existing.empty:
            df = pd.concat([existing, df], ignore_index=True)
            date_col = next((col for col in df.columns if "date" in str(col).lower()), "Date")
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col])
            df = df.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")
            df[date_col] = df[date_col].dt.strftime("%Y-%m-%d")
    return save_stock_history(symbol, df)

# AGENTS: Mission-critical yfinance adapter; converts tickers like BRK.B to yfinance's BRK-B form.
def yf_symbol_for_ticker(symbol: str) -> str:
    return symbol.replace(".", "-")

# AGENTS: Mission-critical yfinance adapter; extracts one symbol frame from single- or multi-ticker downloads.
def yfinance_symbol_frame(downloaded: pd.DataFrame, yf_symbol: str, batch_size: int) -> pd.DataFrame:
    if downloaded.empty:
        return pd.DataFrame()
    if isinstance(downloaded.columns, pd.MultiIndex):
        if yf_symbol in downloaded.columns.get_level_values(0):
            return downloaded[yf_symbol].copy()
        if yf_symbol in downloaded.columns.get_level_values(-1):
            return downloaded.xs(yf_symbol, level=-1, axis=1).copy()
        return pd.DataFrame()
    return downloaded.copy() if batch_size == 1 else pd.DataFrame()

# AGENTS: Mission-critical stock cleaner; maps yfinance output into the column shape expected by flatfile_iv.py.
def normalize_yfinance_history(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.copy().reset_index()
    if "Date" not in out.columns:
        out = out.rename(columns={out.columns[0]: "Date"})
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"])

    close_col = next((col for col in out.columns if str(col).lower() == "close"), None)
    if close_col is None:
        close_col = next((col for col in out.columns if "close" in str(col).lower()), None)
    if close_col is None:
        return pd.DataFrame()

    out = out.sort_values("Date").copy()

    split_col = next((col for col in out.columns if str(col).lower() in {"stock splits", "stocksplits"}), None)
    splits = (
        pd.to_numeric(out[split_col], errors="coerce").fillna(0.0)
        if split_col is not None
        else pd.Series(0.0, index=out.index)
    )
    split_multiplier = splits.where(splits > 0.0, 1.0)
    future_split_factor = split_multiplier.iloc[::-1].cumprod().iloc[::-1] / split_multiplier

    out[PRICE_CLOSE_FIELD] = pd.to_numeric(out[close_col], errors="coerce") * future_split_factor
    out = out.dropna(subset=[PRICE_CLOSE_FIELD])

    for price_col in ["Open", "High", "Low"]:
        source_col = next((col for col in out.columns if str(col).lower() == price_col.lower()), None)
        if source_col is not None:
            out[price_col] = pd.to_numeric(out[source_col], errors="coerce") * future_split_factor

    dividend_col = next((col for col in out.columns if str(col).lower() == "dividends"), None)
    if dividend_col is None:
        out["Dividends"] = 0.0
        dividend_col = "Dividends"
    out[dividend_col] = pd.to_numeric(out[dividend_col], errors="coerce").fillna(0.0)
    trailing_dividends = (
        out.set_index("Date")[dividend_col]
        .rolling("365D", min_periods=1)
        .sum()
        .reindex(out["Date"])
        .to_numpy()
    )
    out[DIVIDEND_YIELD_FIELD] = trailing_dividends / out[PRICE_CLOSE_FIELD]
    out[DIVIDEND_YIELD_FIELD] = (
        pd.to_numeric(out[DIVIDEND_YIELD_FIELD], errors="coerce")
        .replace([float("inf"), float("-inf")], 0.0)
        .fillna(0.0)
        .clip(lower=0.0)
    )
    out["ticker"] = symbol
    out["lseg_universe"] = "yfinance-raw-split-safe"
    out["Stock Splits"] = splits
    return apply_close_price_floor(out)

# AGENTS: Mission-critical stock fetcher; this is main.py's active replacement for stock_data_getter.py.
def download_yfinance_batch(
    symbols: List[str],
    start_date: date,
    end_date: date,
) -> Dict[str, pd.DataFrame]:
    yf_to_symbol = {yf_symbol_for_ticker(symbol): symbol for symbol in symbols}
    downloaded = yf.download(
        tickers=list(yf_to_symbol),
        start=start_date.isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),
        group_by="ticker",
        auto_adjust=False,
        actions=True,
        progress=False,
        threads=True,
    )

    frames: Dict[str, pd.DataFrame] = {}
    for yf_symbol, symbol in yf_to_symbol.items():
        raw = yfinance_symbol_frame(downloaded, yf_symbol, len(symbols))
        frames[symbol] = normalize_yfinance_history(symbol, raw)
    return frames

# AGENTS: Runtime support; only matters when yfinance raises throttling errors.
def is_yfinance_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "rate limit" in message or "too many requests" in message or "429" in message

# AGENTS: Runtime support around the mission-critical fetch; may sleep for long periods on yfinance throttling.
def download_yfinance_batch_with_rate_limit_retry(
    symbols: List[str],
    start_date: date,
    end_date: date,
) -> Dict[str, pd.DataFrame]:
    attempt = 1
    while True:
        try:
            return download_yfinance_batch(symbols, start_date, end_date)
        except Exception as exc:
            if not is_yfinance_rate_limit_error(exc):
                raise
            print(
                "  yfinance rate limit hit for "
                f"{symbols}; retrying attempt {attempt + 1} in "
                f"{YFINANCE_RATE_LIMIT_SLEEP_SECONDS // 60} minutes"
            )
            time.sleep(YFINANCE_RATE_LIMIT_SLEEP_SECONDS)
            attempt += 1

# AGENTS: Mission-critical phase 3; ensures underlying stock close histories exist before IV calculations.
def run_stock_pipeline(
    membership_windows: Dict[str, List[MembershipWindow]],
    start_date: date,
    end_date: date,
) -> Dict[str, int]:
    stats = {"downloaded": 0, "skipped_existing": 0, "failed": 0}
    symbols_by_range: Dict[Tuple[date, date], List[str]] = {}

    for symbol in all_membership_symbols(membership_windows):
        fetch_range = symbol_fetch_range(membership_windows[symbol], start_date, end_date)
        if fetch_range is None:
            continue
        output_path = STOCK_OUTPUT_ROOT / f"{symbol}_stock_data.csv"
        missing_range = missing_stock_fetch_range(output_path, fetch_range[0], fetch_range[1])
        if missing_range is None:
            stats["skipped_existing"] += 1
            continue
        symbols_by_range.setdefault(missing_range, []).append(symbol)

    print(f"Fetching stock history with yfinance in batches of {YFINANCE_BATCH_SIZE}")
    total_batches = sum((len(symbols) + YFINANCE_BATCH_SIZE - 1) // YFINANCE_BATCH_SIZE for symbols in symbols_by_range.values())
    progress = ProgressBar(total_batches, "Stocks")
    for (range_start, range_end), symbols in sorted(symbols_by_range.items()):
        print(f"  range {range_start} -> {range_end}: {len(symbols)} symbols")
        for batch in chunked(symbols, YFINANCE_BATCH_SIZE):
            try:
                frames = download_yfinance_batch_with_rate_limit_retry(batch, range_start, range_end)
            except Exception as exc:
                stats["failed"] += len(batch)
                print(f"  yfinance batch failed {batch}: {exc}")
                progress.advance(f"{range_start}->{range_end}")
                continue

            for symbol in batch:
                frame = frames.get(symbol, pd.DataFrame())
                if frame.empty:
                    stats["failed"] += 1
                    print(f"  yfinance no data {symbol}")
                    continue
                saved_path = save_or_append_stock_history(symbol, frame)
                stats["downloaded"] += 1
                print(f"  saved {symbol} rows={len(frame)} -> {saved_path}")
            progress.advance(f"{range_start}->{range_end}")
    return stats

# AGENTS: Thin duplicate wrapper; probably removable if callers use run_options_pipeline() directly.
def run_option_calculation_pipeline(
    membership_windows: Dict[str, List[MembershipWindow]],
    start_date: date,
    end_date: date,
) -> Dict[str, int]:
    return run_options_pipeline(membership_windows, start_date, end_date)

# AGENTS: Mission-critical phase 4; processes raw OPRA files into IV/Greek outputs and final features.
def run_options_pipeline(
    membership_windows: Dict[str, List[MembershipWindow]],
    start_date: date,
    end_date: date,
) -> Dict[str, int]:
    files = flatfile_iv.iter_input_files(OPTIONS_INPUT_ROOT)
    files = flatfile_iv.filter_files_by_trade_date(files, start_date, end_date)
    if LIMIT_OPTION_FILES is not None:
        files = files[:LIMIT_OPTION_FILES]
    if not files:
        print(f"No option flatfiles found under {OPTIONS_INPUT_ROOT} for {start_date} -> {end_date}")
        return {"files": 0, "feature_rows": 0}

    latest_computed_date = latest_atm_computed_date()
    before_resume_filter = len(files)
    if latest_computed_date is not None:
        files = [
            path for path in files
            if flatfile_iv.trade_date_from_path(path) > latest_computed_date
        ]
        skipped = before_resume_filter - len(files)
        print(
            f"Skipping {skipped} raw option flatfile(s) on or before "
            f"latest completed ATM date {latest_computed_date.isoformat()}"
        )
    else:
        newest_raw_date = max(flatfile_iv.trade_date_from_path(path) for path in files)
        files = [
            path for path in files
            if flatfile_iv.trade_date_from_path(path) == newest_raw_date
        ]
        skipped = before_resume_filter - len(files)
        print(
            f"No ATM completion checkpoint found; skipping {skipped} older raw option "
            f"flatfile(s) and processing newest raw date only: {newest_raw_date.isoformat()}"
        )

    if not files:
        print("No new option flatfiles to process after ATM_Normalized_Options checkpoint filter")
        return {"files": 0, "feature_rows": 0}

    temp_final_features_dir = FINAL_FEATURES_DIR.with_name(f".{FINAL_FEATURES_DIR.name}.tmp")
    if temp_final_features_dir.exists():
        shutil.rmtree(temp_final_features_dir)

    close_cache = flatfile_iv.load_close_cache(CACHE_OUTPUT)
    feature_rows = 0

    task_inputs = []
    for path in files:
        trade_date = flatfile_iv.trade_date_from_path(path)
        active_underlyings = active_symbols_on_date(membership_windows, trade_date)
        if not active_underlyings:
            continue
        task_inputs.append((path, active_underlyings))

    print(f"Processing {len(task_inputs)} option flatfiles with date-specific S&P membership filters")
    risk_free_rates = flatfile_iv.load_risk_free_rates_for_files([path for path, _ in task_inputs])
    print(
        f"Loaded {len(risk_free_rates)} {flatfile_iv.FRED_RISK_FREE_SERIES_ID} "
        "trade-date risk-free rate values from FRED"
    )

    row_workers = OPTIONS_WORKERS if len(task_inputs) == 1 else 1
    file_workers = OPTIONS_WORKERS if len(task_inputs) > 1 else 1
    print(f"Option worker processes available: {OPTIONS_WORKERS}")
    print(f"Option file worker processes: {file_workers}")
    print(f"Option row worker processes per file: {row_workers}")

    tasks = []
    for path, active_underlyings in task_inputs:
        tasks.append(
            (
                path,
                OPTIONS_INPUT_ROOT,
                OPTIONS_OUTPUT_ROOT,
                OVERWRITE_OPTIONS,
                close_cache,
                False,
                "AAPL",
                active_underlyings,
                row_workers,
                risk_free_rates,
            )
        )
    flatfile_iv.preflight_process_file_task_outputs(tasks)

    progress = ProgressBar(len(tasks), "Options")
    phase_start_time = time.monotonic()

    def record_completed_file(completed: int, result: tuple[Path, pd.DataFrame, Dict[str, Optional[float]]]) -> None:
        nonlocal feature_rows
        output_path, features, close_cache_updates = result
        elapsed = time.monotonic() - phase_start_time
        remaining = max(0, len(tasks) - completed)
        eta_seconds = (elapsed / completed) * remaining if completed else 0.0
        close_cache.update(close_cache_updates)
        daily_features_path, daily_feature_rows = flatfile_iv.export_daily_feature_file(
            output_path,
            DAILY_FEATURES_ROOT,
            OPTIONS_OUTPUT_ROOT,
            reuse_existing=not OVERWRITE_OPTIONS,
        )
        flatfile_iv.save_close_cache(CACHE_OUTPUT, close_cache)
        feature_rows += daily_feature_rows
        eta_text = format_duration(eta_seconds)
        elapsed_text = format_duration(elapsed)
        print(
            f"[{completed}/{len(tasks)}] wrote {output_path} rows={len(features)} "
            f"features={daily_features_path} feature_rows={daily_feature_rows} "
            f"elapsed={elapsed_text} eta={eta_text}"
        )
        progress.advance(f"{output_path.name} ETA {eta_text}")

    if file_workers > 1:
        try:
            mp_context = mp.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=file_workers,
                mp_context=mp_context,
                max_tasks_per_child=1,
            ) as executor:
                for completed, result in enumerate(executor.map(flatfile_iv.process_file_task, tasks), start=1):
                    record_completed_file(completed, result)
        except BrokenProcessPool as exc:
            print(
                "Option worker process pool failed; rebuilding daily features "
                "from per-day IV checkpoints, then continuing sequentially. "
                f"Original error: {exc}"
            )
            feature_rows = 0
            for completed, task in enumerate(tasks, start=1):
                record_completed_file(completed, flatfile_iv.process_file_task(task))
    else:
        for completed, task in enumerate(tasks, start=1):
            record_completed_file(completed, flatfile_iv.process_file_task(task))

    flatfile_iv.export_final_feature_files(DAILY_FEATURES_ROOT, temp_final_features_dir)
    if temp_final_features_dir.exists():
        if FINAL_FEATURES_DIR.exists():
            shutil.rmtree(FINAL_FEATURES_DIR)
        temp_final_features_dir.replace(FINAL_FEATURES_DIR)

    return {"files": len(tasks), "feature_rows": feature_rows}

# AGENTS: Mission-critical phase 2; delegates raw OPRA downloads to flat_file.download_flat_files().
def run_option_flatfile_pipeline() -> Dict[str, int]:
    print("Ensuring OPRA option flatfiles exist after the ATM_Normalized_Options checkpoint")
    latest_computed_date = latest_atm_computed_date()
    if latest_computed_date is None:
        download_start = STOCK_END_DATE
        print(
            "No ATM_Normalized_Options checkpoint found; requesting newest stock-window "
            f"date only: {download_start.isoformat()}"
        )
    else:
        download_start = latest_computed_date + timedelta(days=1)
    if download_start > STOCK_END_DATE:
        print(
            f"ATM_Normalized_Options is already complete through {latest_computed_date}; "
            f"no raw option flatfile download needed through {STOCK_END_DATE}"
        )
        return {"downloaded": 0, "skipped_existing": 0, "failed": 0}

    stats = flat_file.download_flat_files(
        start_date=download_start,
        end_date=STOCK_END_DATE,
        output_dir=OPTIONS_INPUT_ROOT,
    )
    return {
        "downloaded": stats.downloaded,
        "skipped_existing": stats.skipped_existing,
        "failed": stats.failed,
    }

# AGENTS: Mission-critical entrypoint; the cleanup target should preserve everything reachable from here.
def main() -> None:
    load_dotenv()
    option_min_date, option_max_date = available_option_date_range(OPTIONS_INPUT_ROOT)
    options_start = max(option_min_date or STOCK_START_DATE, CONSTITUENCY_START_DATE)
    options_end = min(option_max_date or STOCK_END_DATE, CONSTITUENCY_END_DATE)
    stock_start = STOCK_START_DATE
    stock_end = STOCK_END_DATE

    stats = PipelineStats()
    session_opened = False
    key = lseg_app_key()
    if ld is not None and key:
        try:
            ld.open_session(app_key=key)
            session_opened = True
            print("LSEG session opened for constituency snapshots")
        except Exception as exc:
            print(f"LSEG unavailable; using existing constituency files. Reason: {exc}")
    else:
        print(f"LSEG unavailable; using existing constituency files. Reason: {lseg_unavailable_reason(key)}")

    try:
        if session_opened:
            print("Phase 1/5: Pulling S&P constituency snapshots")
            written = pull_sp500_constituency(CONSTITUENCY_START_DATE, CONSTITUENCY_END_DATE)
            stats.constituency_files = len(written)

        snapshots = load_constituency_snapshots(CONSTITUENCY_ROOT)
        if not snapshots:
            raise RuntimeError(
                f"No constituency snapshots found under {CONSTITUENCY_ROOT}. "
                f"LSEG status: {lseg_unavailable_reason(key)}. "
                "Install/configure LSEG or add constituency_YYYY-MM-DD.csv files."
            )

        print("Phase 2/5: Pulling all raw option flatfiles")
        flatfile_stats = run_option_flatfile_pipeline()
        stats.option_flatfiles_downloaded = flatfile_stats["downloaded"]
        stats.option_flatfiles_skipped_existing = flatfile_stats["skipped_existing"]
        stats.option_flatfiles_failed = flatfile_stats["failed"]
        if REQUIRE_COMPLETE_OPTION_FLATFILES and stats.option_flatfiles_failed > 0:
            raise RuntimeError(
                "Option flatfile pull reported failures, so IV/Greek calculations were not started. "
                "Fix the flatfile download issue or set REQUIRE_COMPLETE_OPTION_FLATFILES=False."
            )

        option_min_date, option_max_date = available_option_date_range(OPTIONS_INPUT_ROOT)
        options_start = max(option_min_date or STOCK_START_DATE, CONSTITUENCY_START_DATE)
        options_end = min(option_max_date or STOCK_END_DATE, CONSTITUENCY_END_DATE)
        stock_start = STOCK_START_DATE
        stock_end = STOCK_END_DATE

        membership_windows = build_membership_windows(
            snapshots,
            min(STOCK_START_DATE, options_start),
            max(STOCK_END_DATE, options_end),
            LIMIT_SYMBOLS,
        )
        stats.constituency_symbols = len(membership_windows)
        print(
            f"Built quarterly S&P membership windows for {stats.constituency_symbols} symbols "
        f"from {snapshots[0][0]} through {snapshots[-1][0]}"
        )

        print("Phase 3/5: Ensuring stock price coverage")
        print(f"Stock coverage target: {stock_start} through {stock_end}")
        stock_stats = run_stock_pipeline(membership_windows, stock_start, stock_end)
        stats.stock_downloaded = stock_stats["downloaded"]
        stats.stock_skipped_existing = stock_stats["skipped_existing"]
        stats.stock_failed = stock_stats["failed"]

        print("Phase 4/5: Running option IV/Greek calculations")
        option_stats = run_option_calculation_pipeline(membership_windows, options_start, options_end)
        stats.options_files = option_stats["files"]
        stats.option_feature_rows = option_stats["feature_rows"]
        print("Phase 5/5: Pipeline summary")
    finally:
        if session_opened:
            ld.close_session()

    print(
        "Pipeline finished. "
        f"constituency_files={stats.constituency_files} "
        f"constituency_symbols={stats.constituency_symbols} "
        f"stock_downloaded={stats.stock_downloaded} "
        f"stock_skipped_existing={stats.stock_skipped_existing} "
        f"stock_failed={stats.stock_failed} "
        f"option_flatfiles_downloaded={stats.option_flatfiles_downloaded} "
        f"option_flatfiles_skipped_existing={stats.option_flatfiles_skipped_existing} "
        f"option_flatfiles_failed={stats.option_flatfiles_failed} "
        f"options_files={stats.options_files} "
        f"option_feature_rows={stats.option_feature_rows}"
    )


if __name__ == "__main__":
    main()
