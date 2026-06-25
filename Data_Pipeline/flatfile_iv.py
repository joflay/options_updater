import argparse
import concurrent.futures
import grp
import multiprocessing as mp
import os
import pwd
import re
import shutil
import sys
from concurrent.futures.process import BrokenProcessPool
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from config import DATA_DIR
from interest_rate_getter import SERIES_ID as FRED_RISK_FREE_SERIES_ID
from interest_rate_getter import fetch_fred_series, load_fred_api_key, load_local_risk_free_rates, risk_free_rate_csv_path
from utils import american_option_price_crr, compute_iv, hypothetical_atm_greeks

# Run examples:
#   python3 Data_Pipeline/flatfile_iv.py
#   python3 Data_Pipeline/flatfile_iv.py --overwrite
#   python3 Data_Pipeline/flatfile_iv.py --limit-files 1
#   python3 Data_Pipeline/flatfile_iv.py --start-date 2024-04-01 --end-date 2024-04-01
#   python3 Data_Pipeline/flatfile_iv.py --underlyings AAPL AMZN
#   python3 Data_Pipeline/flatfile_iv.py --testing --test-stock AAPL
#   python3 Data_Pipeline/flatfile_iv.py --underlyings AAPL --testing --test-stock AAPL --limit-files 1 --overwrite
# Available CLI inputs:
#   --input-root PATH
#   --output-root PATH
#   --daily-features-root PATH
#   --cache-output PATH
#   --final-features-dir PATH
#   --limit-files INT
#   --start-date YYYY-MM-DD
#   --end-date YYYY-MM-DD
#   --testing
#   --test-stock SYMBOL
#   --underlyings SYMBOL [SYMBOL ...]
#   --workers INT
#   --overwrite


# I am using this for testing: python3 Data_Pipeline/flatfile_iv.py --start-date 2024-04-08 --end-date 2024-04-08 --underlyings NVDA --testing --test-stock NVDA --overwrite

SHARED_DATA_ROOT = Path(os.environ.get("OPTIONS_DATASET_ROOT", "/srv/data/options_model_features"))
SHARED_FLATFILE_ROOT = SHARED_DATA_ROOT / "FlatFiles" / "us_options_opra" / "day_aggs_v1"
ATM_NORMALIZED_ROOT = Path(os.environ.get("OPTIONS_ATM_NORMALIZED_ROOT", str(SHARED_DATA_ROOT / "ATM_Normalized_Options")))
INPUT_ROOT = Path(
    os.environ.get(
        "OPTIONS_FLATFILES_ROOT",
        str(
            SHARED_FLATFILE_ROOT
            if SHARED_DATA_ROOT.exists()
            else Path(DATA_DIR) / "flatfiles" / "us_options_opra" / "day_aggs_v1"
        ),
    )
)
OUTPUT_ROOT = Path(os.environ.get("OPTIONS_IV_ROOT", str(ATM_NORMALIZED_ROOT / "contracts")))
DAILY_FEATURES_ROOT = Path(os.environ.get("OPTIONS_DAILY_FEATURES_ROOT", str(ATM_NORMALIZED_ROOT / "features" / "day_aggs_v1")))
CACHE_OUTPUT = Path(os.environ.get("OPTIONS_CLOSE_CACHE", str(Path(DATA_DIR) / "features" / "underlying_close_cache.csv")))
CLEAN_STOCK_ROOT = Path(os.environ.get("OPTIONS_CLEAN_STOCK_ROOT", str(Path(__file__).resolve().parent.parent / "clean stocks")))
FINAL_FEATURES_DIR = Path(os.environ.get("OPTIONS_FINAL_FEATURES_DIR", str(Path(__file__).resolve().parent.parent / "final_features")))
SUPPORTED_UNDERLYING_RE = re.compile(r"^[A-Z]{1,5}$")
OPRA_TICKER_RE = re.compile(r"^O:([A-Z]{1,6})(\d{6})([CP])(\d{8})$")
DTE_MIN_DAYS = 30
DTE_MAX_DAYS = 90
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
DIVIDEND_YIELD_FIELD = "dividend_yield"
StockHistoryCache = Dict[str, Optional[Tuple[pd.Series, pd.Series, pd.Series]]]
IvTask = Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], str]
IvResult = Tuple[Optional[float], str]
GreekTask = Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], str]

# AGENTS: Standalone CLI only; main.py passes arguments directly through process_file_task() instead.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute implied volatilities for call and put options in OPRA flat files."
    )
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--daily-features-root", type=Path, default=DAILY_FEATURES_ROOT)
    parser.add_argument("--cache-output", type=Path, default=CACHE_OUTPUT)
    parser.add_argument("--final-features-dir", type=Path, default=FINAL_FEATURES_DIR)
    parser.add_argument("--limit-files", type=int, default=None)
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
        "--testing",
        action="store_true",
        help="Testing mode: force all underlying close lookups to use the test stock symbol.",
    )
    parser.add_argument(
        "--test-stock",
        default="AAPL",
        help="Stock symbol to use for underlying close lookups in testing mode.",
    )
    parser.add_argument(
        "--underlyings",
        nargs="+",
        default=None,
        help="Optional list of underlying symbols to process, e.g. --underlyings AAPL AMZN",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=positive_int_env("OPTIONS_WORKERS", DEFAULT_OPTIONS_WORKERS),
        help=(
            "Worker processes. Uses file-level parallelism when multiple input files "
            "are selected, otherwise row-level parallelism within the file."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="Rebuild daily and final feature CSVs from existing IV outputs without recalculating IV.",
    )
    parser.add_argument(
        "--overwrite-features",
        action="store_true",
        help="Force daily feature CSVs to be rebuilt from existing IV outputs.",
    )
    return parser.parse_args()

# AGENTS: Mission-critical file discovery; called by main.py and this module's standalone CLI.
def iter_input_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.csv.gz"))

# AGENTS: Mission-critical date parser; used to filter/process raw OPRA files.
def trade_date_from_path(path: Path) -> date:
    return datetime.strptime(path.stem.replace(".csv", ""), "%Y-%m-%d").date()

# AGENTS: Mission-critical file filter; called by main.py and this module's standalone CLI.
def filter_files_by_trade_date(
    files: List[Path],
    start_date: Optional[date],
    end_date: Optional[date],
) -> List[Path]:
    if start_date is None and end_date is None:
        return files

    filtered: List[Path] = []
    for path in files:
        trade_date = trade_date_from_path(path)
        if start_date is not None and trade_date < start_date:
            continue
        if end_date is not None and trade_date > end_date:
            continue
        filtered.append(path)
    return filtered

# AGENTS: Mission-critical risk-free-rate loader; main.py calls this before option processing.
def load_risk_free_rates_for_files(files: List[Path]) -> Dict[date, float]:
    trade_dates = sorted({trade_date_from_path(path) for path in files})
    if not trade_dates:
        return {}

    fetch_start_date = trade_dates[0] - timedelta(days=14)
    local_rates = load_local_risk_free_rates(fetch_start_date, trade_dates[-1], FRED_RISK_FREE_SERIES_ID)
    if not local_rates.empty:
        print(
            f"Loaded {len(local_rates)} {FRED_RISK_FREE_SERIES_ID} risk-free "
            f"observations from {risk_free_rate_csv_path()}"
        )
        rates = local_rates
    else:
        api_key = load_fred_api_key()
        rates = fetch_fred_series(FRED_RISK_FREE_SERIES_ID, fetch_start_date, trade_dates[-1], api_key)
    if rates.empty:
        raise ValueError(
            f"No {FRED_RISK_FREE_SERIES_ID} risk-free observations found from "
            f"{fetch_start_date.isoformat()} to {trade_dates[-1].isoformat()}."
        )

    rates = rates.dropna(subset=["date", "rate"]).copy()
    rates["date"] = pd.to_datetime(rates["date"], errors="coerce").dt.date
    observed_rates = sorted(
        (
            (row_date, float(rate))
            for row_date, rate in zip(rates["date"], rates["rate"])
            if row_date is not None and pd.notna(row_date) and pd.notna(rate)
        ),
        key=lambda item: item[0],
    )

    rates_by_date: Dict[date, float] = {}
    source_dates_by_trade_date: Dict[date, date] = {}
    rate_index = 0
    latest_rate: Optional[Tuple[date, float]] = None
    for trade_date_value in trade_dates:
        while rate_index < len(observed_rates) and observed_rates[rate_index][0] <= trade_date_value:
            latest_rate = observed_rates[rate_index]
            rate_index += 1
        if latest_rate is None:
            continue
        source_date, rate = latest_rate
        rates_by_date[trade_date_value] = rate
        source_dates_by_trade_date[trade_date_value] = source_date

    filled_dates = [
        trade_date_value
        for trade_date_value, source_date in source_dates_by_trade_date.items()
        if source_date != trade_date_value
    ]
    if filled_dates:
        sample = ", ".join(
            f"{trade_date_value.isoformat()}<-{source_dates_by_trade_date[trade_date_value].isoformat()}"
            for trade_date_value in filled_dates[:10]
        )
        if len(filled_dates) > 10:
            sample = f"{sample}, ..."
        print(
            f"Filled {len(filled_dates)} missing {FRED_RISK_FREE_SERIES_ID} risk-free "
            f"observation(s) with the most recent prior risk-free rate: {sample}"
        )

    missing_dates = [trade_date_value for trade_date_value in trade_dates if trade_date_value not in rates_by_date]
    if missing_dates:
        sample = ", ".join(trade_date_value.isoformat() for trade_date_value in missing_dates[:10])
        if len(missing_dates) > 10:
            sample = f"{sample}, ..."
        raise ValueError(
            f"Missing {FRED_RISK_FREE_SERIES_ID} risk-free observations for "
            f"{len(missing_dates)} trade date(s): {sample}"
        )

    return rates_by_date

# AGENTS: Mission-critical OPRA parser helper; normalizes adjusted/root symbols after ticker parsing.
def normalize_underlying_symbol(symbol: str) -> str:
    if not symbol:
        return symbol

    normalized = str(symbol).strip().upper()
    if normalized and normalized[-1].isdigit() and normalized[:-1].isalpha():
        normalized = normalized[:-1]
    return normalized


def parse_mixed_datetimes(values: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(values, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return values.apply(lambda value: pd.to_datetime(value, errors="coerce"))


# AGENTS: Mission-critical OPRA parser; extracts underlying, expiration, type, and strike from option tickers.
def parse_opra_option_ticker(ticker: str) -> Optional[Dict[str, Optional[str]]]:
    if not ticker:
        return None

    match = OPRA_TICKER_RE.match(str(ticker).strip().upper())
    if not match:
        return None

    underlying = normalize_underlying_symbol(match.group(1))
    expiry_raw = match.group(2)
    option_flag = match.group(3)
    strike_raw = match.group(4)

    try:
        expiration_date = datetime.strptime(expiry_raw, "%y%m%d").date().isoformat()
    except ValueError:
        expiration_date = None

    try:
        strike = str(int(strike_raw) / 1000.0)
    except ValueError:
        strike = None

    return {
        "underlying": underlying,
        "expiration_date": expiration_date,
        "option_type": "call" if option_flag == "C" else "put",
        "strike": strike,
    }

# AGENTS: Mission-critical universe guard; filters malformed or unsupported underlying symbols.
def is_supported_underlying(symbol: str) -> bool:
    return bool(symbol and SUPPORTED_UNDERLYING_RE.fullmatch(symbol))

# AGENTS: Mission-critical close lookup helper; loads local clean-stock files into a process cache.
def load_stock_history(symbol: str, stock_cache: StockHistoryCache) -> Optional[Tuple[pd.Series, pd.Series, pd.Series]]:
    if symbol in stock_cache:
        return stock_cache[symbol]

    stock_file = CLEAN_STOCK_ROOT / f"{symbol}_stock_data.csv"
    if not stock_file.exists():
        stock_cache[symbol] = None
        return None

    try:
        df = pd.read_csv(stock_file)
    except Exception:
        stock_cache[symbol] = None
        return None

    if df.empty:
        stock_cache[symbol] = None
        return None

    date_col = next((c for c in df.columns if "date" in str(c).lower()), df.columns[0])
    close_col = next(
        (
            c
            for c in df.columns
            if c != date_col and ("close" in str(c).lower() or "priceclose" in str(c).lower())
        ),
        None,
    )
    if close_col is None:
        stock_cache[symbol] = None
        return None

    dividend_yield_col = next(
        (c for c in df.columns if str(c).lower() == DIVIDEND_YIELD_FIELD),
        None,
    )
    dates = pd.to_datetime(df[date_col], errors="coerce")
    close = pd.to_numeric(df[close_col], errors="coerce")
    if dividend_yield_col is None:
        dividend_yield = pd.Series(0.0, index=df.index)
    else:
        dividend_yield = pd.to_numeric(df[dividend_yield_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    stock_cache[symbol] = (dates, close, dividend_yield)
    return stock_cache[symbol]

# AGENTS: Mission-critical close lookup helper; returns nearest local underlying close for a trade date.
def load_local_underlying_close(
    symbol: str,
    trade_date: date,
    stock_cache: StockHistoryCache,
) -> Optional[float]:
    history = load_stock_history(symbol, stock_cache)
    if history is None:
        return None

    dates, close, _ = history
    match = close[dates.dt.date == trade_date]
    if match.empty:
        return None

    value = match.iloc[-1]
    return float(value) if pd.notna(value) else None

# AGENTS: Mission-critical dividend lookup helper; returns local trailing dividend yield for a trade date.
def load_local_dividend_yield(
    symbol: str,
    trade_date: date,
    stock_cache: StockHistoryCache,
) -> float:
    history = load_stock_history(symbol, stock_cache)
    if history is None:
        return 0.0

    dates, _, dividend_yield = history
    match = dividend_yield[dates.dt.date == trade_date]
    if match.empty:
        return 0.0

    value = match.iloc[-1]
    return float(value) if pd.notna(value) else 0.0

# AGENTS: Mission-critical cache reader; avoids repeated stock close lookups across option files.
def load_close_cache(path: Path) -> Dict[str, Optional[float]]:
    if not path.exists():
        return {}

    try:
        df = pd.read_csv(path)
    except Exception:
        return {}

    cache: Dict[str, Optional[float]] = {}
    for row in df.itertuples(index=False):
        symbol = getattr(row, "symbol", None)
        trade_date_value = getattr(row, "trade_date", None)
        close = getattr(row, "close", None)
        if not symbol or not trade_date_value:
            continue
        cache[f"{symbol}|{trade_date_value}"] = None if pd.isna(close) else float(close)
    return cache

# AGENTS: Mission-critical cache writer; main.py calls this after each processed option flatfile.
def save_close_cache(path: Path, cache: Dict[str, Optional[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, value in sorted(cache.items()):
        if value is None:
            continue
        symbol, trade_date_value = key.split("|", 1)
        rows.append(
            {
                "symbol": symbol,
                "trade_date": trade_date_value,
                "close": value,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)

# AGENTS: Mission-critical close hydrator; maps every active underlying/date pair to a stock close.
def fetch_underlying_closes(
    symbols: Iterable[str],
    trade_date: date,
    close_cache: Dict[str, Optional[float]],
    stock_cache: StockHistoryCache,
    testing: bool = False,
    test_stock: str = "AAPL",
    cache_updates: Optional[Dict[str, Optional[float]]] = None,
) -> Dict[str, Optional[float]]:
    closes: Dict[str, Optional[float]] = {}
    trade_date_key = trade_date.isoformat()
    resolved_test_stock = normalize_underlying_symbol(test_stock)

    for symbol in sorted(set(symbols)):
        symbol = normalize_underlying_symbol(symbol)
        if not symbol:
            continue
        lookup_symbol = resolved_test_stock if testing else symbol
        cache_key = f"{lookup_symbol}|{trade_date_key}"
        if cache_key in close_cache:
            closes[symbol] = close_cache[cache_key]
            continue

        close = load_local_underlying_close(lookup_symbol, trade_date, stock_cache)
        if close is None:
            closes[symbol] = None
            close_cache[cache_key] = None
            if cache_updates is not None:
                cache_updates[cache_key] = None
        else:
            closes[symbol] = close
            close_cache[cache_key] = close
            if cache_updates is not None:
                cache_updates[cache_key] = close

    return closes

# AGENTS: Mission-critical dividend hydrator; maps every active underlying/date pair to a stock dividend yield.
def fetch_underlying_dividend_yields(
    symbols: Iterable[str],
    trade_date: date,
    stock_cache: StockHistoryCache,
    testing: bool = False,
    test_stock: str = "AAPL",
) -> Dict[str, float]:
    dividend_yields: Dict[str, float] = {}
    resolved_test_stock = normalize_underlying_symbol(test_stock)

    for symbol in sorted(set(symbols)):
        symbol = normalize_underlying_symbol(symbol)
        if not symbol:
            continue
        lookup_symbol = resolved_test_stock if testing else symbol
        dividend_yields[symbol] = load_local_dividend_yield(lookup_symbol, trade_date, stock_cache)

    return dividend_yields

# AGENTS: Mission-critical row worker; computes IV for one option row and is used in serial/parallel paths.
def compute_row_iv_task(task: IvTask) -> IvResult:
    underlying_close, strike, close, time_to_expiry_years, risk_free_rate, dividend_yield, option_type = task
    if underlying_close is None or strike is None or close is None:
        return None, "missing_input"
    if time_to_expiry_years is None or time_to_expiry_years <= 0:
        return None, "invalid_expiry"
    if risk_free_rate is None:
        return None, "missing_risk_free_rate"
    q = 0.0 if dividend_yield is None else max(0.0, float(dividend_yield))

    intrinsic = max(underlying_close - strike, 0.0) if option_type == "call" else max(strike - underlying_close, 0.0)
    upper = underlying_close if option_type == "call" else strike
    if close < intrinsic - 1e-6 or close > upper + 1e-6:
        return None, "arbitrage_bounds"

    # AGENTS: Mission-critical nested solver target; local fast IV solve used before falling back to utils.compute_iv().
    def objective(sigma: float) -> float:
        return american_option_price_crr(
            S=float(underlying_close),
            K=float(strike),
            T=float(time_to_expiry_years),
            r=float(risk_free_rate),
            sigma=sigma,
            option_type=option_type,
            q=q,
        ) - float(close)

    try:
        lo_val = objective(0.001)
        hi_val = objective(5.0)
    except Exception:
        return None, "solver_error"

    if lo_val * hi_val > 0:
        return None, "no_root_in_range"

    iv = compute_iv(
        market_price=float(close),
        S=float(underlying_close),
        K=float(strike),
        T=float(time_to_expiry_years),
        r=float(risk_free_rate),
        option_type=option_type,
        q=q,
    )
    if iv is None:
        return None, "solver_error"
    return iv, "ok"

# AGENTS: Mission-critical row worker; computes hypothetical ATM Greeks for one option row.
def compute_row_atm_greeks_task(task: GreekTask) -> dict:
    implied_volatility, underlying_close, time_to_expiry_years, risk_free_rate, dividend_yield, option_type = task
    if implied_volatility is None or underlying_close is None:
        return {"delta": None, "gamma": None, "theta": None, "vega": None}
    if time_to_expiry_years is None or time_to_expiry_years <= 0 or risk_free_rate is None:
        return {"delta": None, "gamma": None, "theta": None, "vega": None}
    q = 0.0 if dividend_yield is None else max(0.0, float(dividend_yield))
    return hypothetical_atm_greeks(
        sigma=float(implied_volatility),
        S=float(underlying_close),
        T=float(time_to_expiry_years),
        r=float(risk_free_rate),
        option_type=option_type,
        q=q,
    )

# AGENTS: Mission-critical execution helper; switches between serial and process-pool row execution.
def run_batched_tasks(
    tasks: List[Tuple],
    workers: int,
    batch_size: int,
    task_label: str,
    trade_date: date,
    worker_fn: Callable[[Tuple], object],
) -> List[object]:
    results: List[object] = []
    total = len(tasks)
    if total == 0:
        return results

    if workers > 1 and total > 1:
        mp_context = mp.get_context("spawn")
        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp_context,
                max_tasks_per_child=25000,
            ) as executor:
                for start_idx in range(0, total, batch_size):
                    end_idx = min(start_idx + batch_size, total)
                    batch = tasks[start_idx:end_idx]
                    print(f"  {task_label} progress {end_idx}/{total} rows for {trade_date.isoformat()}")
                    chunksize = max(1, len(batch) // (workers * 4))
                    results.extend(executor.map(worker_fn, batch, chunksize=chunksize))
        except BrokenProcessPool as exc:
            print(
                f"  {task_label} worker process pool failed for {trade_date.isoformat()}; "
                f"retrying remaining rows sequentially. Original error: {exc}"
            )
            completed = len(results)
            for start_idx in range(completed, total, batch_size):
                end_idx = min(start_idx + batch_size, total)
                batch = tasks[start_idx:end_idx]
                print(f"  {task_label} progress {end_idx}/{total} rows for {trade_date.isoformat()}")
                results.extend(worker_fn(task) for task in batch)
    else:
        for start_idx in range(0, total, batch_size):
            end_idx = min(start_idx + batch_size, total)
            batch = tasks[start_idx:end_idx]
            print(f"  {task_label} progress {end_idx}/{total} rows for {trade_date.isoformat()}")
            results.extend(worker_fn(task) for task in batch)

    return results

# AGENTS: Mission-critical task adapter for main.py; unpacks the tuple sent by run_options_pipeline().
def process_file_task(args: Tuple) -> tuple[Path, pd.DataFrame, Dict[str, Optional[float]]]:
    return process_file(*args)


def process_file_output_path(path: Path, input_root: Path, output_root: Path) -> Path:
    return output_root / path.relative_to(input_root).with_suffix("")


def process_file_will_write_output(
    path: Path,
    input_root: Path,
    output_root: Path,
    overwrite: bool,
) -> tuple[bool, Path]:
    output_path = process_file_output_path(path, input_root, output_root)
    if output_path.exists() and not overwrite and existing_iv_output_is_valid(output_path):
        return False, output_path
    return True, output_path


def preflight_process_file_task_outputs(tasks: Iterable[Tuple]) -> None:
    for task in tasks:
        path, input_root, output_root, overwrite = task[:4]
        will_write, output_path = process_file_will_write_output(path, input_root, output_root, overwrite)
        if will_write:
            ensure_csv_output_path_writable(output_path)


def format_path_permissions(path: Path) -> str:
    try:
        stat_result = path.stat()
    except OSError:
        return "unavailable"
    try:
        owner = pwd.getpwuid(stat_result.st_uid).pw_name
    except KeyError:
        owner = str(stat_result.st_uid)
    try:
        group = grp.getgrgid(stat_result.st_gid).gr_name
    except KeyError:
        group = str(stat_result.st_gid)
    return (
        f"owner={owner} "
        f"group={group} "
        f"mode={oct(stat_result.st_mode & 0o777)}"
    )


def ensure_csv_output_path_writable(output_path: Path) -> None:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot create output directory {output_path.parent} "
            f"({format_path_permissions(output_path.parent.parent)})."
        ) from exc

    if not os.access(output_path.parent, os.W_OK | os.X_OK):
        raise PermissionError(
            f"Cannot write {output_path}; parent directory is not writable by this user "
            f"({format_path_permissions(output_path.parent)}). "
            "Fix the directory ownership/permissions or choose a writable output root."
        )

    temp_output_path = build_temp_path(output_path)
    if temp_output_path.exists() and not os.access(temp_output_path, os.W_OK):
        raise PermissionError(
            f"Cannot overwrite stale temp file {temp_output_path} "
            f"({format_path_permissions(temp_output_path)}). "
            "Remove it or fix its ownership/permissions before rerunning."
        )


def write_output_csv_atomic(df: pd.DataFrame, output_path: Path) -> None:
    ensure_csv_output_path_writable(output_path)
    temp_output_path = build_temp_path(output_path)
    df.to_csv(temp_output_path, index=False)
    temp_output_path.replace(output_path)

# AGENTS: Mission-critical option-file processor; parses, filters, prices, and writes one OPRA day file.
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
    risk_free_rates: Dict[date, float],
) -> tuple[Path, pd.DataFrame, Dict[str, Optional[float]]]:
    trade_date = trade_date_from_path(path)
    rel_path = path.relative_to(input_root)
    output_path = process_file_output_path(path, input_root, output_root)
    stock_cache: StockHistoryCache = {}
    local_close_cache = dict(close_cache)
    close_cache_updates: Dict[str, Optional[float]] = {}

    if output_path.exists() and not overwrite:
        if existing_iv_output_is_valid(output_path):
            return output_path, pd.DataFrame(), close_cache_updates

    ensure_csv_output_path_writable(output_path)

    print(f"  processing {rel_path}")

    df = pd.read_csv(path, compression="gzip")
    if df.empty:
        write_output_csv_atomic(df, output_path)
        return output_path, pd.DataFrame(), close_cache_updates

    parsed = df["ticker"].apply(parse_opra_option_ticker)
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
        write_output_csv_atomic(enriched, output_path)
        return output_path, pd.DataFrame(), close_cache_updates

    enriched["trade_date"] = pd.Timestamp(trade_date)
    enriched["expiration_date"] = pd.to_datetime(enriched["expiration_date"], errors="coerce")
    enriched["close"] = pd.to_numeric(enriched["close"], errors="coerce")
    enriched["strike"] = pd.to_numeric(enriched["strike"], errors="coerce")
    enriched["dte_days"] = (enriched["expiration_date"] - enriched["trade_date"]).dt.days
    enriched["time_to_expiry_years"] = enriched["dte_days"] / 365.25
    enriched = enriched[enriched["dte_days"].between(DTE_MIN_DAYS, DTE_MAX_DAYS)].copy()

    enriched["underlying"] = enriched["underlying"].map(normalize_underlying_symbol)
    enriched = enriched[enriched["underlying"].map(is_supported_underlying)].copy()
    if allowed_underlyings is not None:
        enriched = enriched[enriched["underlying"].isin(allowed_underlyings)].copy()
    enriched = enriched[enriched["volume"].fillna(0) > 0].copy()
    enriched = enriched[enriched["close"].fillna(0) > 0].copy()
    if enriched.empty:
        write_output_csv_atomic(enriched, output_path)
        return output_path, pd.DataFrame(), close_cache_updates

    if testing:
        enriched = enriched[enriched["underlying"] == test_stock].copy()
        if enriched.empty:
            write_output_csv_atomic(enriched, output_path)
            print(f"  diagnostics testing filter removed all rows for test_stock={test_stock}")
            return output_path, pd.DataFrame(), close_cache_updates

    underlying_closes = fetch_underlying_closes(
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
    underlying_dividend_yields = fetch_underlying_dividend_yields(
        enriched["underlying"].dropna().unique(),
        trade_date,
        stock_cache,
        testing=testing,
        test_stock=test_stock,
    )
    enriched[DIVIDEND_YIELD_FIELD] = enriched["underlying"].map(underlying_dividend_yields).fillna(0.0)
    risk_free_rate = risk_free_rates.get(trade_date)
    if risk_free_rate is None:
        raise ValueError(f"Missing risk-free rate for {trade_date.isoformat()}")
    enriched["risk_free_rate"] = risk_free_rate
    rows_before_iv = len(enriched)

    iv_tasks: List[IvTask] = list(
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
    iv_results = run_batched_tasks(
        tasks=iv_tasks,
        workers=row_workers,
        batch_size=5000,
        task_label="IV",
        trade_date=trade_date,
        worker_fn=compute_row_iv_task,
    )
    iv_values = [result[0] for result in iv_results]
    iv_failure_reasons = [result[1] for result in iv_results]
    enriched["implied_volatility"] = iv_values
    enriched["iv_failure_reason"] = iv_failure_reasons
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

    enriched = enriched.dropna(subset=["implied_volatility"]).copy()
    if enriched.empty:
        enriched["atm_delta"] = pd.Series(dtype=float)
        enriched["atm_gamma"] = pd.Series(dtype=float)
        enriched["atm_theta"] = pd.Series(dtype=float)
        enriched["atm_vega"] = pd.Series(dtype=float)
        write_output_csv_atomic(enriched, output_path)
        features = enriched[
            [
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
                DIVIDEND_YIELD_FIELD,
                "implied_volatility",
                "atm_delta",
                "atm_gamma",
                "atm_theta",
                "atm_vega",
                "volume",
                "transactions",
            ]
        ].copy()
        print(
            f"  diagnostics total_options={rows_before_iv} "
            f"calls={n_calls} puts={n_puts} "
            f"missing_underlying={missing_underlying} "
            f"invalid_expiry={invalid_expiry} "
            f"iv_written={iv_ready_rows} "
            f"iv_failed={iv_failed_rows}"
        )
        if iv_failure_counts:
            print(f"  iv_failure_breakdown={iv_failure_counts}")
        return output_path, features, close_cache_updates

    greek_tasks: List[GreekTask] = list(
        zip(
            enriched["implied_volatility"].where(pd.notna(enriched["implied_volatility"]), None).tolist(),
            enriched["underlying_close"].where(pd.notna(enriched["underlying_close"]), None).tolist(),
            enriched["time_to_expiry_years"].where(pd.notna(enriched["time_to_expiry_years"]), None).tolist(),
            enriched["risk_free_rate"].where(pd.notna(enriched["risk_free_rate"]), None).tolist(),
            enriched[DIVIDEND_YIELD_FIELD].where(pd.notna(enriched[DIVIDEND_YIELD_FIELD]), None).tolist(),
            enriched["option_type"].tolist(),
        )
    )
    greek_values = run_batched_tasks(
        tasks=greek_tasks,
        workers=row_workers,
        batch_size=5000,
        task_label="ATM Greeks",
        trade_date=trade_date,
        worker_fn=compute_row_atm_greeks_task,
    )
    greeks_df = pd.DataFrame(greek_values, index=enriched.index)
    enriched[["atm_delta", "atm_gamma", "atm_theta", "atm_vega"]] = greeks_df[
        ["delta", "gamma", "theta", "vega"]
    ]

    write_output_csv_atomic(enriched, output_path)
    features = enriched[
        [
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
            DIVIDEND_YIELD_FIELD,
            "implied_volatility",
            "atm_delta",
            "atm_gamma",
            "atm_theta",
            "atm_vega",
            "volume",
            "transactions",
        ]
    ].copy()
    print(
        f"  diagnostics total_options={rows_before_iv} "
        f"calls={n_calls} puts={n_puts} "
        f"missing_underlying={missing_underlying} "
        f"invalid_expiry={invalid_expiry} "
        f"iv_written={iv_ready_rows} "
        f"iv_failed={iv_failed_rows}"
    )
    if iv_failure_counts:
        print(f"  iv_failure_breakdown={iv_failure_counts}")
    return output_path, features, close_cache_updates

REQUIRED_IV_OUTPUT_COLS = {
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
    DIVIDEND_YIELD_FIELD,
    "implied_volatility",
    "atm_delta",
    "atm_gamma",
    "atm_theta",
    "atm_vega",
    "volume",
    "transactions",
}


# AGENTS: Resume support; used to avoid reprocessing valid per-day IV outputs.
def existing_iv_output_is_valid(output_path: Path) -> bool:
    try:
        columns = pd.read_csv(output_path, nrows=0).columns
    except Exception:
        return False

    if not REQUIRED_IV_OUTPUT_COLS.issubset(columns):
        print(f"  reprocessing stale IV file (schema mismatch): {output_path}")
        return False
    return True

FINAL_FEATURE_MAP = {
    "avg_call_iv": "implied_volatility_call",
    "avg_put_iv": "implied_volatility_put",
    "avg_call_delta": "atm_delta_call",
    "avg_put_delta": "atm_delta_put",
    "avg_call_gamma": "atm_gamma_call",
    "avg_put_gamma": "atm_gamma_put",
    "avg_call_theta": "atm_theta_call",
    "avg_put_theta": "atm_theta_put",
    "avg_call_vega": "atm_vega_call",
    "avg_put_vega": "atm_vega_put",
}

DAILY_FEATURE_COLUMNS = ["trade_date", "underlying", *FINAL_FEATURE_MAP.keys()]


def build_daily_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=DAILY_FEATURE_COLUMNS)

    if "option_type" not in df.columns and "ticker" in df.columns:
        parsed = df["ticker"].apply(parse_opra_option_ticker)
        option_type = [
            item.get("option_type") if isinstance(item, dict) else None
            for item in parsed
        ]
        df["option_type"] = option_type

    required = {
        "trade_date", "underlying", "option_type", "implied_volatility",
        "underlying_close", "time_to_expiry_years", "risk_free_rate",
    }
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Cannot build daily features; missing columns {missing}")
    if DIVIDEND_YIELD_FIELD not in df.columns:
        df[DIVIDEND_YIELD_FIELD] = 0.0

    df["trade_date"] = parse_mixed_datetimes(df["trade_date"]).dt.date
    df["underlying"] = df["underlying"].map(normalize_underlying_symbol)
    df = df[df["option_type"].isin(["call", "put"])].copy()
    df = df[df["underlying"].notna()].copy()
    greek_cols = ["atm_delta", "atm_gamma", "atm_theta", "atm_vega"]
    has_contract_greeks = all(greek_col in df.columns for greek_col in greek_cols)

    aggregated = build_final_feature_frame(df, has_contract_greeks, greek_cols)

    if aggregated.empty:
        return pd.DataFrame(columns=DAILY_FEATURE_COLUMNS)

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
                "atm_delta": f"atm_delta_{suffix}",
                "atm_gamma": f"atm_gamma_{suffix}",
                "atm_theta": f"atm_theta_{suffix}",
                "atm_vega": f"atm_vega_{suffix}",
            }
        )
        side_cols = [
            col
            for col in [
                "trade_date",
                "underlying",
                f"implied_volatility_{suffix}",
                f"atm_delta_{suffix}",
                f"atm_gamma_{suffix}",
                f"atm_theta_{suffix}",
                f"atm_vega_{suffix}",
            ]
            if col in side.columns
        ]
        feature_df = feature_df.merge(
            side[side_cols],
            on=["trade_date", "underlying"],
            how="left",
        )

    for feature_name, source_col in FINAL_FEATURE_MAP.items():
        feature_df[feature_name] = feature_df[source_col] if source_col in feature_df.columns else pd.NA
    return feature_df[DAILY_FEATURE_COLUMNS]


def daily_feature_output_path(iv_output_path: Path, iv_root: Path, daily_features_root: Path) -> Path:
    return daily_features_root / iv_output_path.relative_to(iv_root)


def count_csv_data_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return max(0, sum(1 for _ in handle) - 1)
    except OSError:
        return 0


def daily_feature_file_is_valid(path: Path, source_path: Optional[Path] = None) -> bool:
    try:
        if source_path is not None and path.stat().st_mtime < source_path.stat().st_mtime:
            return False
        columns = pd.read_csv(path, nrows=0).columns
    except (Exception, OSError):
        return False
    return set(DAILY_FEATURE_COLUMNS).issubset(columns)


def export_daily_feature_file(
    iv_output_path: Path,
    daily_features_root: Path,
    iv_root: Path = OUTPUT_ROOT,
    reuse_existing: bool = True,
) -> tuple[Path, int]:
    output_path = daily_feature_output_path(iv_output_path, iv_root, daily_features_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if reuse_existing and output_path.exists() and daily_feature_file_is_valid(output_path, iv_output_path):
        return output_path, count_csv_data_rows(output_path)

    df = pd.read_csv(iv_output_path)
    feature_df = build_daily_feature_frame(df)
    if "underlying" in df.columns and "underlying" in feature_df.columns:
        source_underlyings = set(df["underlying"].dropna().map(normalize_underlying_symbol).unique())
        feature_underlyings = set(feature_df["underlying"].dropna().unique())
        missing_underlyings = sorted(source_underlyings - feature_underlyings)
        if missing_underlyings:
            sample = ", ".join(missing_underlyings[:10])
            if len(missing_underlyings) > 10:
                sample = f"{sample}, ..."
            print(
                f"  WARNING: {len(missing_underlyings)} underlying(s) missing from "
                f"daily features for {iv_output_path}: {sample}"
            )
    write_output_csv_atomic(feature_df, output_path)
    return output_path, len(feature_df)


def iter_daily_feature_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.csv"))


def export_final_feature_files(daily_features_root: Path, output_dir: Path) -> None:
    daily_feature_files = iter_daily_feature_files(daily_features_root)
    if not daily_feature_files:
        print(f"Skipping final feature export; no daily feature files under {daily_features_root}")
        return

    daily_frames = []
    for path in daily_feature_files:
        df = pd.read_csv(path)
        if not df.empty:
            daily_frames.append(df)

    if not daily_frames:
        print(f"Skipping final feature export; daily feature files under {daily_features_root} are empty")
        return

    all_daily_features = pd.concat(daily_frames, ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    for feature_name in FINAL_FEATURE_MAP:
        if feature_name not in all_daily_features.columns:
            print(f"Writing empty {feature_name}; source column {feature_name} not present")
            feature_df = pd.DataFrame(columns=["trade_date", "underlying", feature_name])
        else:
            feature_df = all_daily_features[["trade_date", "underlying", feature_name]].copy()
        feature_df.to_csv(output_dir / f"{feature_name}.csv", index=False)

    print(f"Wrote final feature CSVs to {output_dir}")

# AGENTS: Mission-critical final feature builder; aggregates per-option rows into stock/date feature rows.
def build_final_feature_frame(
    df: pd.DataFrame,
    has_contract_greeks: bool,
    greek_cols: List[str],
) -> pd.DataFrame:
    if has_contract_greeks:
        aggregation_cols = ["implied_volatility", *greek_cols]
        return (
            df.groupby(["trade_date", "underlying", "option_type"])[aggregation_cols]
            .mean()
            .reset_index()
        )

    print(
        "Contract-level ATM Greeks are missing from the per-day IV file; "
        "computing final delta/gamma/theta/vega features from grouped American-model "
        "inputs instead."
    )

    grouped = (
        df.groupby(["trade_date", "underlying", "option_type"])
        .agg(
            implied_volatility=("implied_volatility", "mean"),
            underlying_close=("underlying_close", "mean"),
            time_to_expiry_years=("time_to_expiry_years", "mean"),
            risk_free_rate=("risk_free_rate", "mean"),
            dividend_yield=(DIVIDEND_YIELD_FIELD, "mean"),
        )
        .reset_index()
    )

    greek_frame = grouped.apply(calculate_group_level_atm_greeks, axis=1, result_type="expand")
    return pd.concat([grouped, greek_frame], axis=1)[
        ["trade_date", "underlying", "option_type", "implied_volatility", *greek_cols]
    ]

# AGENTS: Mission-critical final feature helper; recomputes group-level ATM Greeks from aggregate IV values.
def calculate_group_level_atm_greeks(row: pd.Series) -> pd.Series:
    if pd.isna(row["implied_volatility"]) or pd.isna(row["underlying_close"]):
        return pd.Series(
            {"atm_delta": None, "atm_gamma": None, "atm_theta": None, "atm_vega": None}
        )
    if pd.isna(row["time_to_expiry_years"]) or row["time_to_expiry_years"] <= 0:
        return pd.Series(
            {"atm_delta": None, "atm_gamma": None, "atm_theta": None, "atm_vega": None}
        )

    dividend_yield = row.get(DIVIDEND_YIELD_FIELD, 0.0)
    q = 0.0 if pd.isna(dividend_yield) else max(0.0, float(dividend_yield))
    greeks = hypothetical_atm_greeks(
        sigma=float(row["implied_volatility"]),
        S=float(row["underlying_close"]),
        T=float(row["time_to_expiry_years"]),
        r=float(row["risk_free_rate"]),
        option_type=str(row["option_type"]),
        q=q,
    )
    return pd.Series(
        {
            "atm_delta": greeks["delta"],
            "atm_gamma": greeks["gamma"],
            "atm_theta": greeks["theta"],
            "atm_vega": greeks["vega"],
        }
    )

# AGENTS: Mission-critical temp-file helper; called by main.py and standalone CLI for atomic-ish output replacement.
def build_temp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


# AGENTS: Standalone CLI support; rebuilds feature outputs from already-computed IV checkpoints.
def rebuild_feature_files_from_iv_outputs(
    iv_root: Path,
    daily_features_root: Path,
    final_features_dir: Path,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    limit_files: Optional[int] = None,
    overwrite_daily_features: bool = False,
) -> int:
    iv_output_files = iter_daily_feature_files(iv_root)
    iv_output_files = filter_files_by_trade_date(iv_output_files, start_date, end_date)
    if limit_files is not None:
        iv_output_files = iv_output_files[:limit_files]

    if not iv_output_files:
        print(f"No IV output files found under {iv_root}")
        return 0

    print(f"Rebuilding features from {len(iv_output_files)} existing IV output files under {iv_root}")
    print(f"Writing daily stock/date feature CSVs under {daily_features_root}")
    print(f"Writing final feature CSVs to {final_features_dir}")

    feature_rows = 0
    for completed, iv_output_path in enumerate(iv_output_files, start=1):
        daily_features_path, daily_feature_rows = export_daily_feature_file(
            iv_output_path,
            daily_features_root,
            iv_root,
            reuse_existing=not overwrite_daily_features,
        )
        feature_rows += daily_feature_rows
        print(
            f"[{completed}/{len(iv_output_files)}] features={daily_features_path} "
            f"feature_rows={daily_feature_rows}"
        )

    temp_final_features_dir = final_features_dir.with_name(f".{final_features_dir.name}.tmp")
    if temp_final_features_dir.exists():
        shutil.rmtree(temp_final_features_dir)

    export_final_feature_files(daily_features_root, temp_final_features_dir)
    if temp_final_features_dir.exists():
        if final_features_dir.exists():
            shutil.rmtree(final_features_dir)
        temp_final_features_dir.replace(final_features_dir)

    print(f"Wrote {feature_rows} stock/date feature rows under {daily_features_root}")
    return feature_rows


# AGENTS: Standalone CLI only; main.py imports helpers and does not call this entrypoint.
def main() -> None:
    args = parse_args()
    if args.start_date is not None and args.end_date is not None and args.start_date > args.end_date:
        raise ValueError("--start-date cannot be later than --end-date")
    if args.features_only:
        rebuild_feature_files_from_iv_outputs(
            args.output_root,
            args.daily_features_root,
            args.final_features_dir,
            args.start_date,
            args.end_date,
            args.limit_files,
            overwrite_daily_features=args.overwrite_features,
        )
        return
    test_stock = normalize_underlying_symbol(args.test_stock)
    allowed_underlyings = (
        {normalize_underlying_symbol(symbol) for symbol in args.underlyings if normalize_underlying_symbol(symbol)}
        if args.underlyings
        else None
    )
    files = iter_input_files(args.input_root)
    files = filter_files_by_trade_date(files, args.start_date, args.end_date)
    if args.limit_files is not None:
        files = files[:args.limit_files]

    if not files:
        print(f"No input files found under {args.input_root}")
        return

    print(f"Processing {len(files)} flat files from {args.input_root}")
    print(
        f"Saving IV-enriched call+put data under {args.output_root} "
        f"for {DTE_MIN_DAYS}-{DTE_MAX_DAYS} DTE options only"
    )
    print(f"Writing daily stock/date feature CSVs under {args.daily_features_root}")
    print(f"Writing final feature CSVs to {args.final_features_dir}")
    print(f"clean stock lookup root: {CLEAN_STOCK_ROOT}")
    if args.start_date is not None or args.end_date is not None:
        print(f"trade date filter: start={args.start_date} end={args.end_date}")
    print(f"testing mode: {args.testing}")
    if args.testing:
        print(f"testing stock override: {test_stock}")
    if allowed_underlyings is not None:
        print(f"underlying filter: {sorted(allowed_underlyings)}")
    if len(files) > 1 and args.workers > 1:
        print(f"file worker processes: {args.workers}")
        print("row worker processes per file: 1")
    else:
        print("file worker processes: 1")
        print(f"row worker processes per file: {args.workers}")

    risk_free_rates = load_risk_free_rates_for_files(files)
    print(
        f"loaded {len(risk_free_rates)} {FRED_RISK_FREE_SERIES_ID} trade-date risk-free rate values "
        f"from FRED"
    )

    completed = 0
    close_cache = load_close_cache(args.cache_output)
    feature_rows = 0
    temp_final_features_dir = args.final_features_dir.with_name(f".{args.final_features_dir.name}.tmp")

    if temp_final_features_dir.exists():
        shutil.rmtree(temp_final_features_dir)

    row_workers = args.workers if len(files) == 1 else 1
    file_tasks = [
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
            risk_free_rates,
        )
        for path in files
    ]
    preflight_process_file_task_outputs(file_tasks)

    if len(files) > 1 and args.workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            for output_path, features, close_cache_updates in executor.map(process_file_task, file_tasks):
                close_cache.update(close_cache_updates)
                daily_features_path, daily_feature_rows = export_daily_feature_file(
                    output_path,
                    args.daily_features_root,
                    args.output_root,
                    reuse_existing=not (args.overwrite or args.overwrite_features),
                )
                feature_rows += daily_feature_rows
                save_close_cache(args.cache_output, close_cache)
                completed += 1
                print(
                    f"[{completed}/{len(files)}] wrote {output_path} rows={len(features)} "
                    f"features={daily_features_path} feature_rows={daily_feature_rows}"
                )
    else:
        for output_path, features, close_cache_updates in map(process_file_task, file_tasks):
            close_cache.update(close_cache_updates)
            daily_features_path, daily_feature_rows = export_daily_feature_file(
                output_path,
                args.daily_features_root,
                args.output_root,
                reuse_existing=not (args.overwrite or args.overwrite_features),
            )
            feature_rows += daily_feature_rows
            save_close_cache(args.cache_output, close_cache)
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
