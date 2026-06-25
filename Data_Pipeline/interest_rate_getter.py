from datetime import date
from pathlib import Path
import json
import os
from typing import Optional
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from config import DATA_DIR


SERIES_ID = "DGS3MO"
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
DEFAULT_RISK_FREE_RATE_CSV = Path("/srv/data/risk_free_rate/DGS3MO_risk_free_rate.csv")
ENV_CANDIDATES = (
    Path(__file__).with_name(".env"),
    Path(__file__).resolve().parent.parent / ".env",
)
FRED_API_KEY_NAMES = {"FRED_API_KEY", "fred_api", "Fred_Key", "FRED_KEY"}
RISK_FREE_RATE_CSV_NAMES = {"OPTIONS_RISK_FREE_RATE_CSV", "RISK_FREE_RATE_CSV"}

# AGENTS: Standalone default-date helper; duplicated concept in main.py and flat_file.py.
def two_years_ago(end_date: date) -> date:
    """Return the same calendar day two years earlier, adjusting for leap years."""
    try:
        return end_date.replace(year=end_date.year - 2)
    except ValueError:
        return end_date.replace(month=2, day=28, year=end_date.year - 2)

# AGENTS: Mission-critical for IV pipeline; flatfile_iv.py calls this before fetching FRED rates.
def load_fred_api_key() -> str:
    """Load the FRED API key from .env or the process environment."""
    for key_name in FRED_API_KEY_NAMES:
        env_key = os.environ.get(key_name)
        if env_key:
            return env_key.strip()

    for env_path in ENV_CANDIDATES:
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            normalized_key = key.strip()
            if normalized_key in FRED_API_KEY_NAMES:
                return value.strip().strip('"').strip("'")

    raise ValueError(
        "Missing FRED API key. Set FRED_API_KEY, FRED_KEY, Fred_Key, or fred_api "
        "in Data_Pipeline/.env."
    )

def risk_free_rate_csv_path() -> Path:
    """Return the configured local risk-free rate CSV path."""
    for key_name in RISK_FREE_RATE_CSV_NAMES:
        value = os.environ.get(key_name)
        if value:
            return Path(value).expanduser()
    return DEFAULT_RISK_FREE_RATE_CSV

def load_local_risk_free_rates(
    start_date: date,
    end_date: date,
    series_id: str = SERIES_ID,
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load local risk-free rates in the same date/rate shape as fetch_fred_series()."""
    csv_path = path or risk_free_rate_csv_path()
    if not csv_path.exists():
        return pd.DataFrame(columns=["date", "rate"])

    df = pd.read_csv(csv_path)
    date_column = "Date" if "Date" in df.columns else "date" if "date" in df.columns else None
    if date_column is None:
        raise ValueError(f"Local risk-free rate CSV is missing a Date/date column: {csv_path}")

    if "series_id" in df.columns:
        df = df[df["series_id"].astype(str).str.upper() == series_id.upper()].copy()

    if "risk_free_rate" in df.columns:
        rate_column = "risk_free_rate"
        divide_by_100 = False
    elif "rate" in df.columns:
        rate_column = "rate"
        divide_by_100 = False
    elif "risk_free_rate_percent" in df.columns:
        rate_column = "risk_free_rate_percent"
        divide_by_100 = True
    elif "value" in df.columns:
        rate_column = "value"
        divide_by_100 = True
    else:
        raise ValueError(
            f"Local risk-free rate CSV is missing a supported rate column: {csv_path}"
        )

    rates = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_column], errors="coerce"),
            "rate": pd.to_numeric(df[rate_column], errors="coerce"),
        }
    ).dropna(subset=["date", "rate"])
    if divide_by_100:
        rates["rate"] = rates["rate"] / 100.0

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    rates = rates[(rates["date"] >= start_ts) & (rates["date"] <= end_ts)]
    return rates.sort_values("date").reset_index(drop=True)

# AGENTS: Mission-critical for IV pipeline; supplies DGS3MO risk-free rates used by option pricing.
def fetch_fred_series(series_id: str, start_date: date, end_date: date, api_key: str) -> pd.DataFrame:
    """Fetch a date range for one FRED series."""
    query = urlencode(
        {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start_date.isoformat(),
            "observation_end": end_date.isoformat(),
            "sort_order": "asc",
        }
    )
    with urlopen(f"{FRED_BASE_URL}?{query}") as response:
        payload = json.loads(response.read().decode("utf-8"))

    observations = payload.get("observations", [])
    df = pd.DataFrame(observations)
    if df.empty:
        return pd.DataFrame(columns=["date", "rate"])

    df = df.loc[:, ["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # FRED DGS series are quoted as annualized percentages, e.g. 5.25.
    # The option pricing code expects decimal rates, e.g. 0.0525.
    df["rate"] = pd.to_numeric(df["value"], errors="coerce") / 100.0
    df = df.drop(columns=["value"]).dropna(subset=["date"])
    return df

# AGENTS: Standalone CLI only; main.py/flatfile_iv.py call load_fred_api_key() and fetch_fred_series() directly.
def main() -> None:
    end_date = date.today()
    start_date = two_years_ago(end_date)
    api_key = load_fred_api_key()

    df = fetch_fred_series(SERIES_ID, start_date, end_date, api_key)
    output_dir = Path(DATA_DIR) / "interest_rates"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{SERIES_ID}_{start_date.isoformat()}_{end_date.isoformat()}.csv"
    df.to_csv(output_path, index=False)

    print(
        f"Saved {len(df)} rows for {SERIES_ID} "
        f"from {start_date.isoformat()} to {end_date.isoformat()} at {output_path}"
    )


if __name__ == "__main__":
    main()
