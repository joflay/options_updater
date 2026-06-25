from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from config import API_KEY, DATA_DIR

SHARED_FLATFILE_ROOT = Path("/srv/data/options_model_features/FlatFiles/us_options_opra/day_aggs_v1")


@dataclass
# AGENTS: Mission-critical return object for main.py phase 2 flatfile download statistics.
class FlatFileStats:
    downloaded: int = 0
    skipped_existing: int = 0
    skipped_missing: int = 0
    failed: int = 0
    massive_forbidden: bool = False

# AGENTS: Mission-critical default-date helper for standalone use; duplicated concept in main.py and interest_rate_getter.py.
def two_years_ago(end_date: date) -> date:
    """Return the same calendar day two years earlier, adjusting for leap years."""
    try:
        return end_date.replace(year=end_date.year - 2)
    except ValueError:
        return end_date.replace(month=2, day=28, year=end_date.year - 2)

# AGENTS: Mission-critical Massive path builder used by download_recent_files().
def build_massive_object_key(target_date: date) -> str:
    return f"us_options_opra/day_aggs_v1/{target_date:%Y/%m/%Y-%m-%d}.csv.gz"

# AGENTS: Mission-critical local path builder; enforces the YYYY/MM raw flatfile tree main.py expects.
def local_output_path(base_dir: Path, target_date: date) -> Path:
    return base_dir / f"{target_date:%Y}" / f"{target_date:%m}" / f"{target_date:%Y-%m-%d}.csv.gz"

def nth_weekday_of_month(year: int, month: int, weekday: int, occurrence: int) -> date:
    current = date(year, month, 1)
    while current.weekday() != weekday:
        current += timedelta(days=1)
    return current + timedelta(days=7 * (occurrence - 1))


def last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    if month == 12:
        current = date(year, 12, 31)
    else:
        current = date(year, month + 1, 1) - timedelta(days=1)
    while current.weekday() != weekday:
        current -= timedelta(days=1)
    return current


def observed_fixed_holiday(year: int, month: int, day: int) -> date:
    holiday = date(year, month, day)
    if holiday.weekday() == 5:
        return holiday - timedelta(days=1)
    if holiday.weekday() == 6:
        return holiday + timedelta(days=1)
    return holiday


def easter_date(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def market_holidays(year: int) -> set[date]:
    holidays = {
        observed_fixed_holiday(year, 1, 1),
        nth_weekday_of_month(year, 1, 0, 3),
        nth_weekday_of_month(year, 2, 0, 3),
        easter_date(year) - timedelta(days=2),
        last_weekday_of_month(year, 5, 0),
        observed_fixed_holiday(year, 6, 19),
        observed_fixed_holiday(year, 7, 4),
        nth_weekday_of_month(year, 9, 0, 1),
        nth_weekday_of_month(year, 11, 3, 4),
        observed_fixed_holiday(year, 12, 25),
    }
    # One-off full market closure for the national day of mourning for Jimmy Carter.
    if year == 2025:
        holidays.add(date(2025, 1, 9))
    return holidays


# AGENTS: Mission-critical recent-download filter; skips weekends and known US options market holidays.
def is_business_day(target_date: date) -> bool:
    return target_date.weekday() < 5 and target_date not in market_holidays(target_date.year)

# AGENTS: Mission-critical Massive S3 client factory for recent OPRA flatfiles; contains credential-sensitive logic.
def build_massive_client():
    session = boto3.Session(
        aws_access_key_id="13268d39-8833-43f4-8250-29950a99924b",
        aws_secret_access_key=API_KEY,
    )
    return session.client(
        "s3",
        endpoint_url="https://files.massive.com",
        config=Config(signature_version="s3v4"),
    )

# AGENTS: Mission-critical downloader; pulls missing raw OPRA flatfiles from Massive only.
def download_massive_files(
    massive_s3,
    output_dir: Path,
    start_date: date,
    end_date: date,
    stats: FlatFileStats,
) -> None:
    massive_bucket_name = "flatfiles"
    print(f"\nDownloading missing OPRA daily aggregates from Massive for {start_date} through {end_date}...")

    current_date = start_date
    while current_date <= end_date:
        if not is_business_day(current_date):
            current_date += timedelta(days=1)
            continue

        object_key = build_massive_object_key(current_date)
        local_file_path = local_output_path(output_dir, current_date)
        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        if local_file_path.exists():
            stats.skipped_existing += 1
            current_date += timedelta(days=1)
            continue

        try:
            print(f"Downloading Massive file {object_key}")
            massive_s3.download_file(massive_bucket_name, object_key, str(local_file_path))
            stats.downloaded += 1
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code in {"404", "NoSuchKey"}:
                stats.skipped_missing += 1
                print(f"Missing file for {current_date}: {object_key}")
            elif error_code in {"403", "Forbidden", "AccessDenied"}:
                stats.massive_forbidden = True
                print(
                    f"Massive access forbidden for {current_date}: {object_key}. "
                    "Stopping recent download loop to avoid repeated 403s."
                )
                break
            else:
                stats.failed += 1
                print(f"Failed to download Massive file {object_key}: {exc}")

        current_date += timedelta(days=1)

# AGENTS: Mission-critical public API called by main.py; downloads missing flatfiles from Massive only.
def download_flat_files(
    start_date: date | None = None,
    end_date: date | None = None,
    output_dir: Path | None = None,
) -> FlatFileStats:
    resolved_end_date = end_date or date.today()
    resolved_start_date = start_date or two_years_ago(resolved_end_date)
    resolved_output_dir = output_dir or (
        SHARED_FLATFILE_ROOT
        if SHARED_FLATFILE_ROOT.parents[2].exists()
        else Path(DATA_DIR) / "flatfiles" / "us_options_opra" / "day_aggs_v1"
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    stats = FlatFileStats()
    print(f"Requested date range: {resolved_start_date} through {resolved_end_date}")
    print("Flatfile source: Massive flatfiles only")
    print(f"Saving files under: {resolved_output_dir}")

    massive_s3 = build_massive_client()
    download_massive_files(massive_s3, resolved_output_dir, resolved_start_date, resolved_end_date, stats)

    print(
        "\nFinished flat file download. "
        f"downloaded={stats.downloaded}, "
        f"skipped_existing={stats.skipped_existing}, "
        f"skipped_missing={stats.skipped_missing}, "
        f"failed={stats.failed}, "
        f"massive_forbidden={stats.massive_forbidden}"
    )
    return stats

# AGENTS: Standalone CLI only; main.py imports download_flat_files() directly and does not call this wrapper.
def main() -> None:
    download_flat_files()


if __name__ == "__main__":
    main()
