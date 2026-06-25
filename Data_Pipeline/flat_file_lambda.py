import json
import os
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


SOURCE_ENDPOINT_URL = os.environ.get("MASSIVE_S3_ENDPOINT_URL", "https://files.massive.com")
SOURCE_BUCKET = os.environ.get("MASSIVE_S3_BUCKET", "flatfiles")
SOURCE_ACCESS_KEY = os.environ.get("MASSIVE_S3_ACCESS_KEY")
SOURCE_SECRET_KEY = os.environ.get("MASSIVE_S3_SECRET_KEY")

DEST_BUCKET = os.environ.get("DEST_S3_BUCKET")
DEST_PREFIX = os.environ.get("DEST_S3_PREFIX", "options_data/flatfiles/us_options_opra/day_aggs_v1").strip("/")
DEST_REGION = os.environ.get("DEST_AWS_REGION")


def two_years_ago(end_date: date) -> date:
    try:
        return end_date.replace(year=end_date.year - 2)
    except ValueError:
        return end_date.replace(month=2, day=28, year=end_date.year - 2)


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def build_source_object_key(target_date: date) -> str:
    return f"us_options_opra/day_aggs_v1/{target_date:%Y/%m/%Y-%m-%d}.csv.gz"


def build_destination_object_key(target_date: date) -> str:
    prefix = f"{DEST_PREFIX}/" if DEST_PREFIX else ""
    return f"{prefix}{target_date:%Y/%m/%Y-%m-%d}.csv.gz"


def iter_dates(start_date: date, end_date: date) -> Iterable[date]:
    current_date = start_date
    while current_date <= end_date:
        yield current_date
        current_date += timedelta(days=1)


def build_source_s3_client():
    if not SOURCE_ACCESS_KEY or not SOURCE_SECRET_KEY:
        raise ValueError("Missing MASSIVE_S3_ACCESS_KEY or MASSIVE_S3_SECRET_KEY environment variables")

    session = boto3.Session(
        aws_access_key_id=SOURCE_ACCESS_KEY,
        aws_secret_access_key=SOURCE_SECRET_KEY,
    )
    return session.client(
        "s3",
        endpoint_url=SOURCE_ENDPOINT_URL,
        config=Config(signature_version="s3v4"),
    )


def build_dest_s3_client():
    kwargs: Dict[str, str] = {}
    if DEST_REGION:
        kwargs["region_name"] = DEST_REGION
    return boto3.client("s3", **kwargs)


def destination_exists(dest_s3, bucket: str, key: str) -> bool:
    try:
        dest_s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")
        if error_code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def copy_object_between_buckets(source_s3, dest_s3, source_bucket: str, source_key: str, dest_bucket: str, dest_key: str) -> None:
    response = source_s3.get_object(Bucket=source_bucket, Key=source_key)
    body = response["Body"]
    try:
        dest_s3.upload_fileobj(
            body,
            dest_bucket,
            dest_key,
            ExtraArgs={"ContentType": "application/gzip"},
        )
    finally:
        body.close()


def run_copy(start_date: date, end_date: date, overwrite: bool = False) -> Dict[str, object]:
    if not DEST_BUCKET:
        raise ValueError("Missing DEST_S3_BUCKET environment variable")

    source_s3 = build_source_s3_client()
    dest_s3 = build_dest_s3_client()

    copied = 0
    skipped_existing = 0
    skipped_missing = 0
    failed = 0
    failures: List[Dict[str, str]] = []

    for target_date in iter_dates(start_date, end_date):
        source_key = build_source_object_key(target_date)
        dest_key = build_destination_object_key(target_date)

        if not overwrite and destination_exists(dest_s3, DEST_BUCKET, dest_key):
            skipped_existing += 1
            print(f"Skipping existing s3://{DEST_BUCKET}/{dest_key}")
            continue

        try:
            print(f"Copying {source_key} -> s3://{DEST_BUCKET}/{dest_key}")
            copy_object_between_buckets(source_s3, dest_s3, SOURCE_BUCKET, source_key, DEST_BUCKET, dest_key)
            copied += 1
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code in {"404", "NoSuchKey", "NotFound"}:
                skipped_missing += 1
                print(f"Missing source file for {target_date}: {source_key}")
            else:
                failed += 1
                failures.append(
                    {
                        "date": target_date.isoformat(),
                        "source_key": source_key,
                        "dest_key": dest_key,
                        "error": str(exc),
                    }
                )
                print(f"Failed to copy {source_key}: {exc}")

    return {
        "source_bucket": SOURCE_BUCKET,
        "destination_bucket": DEST_BUCKET,
        "destination_prefix": DEST_PREFIX,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "copied": copied,
        "skipped_existing": skipped_existing,
        "skipped_missing": skipped_missing,
        "failed": failed,
        "failures": failures,
    }


def lambda_handler(event, context):
    del context

    event = event or {}
    end_date = parse_iso_date(event["end_date"]) if event.get("end_date") else date.today()
    start_date = parse_iso_date(event["start_date"]) if event.get("start_date") else two_years_ago(end_date)
    overwrite = bool(event.get("overwrite", False))

    result = run_copy(start_date, end_date, overwrite=overwrite)
    status_code = 200 if result["failed"] == 0 else 207
    return {
        "statusCode": status_code,
        "body": json.dumps(result),
    }


if __name__ == "__main__":
    default_end_date = date.today()
    default_start_date = two_years_ago(default_end_date)
    print(
        json.dumps(
            run_copy(default_start_date, default_end_date, overwrite=False),
            indent=2,
        )
    )
