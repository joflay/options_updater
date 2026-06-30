from bisect import bisect_right
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, List, Set, Tuple

import pandas as pd


Snapshot = Tuple[date, Set[str]]


def normalize_constituency_symbol(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def snapshot_date_from_path(path: Path) -> date | None:
    try:
        return datetime.strptime(path.stem.replace("constituency_", ""), "%Y-%m-%d").date()
    except ValueError:
        return None


def load_constituency_snapshots(root: Path) -> List[Snapshot]:
    snapshots: List[Snapshot] = []
    for path in sorted(root.glob("constituency_*.csv")):
        snapshot_date = snapshot_date_from_path(path)
        if snapshot_date is None:
            continue
        frame = pd.read_csv(path, usecols=lambda column: str(column).strip().lower() == "ticker")
        if frame.empty or not len(frame.columns):
            continue
        symbols = {
            symbol
            for symbol in frame[frame.columns[0]].map(normalize_constituency_symbol)
            if symbol
        }
        if symbols:
            snapshots.append((snapshot_date, symbols))
    if not snapshots:
        raise ValueError(f"No usable constituency snapshots found under {root}")
    return snapshots


def active_symbols_on_date(snapshots: Iterable[Snapshot], trade_date: date) -> Set[str]:
    ordered = list(snapshots)
    index = bisect_right([snapshot_date for snapshot_date, _ in ordered], trade_date) - 1
    if index < 0:
        raise ValueError(f"No constituency snapshot exists on or before {trade_date}")
    return set(ordered[index][1])
