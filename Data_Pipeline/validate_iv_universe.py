import re
import sys
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import american_option_price_crr, compute_iv


ROOT = Path(__file__).resolve().parent.parent
OPRA_ROOT = ROOT / "options_data" / "flatfiles" / "us_options_opra" / "day_aggs_v1"
STOCK_ROOT = ROOT / "clean stocks"
OPRA_RE = re.compile(r"^O:([A-Z]{1,6})(\d{6})([CP])(\d{8})$")
DTE_MIN = 30
DTE_MAX = 90
RISK_FREE_RATE = 0.05
ALT_SYMBOLS = {"GOOG": "GOOGL"}


def load_close(symbol: str, trade_date: pd.Timestamp, stock_files: dict[str, Path], cache: dict) -> float | None:
    key = (symbol, trade_date.date())
    if key in cache:
        return cache[key]

    stock_symbol = symbol if symbol in stock_files else ALT_SYMBOLS.get(symbol)
    if stock_symbol is None:
        cache[key] = None
        return None

    path = stock_files.get(stock_symbol)
    if path is None:
        cache[key] = None
        return None

    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
    close_col = next(
        (c for c in df.columns if c != date_col and ("close" in c.lower() or "priceclose" in c.lower())),
        None,
    )
    if close_col is None:
        cache[key] = None
        return None

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    row = df.loc[df[date_col] == trade_date]
    value = float(row[close_col].iloc[-1]) if not row.empty else None
    cache[key] = value
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate IV solve stability across the local OPRA universe.")
    parser.add_argument("--file-stride", type=int, default=1, help="Use every Nth trade file to speed up validation.")
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N processed files.")
    args = parser.parse_args()

    trade_files = sorted(OPRA_ROOT.rglob("*.csv.gz"))[:: max(1, args.file_stride)]
    stock_files = {path.name.replace("_stock_data.csv", ""): path for path in STOCK_ROOT.glob("*_stock_data.csv")}
    close_cache: dict[tuple[str, object], float | None] = {}

    pre_counts = {
        "raw_rows": 0,
        "parsed_rows": 0,
        "dte_filtered_rows": 0,
        "priced_rows": 0,
        "with_close_rows": 0,
        "sampled_rows": 0,
    }
    sampled_frames: list[pd.DataFrame] = []

    for idx, path in enumerate(trade_files, start=1):
        trade_date = pd.Timestamp(path.stem.replace(".csv", ""))
        df = pd.read_csv(path, compression="gzip", usecols=["ticker", "volume", "close"])
        pre_counts["raw_rows"] += len(df)

        extracted = df["ticker"].str.extract(OPRA_RE)
        valid = extracted[0].notna()
        df = df.loc[valid].copy()
        if df.empty:
            continue

        pre_counts["parsed_rows"] += len(df)
        df["underlying"] = extracted.loc[valid, 0].values
        df["expiration_date"] = pd.to_datetime(extracted.loc[valid, 1].values, format="%y%m%d", errors="coerce")
        df["option_type"] = extracted.loc[valid, 2].map({"C": "call", "P": "put"}).values
        df["strike"] = pd.to_numeric(extracted.loc[valid, 3].values, errors="coerce") / 1000.0
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df["trade_date"] = trade_date
        df["dte_days"] = (df["expiration_date"] - df["trade_date"]).dt.days
        df = df[df["dte_days"].between(DTE_MIN, DTE_MAX)].copy()
        pre_counts["dte_filtered_rows"] += len(df)
        if df.empty:
            continue

        df = df[(df["close"] > 0) & (df["volume"] > 0)].copy()
        pre_counts["priced_rows"] += len(df)
        if df.empty:
            continue

        symbols = df["underlying"].dropna().unique().tolist()
        closes = {symbol: load_close(symbol, trade_date, stock_files, close_cache) for symbol in symbols}
        df["underlying_close"] = df["underlying"].map(closes)
        df = df[df["underlying_close"].notna()].copy()
        pre_counts["with_close_rows"] += len(df)
        if df.empty:
            continue

        df["atm_gap"] = (df["strike"] - df["underlying_close"]).abs()
        sampled = (
            df.sort_values(["underlying", "option_type", "atm_gap", "volume"], ascending=[True, True, True, False])
            .groupby(["underlying", "option_type"], as_index=False)
            .head(1)
            .copy()
        )
        pre_counts["sampled_rows"] += len(sampled)
        sampled_frames.append(
            sampled[
                [
                    "trade_date",
                    "underlying",
                    "option_type",
                    "ticker",
                    "underlying_close",
                    "strike",
                    "dte_days",
                    "close",
                    "volume",
                ]
            ]
        )

        if idx % max(1, args.progress_every) == 0:
            print(f"processed_files={idx} sampled_rows={pre_counts['sampled_rows']}", flush=True)

    sampled = pd.concat(sampled_frames, ignore_index=True) if sampled_frames else pd.DataFrame()
    print(f"SAMPLE_READY {len(sampled)}", flush=True)

    results = []
    fail_by_reason: dict[str, int] = {}
    for idx, row in enumerate(sampled.itertuples(index=False), start=1):
        iv = None
        reason = "ok"
        try:
            iv = compute_iv(
                market_price=row.close,
                S=row.underlying_close,
                K=row.strike,
                T=row.dte_days / 365.25,
                r=RISK_FREE_RATE,
                option_type=row.option_type,
            )
        except Exception as exc:
            reason = type(exc).__name__
        else:
            if iv is None:
                intrinsic = max(row.underlying_close - row.strike, 0.0) if row.option_type == "call" else max(row.strike - row.underlying_close, 0.0)
                upper = row.underlying_close if row.option_type == "call" else row.strike
                if row.close < intrinsic - 1e-6 or row.close > upper + 1e-6:
                    reason = "arbitrage_bounds"
                else:
                    reason = "no_root_or_solver"

        reprice = (
            american_option_price_crr(
                S=row.underlying_close,
                K=row.strike,
                T=row.dte_days / 365.25,
                r=RISK_FREE_RATE,
                sigma=iv,
                option_type=row.option_type,
            )
            if iv is not None
            else None
        )
        abs_error = abs(reprice - row.close) if reprice is not None else None
        results.append(
            {
                "trade_date": row.trade_date.date().isoformat(),
                "underlying": row.underlying,
                "option_type": row.option_type,
                "ticker": row.ticker,
                "dte_days": int(row.dte_days),
                "close": float(row.close),
                "underlying_close": float(row.underlying_close),
                "iv": iv,
                "reprice": reprice,
                "abs_error": abs_error,
                "reason": reason,
            }
        )

        if reason != "ok":
            fail_by_reason[reason] = fail_by_reason.get(reason, 0) + 1
        if idx % 10000 == 0:
            print(f"solved={idx}/{len(sampled)}", flush=True)

    results_df = pd.DataFrame(results)
    results_df["iv_ok"] = results_df["iv"].notna()
    per_ticker = results_df.groupby("underlying").agg(
        samples=("ticker", "count"),
        iv_ok=("iv_ok", "sum"),
        iv_fail=("iv_ok", lambda s: int((~s).sum())),
        solve_rate=("iv_ok", "mean"),
        mean_abs_reprice_error=("abs_error", "mean"),
        p95_abs_reprice_error=("abs_error", lambda s: s.quantile(0.95) if s.notna().any() else None),
    )
    per_ticker["solve_rate"] = per_ticker["solve_rate"] * 100.0

    summary = {
        "unique_tickers": int(per_ticker.shape[0]),
        "sampled_contracts": int(len(results_df)),
        "overall_solve_rate_pct": round(float(results_df["iv_ok"].mean() * 100.0), 4) if len(results_df) else None,
        "mean_abs_reprice_error": round(float(results_df["abs_error"].mean()), 10) if results_df["abs_error"].notna().any() else None,
        "p95_abs_reprice_error": round(float(results_df["abs_error"].quantile(0.95)), 10) if results_df["abs_error"].notna().any() else None,
        "tickers_ge_95pct_solve": int((per_ticker["solve_rate"] >= 95.0).sum()),
        "tickers_ge_99pct_solve": int((per_ticker["solve_rate"] >= 99.0).sum()),
        "tickers_lt_90pct_solve": int((per_ticker["solve_rate"] < 90.0).sum()),
        "fail_reasons": fail_by_reason,
        "pre_counts": pre_counts,
    }
    print("SUMMARY", flush=True)
    print(summary, flush=True)
    print("\nBOTTOM_15", flush=True)
    print(per_ticker.sort_values(["solve_rate", "samples"]).head(15).round(6).to_string(), flush=True)
    print("\nTOP_15", flush=True)
    print(per_ticker.sort_values(["solve_rate", "samples"], ascending=[False, False]).head(15).round(6).to_string(), flush=True)


if __name__ == "__main__":
    main()
