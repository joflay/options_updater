"""
Aggregate per-option IV and hypothetical ATM Greeks into 10 stock-level
features following Andreou, Han & Li (2025), Section 2.3.2.

For each stock on each date:
  - Average implied_volatility across all calls  →  sigma_c
  - Average implied_volatility across all puts   →  sigma_p
  - Average atm_delta  across all calls          →  delta_c
  - Average atm_delta  across all puts           →  delta_p
  - Average atm_gamma  across all calls          →  gamma_c
  - Average atm_gamma  across all puts           →  gamma_p
  - Average atm_theta  across all calls          →  theta_c
  - Average atm_theta  across all puts           →  theta_p
  - Average atm_vega   across all calls          →  vega_c
  - Average atm_vega   across all puts           →  vega_p

Usage:
    python features.py                          # uses default paths
    python features.py --input path/to/features.csv --output path/to/agg.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from config import DATA_DIR

INPUT_PATH = Path(DATA_DIR) / "features" / "features.csv"
OUTPUT_PATH = Path(DATA_DIR) / "features" / "stock_option_features.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate per-option features into 10 stock-level features."
    )
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the per-option features DataFrame (one row per option contract),
    compute the 10 paper features: mean IV and mean hypothetical ATM Greeks,
    split by call vs put, grouped by (underlying, trade_date).
    """
    value_cols = ["implied_volatility", "atm_delta", "atm_gamma", "atm_theta", "atm_vega"]
    rename_map = {
        "implied_volatility": "sigma",
        "atm_delta": "delta",
        "atm_gamma": "gamma",
        "atm_theta": "theta",
        "atm_vega": "vega",
    }

    grouped = (
        df.groupby(["underlying", "trade_date", "option_type"])[value_cols]
        .mean()
    )

    calls = grouped.xs("call", level="option_type").rename(
        columns={col: f"{rename_map[col]}_c" for col in value_cols}
    )
    puts = grouped.xs("put", level="option_type").rename(
        columns={col: f"{rename_map[col]}_p" for col in value_cols}
    )

    merged = calls.join(puts, how="outer").reset_index()
    return merged


def main() -> None:
    args = parse_args()

    print(f"Reading per-option features from {args.input}")
    df = pd.read_csv(args.input)
    print(f"  rows={len(df)}  unique underlyings={df['underlying'].nunique()}")

    required = ["underlying", "trade_date", "option_type",
                 "implied_volatility", "atm_delta", "atm_gamma", "atm_theta", "atm_vega"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: missing columns {missing}. Run flatfile_iv.py first.")
        return

    result = aggregate_features(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} stock-date rows to {args.output}")

    # Summary stats
    feature_cols = [c for c in result.columns if c not in ("underlying", "trade_date")]
    print("\nFeature summary:")
    print(result[feature_cols].describe().round(4).to_string())


if __name__ == "__main__":
    main()
