"""
Configuration for Options Data Collection Pipeline.

Replicating: Andreou, Han, Li (2025)
"Predicting Stock Jumps and Crashes Using Options"
Journal of Futures Markets, DOI: 10.1002/fut.22609

Model: LightGBM predicting extreme monthly stock returns using option-implied signals.
Key features: implied volatility (especially puts), delta, IV skew, put/call ratios.
"""

import os

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("MASSIVE_API_KEY", "HJv1hO3biFgL8Wc3jedNXkl4Q1pzscpn")

# ---------------------------------------------------------------------------
# Stock universe — liquid S&P 500 names with active options markets
# ---------------------------------------------------------------------------
STOCK_UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM",
    "JNJ", "V", "PG", "UNH", "HD", "BAC", "MA", "XOM", "ABBV", "AVGO",
    "MRK", "CVX", "LLY", "COST", "PEP", "KO", "WMT",
]

# ---------------------------------------------------------------------------
# Collection date range (monthly snapshots)
# Adjust based on your API subscription's data history window.
# Polygon.io Starter: ~2 years | Developer: ~5 years | Advanced: full history
# ---------------------------------------------------------------------------
START_DATE = "2024-03-01"   # earliest confirmed accessible (Feb 2024 NOT_AUTHORIZED)
END_DATE   = "2026-01-01"   # last sample month (need 1 month look-forward for labels)

# ---------------------------------------------------------------------------
# Options filtering
# ---------------------------------------------------------------------------
DTE_MIN = 20    # minimum days to expiration accepted
DTE_MAX = 50    # maximum days to expiration accepted (target ~30 DTE)

# Moneyness windows: K / S  (< 1 → puts OTM, > 1 → calls OTM)
MONEYNESS_MIN = 0.80   # exclude very deep OTM (illiquid)
MONEYNESS_MAX = 1.20

# Moneyness bucket boundaries for feature aggregation
# Bucket edges:  (lower_bound, upper_bound]
MONEYNESS_BUCKETS = {
    "deep_otm_put":   (0.80, 0.90),
    "otm_put":        (0.90, 0.97),
    "atm":            (0.97, 1.03),
    "otm_call":       (1.03, 1.10),
    "deep_otm_call":  (1.10, 1.20),
}

# Max contracts fetched per (stock, month, side) — keeps API usage reasonable
MAX_CONTRACTS_PER_SIDE = 30

# Minimum option price to attempt IV computation (filters stale/zero quotes)
MIN_OPTION_PRICE = 0.05

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
JUMP_PERCENTILE  = 95   # cross-sectional top 5 % = jump
CRASH_PERCENTILE = 5    # cross-sectional bottom 5 % = crash

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR       = os.path.join(_ROOT, "options_data")
RAW_DIR        = os.path.join(DATA_DIR, "raw")
FEATURES_DIR   = os.path.join(DATA_DIR, "features")

# ---------------------------------------------------------------------------
# API rate limiting
# ---------------------------------------------------------------------------
API_SLEEP = 0.25   # seconds between calls (stay well under 5 req/s free-tier limit)
