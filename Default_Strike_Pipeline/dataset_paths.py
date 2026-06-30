import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_DATA_ROOT = Path(
    os.environ.get(
        "OPTIONS_DATASET_ROOT",
        os.environ.get("OPTIONS_MODEL_FEATURES_ROOT", "/srv/data/options_model_features"),
    )
).expanduser()
CLEAN_STOCK_ROOT = Path(
    os.environ.get(
        "OPTIONS_CLEAN_STOCK_ROOT",
        os.environ.get("OPTIONS_MODEL_CLEAN_STOCK_ROOT", "/srv/data/stocks"),
    )
).expanduser()
CONSTITUENCY_ROOT = Path(
    os.environ.get(
        "OPTIONS_CONSTITUENCY_ROOT",
        os.environ.get("OPTIONS_MODEL_CONSTITUENCY_ROOT", FEATURE_DATA_ROOT / "Constituency"),
    )
).expanduser()
DEFAULT_STRIKE_ROOT = Path(
    os.environ.get(
        "OPTIONS_DEFAULT_STRIKE_ROOT",
        os.environ.get("OPTIONS_MODEL_DEFAULT_STRIKE_ROOT", FEATURE_DATA_ROOT / "Default_Strike"),
    )
).expanduser()
DEFAULT_STRIKE_CONTRACT_ROOT = DEFAULT_STRIKE_ROOT / "contracts"
DEFAULT_STRIKE_FEATURE_ROOT = DEFAULT_STRIKE_ROOT / "Features"
