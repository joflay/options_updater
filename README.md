# Options Updater

Standalone updater repo for the Options Model data pipeline and cron runner.

This repository intentionally contains code and scheduler helpers only. Runtime data is written to:

```text
/srv/data/options_model_features
```

The option IV/Greek stage uses local DGS3MO risk-free rates from:

```text
/srv/data/risk_free_rate/DGS3MO_risk_free_rate.csv
```

ATM-normalized contract and daily feature outputs are written under:

```text
/srv/data/options_model_features/ATM_Normalized_Options
```

## Contents

| Path | Purpose |
| --- | --- |
| `Data_Pipeline/` | Pipeline code for constituency snapshots, stock histories, Massive flatfile pulls, IV/Greek calculations, and feature exports. |
| `data_copy/run_main_to_srv.sh` | Cron-safe wrapper for running `Data_Pipeline/main.py` with all mutable outputs directed to `/srv/data/options_model_features`. |
| `data_copy/cron.md` | Example crontab entry and output path notes. |

## Manual Run

One-time environment setup:

```bash
cd /home/joflay/options_updater
python3 -m venv .venv
.venv/bin/python -m pip install -r Data_Pipeline/requirements.txt
```

Run the pipeline:

```bash
/home/joflay/options_updater/data_copy/run_main_to_srv.sh
```

The wrapper uses `/home/joflay/options_updater/.venv/bin/python` automatically. If the checkout lives somewhere else, create `.venv` in that checkout and run that checkout's `data_copy/run_main_to_srv.sh`.

The option IV/Greek stage uses 4 worker processes by default. Override it for a single run with:

```bash
OPTIONS_WORKERS=2 /home/joflay/options_updater/data_copy/run_main_to_srv.sh
```

## Cron

Edit cron:

```bash
crontab -e
```

Example weekday run at 7:15 UTC:

```cron
15 7 * * 1-5 /home/joflay/options_updater/data_copy/run_main_to_srv.sh
```

Latest run log:

```text
/home/joflay/options_updater/logs/main_pipeline_latest.log
```
