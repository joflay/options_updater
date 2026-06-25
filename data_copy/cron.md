# Cron: Main Pipeline To `/srv`

This cron wrapper runs `Data_Pipeline/main.py` with all mutable pipeline data pointed at:

```text
/srv/data/options_model_features
```

The option IV/Greek stage reads local DGS3MO risk-free rates from:

```text
/srv/data/risk_free_rate/DGS3MO_risk_free_rate.csv
```

ATM-normalized contract and daily feature outputs are written under:

```text
/srv/data/options_model_features/ATM_Normalized_Options
```

It uses the project virtualenv, writes logs under the updater checkout `logs/` directory, and uses `flock` so a second run will not start while the previous run is still active.

The option IV/Greek stage uses 4 worker processes by default. Set `OPTIONS_WORKERS` before the wrapper command to override that for a single run.

One-time setup:

```bash
cd /home/joflay/options_updater
python3 -m venv .venv
.venv/bin/python -m pip install -r Data_Pipeline/requirements.txt
```

Example crontab entry for a weekday run at 7:15 UTC:

```cron
15 7 * * 1-5 /home/joflay/options_updater/data_copy/run_main_to_srv.sh
```

Install or edit cron with:

```bash
crontab -e
```

Useful paths:

| Path | Purpose |
| --- | --- |
| `/srv/data/options_model_features/FlatFiles/us_options_opra/day_aggs_v1/` | Raw OPRA flatfiles pulled from Massive. |
| `/srv/data/options_model_features/clean stocks/` | Stock histories refreshed by `main.py` for IV/Greek inputs. |
| `/srv/data/options_model_features/ATM_Normalized_Options/contracts/` | Per-option IV/Greek calculation outputs. |
| `/srv/data/options_model_features/ATM_Normalized_Options/features/day_aggs_v1/` | Daily stock/date feature outputs. |
| `/srv/data/options_model_features/options_data/features/underlying_close_cache.csv` | Underlying close cache used during option processing. |
| `/srv/data/options_model_features/final_features/` | Split final feature CSVs. |
| `/srv/data/risk_free_rate/DGS3MO_risk_free_rate.csv` | Local DGS3MO risk-free rates used for IV/Greek calculations. |
| `/home/joflay/options_updater/logs/main_pipeline_latest.log` | Symlink to the latest cron run log. |

Run manually:

```bash
/home/joflay/options_updater/data_copy/run_main_to_srv.sh
```

Override the interpreter only when intentionally testing another environment:

```bash
PYTHON=/path/to/python /home/joflay/options_updater/data_copy/run_main_to_srv.sh
```

Override option workers for one run:

```bash
OPTIONS_WORKERS=2 /home/joflay/options_updater/data_copy/run_main_to_srv.sh
```
