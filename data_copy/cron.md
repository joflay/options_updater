# Cron: Main Pipeline To `/srv`

This cron wrapper runs `Data_Pipeline/main.py` with all mutable pipeline data pointed at:

```text
/srv/data/options_model_features
```

It uses the project virtualenv when available, writes logs under the updater checkout `logs/` directory, and uses `flock` so a second run will not start while the previous run is still active.

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
| `/srv/data/options_model_features/options_data/iv/us_options_opra/day_aggs_v1/` | Per-option IV/Greek calculation outputs. |
| `/srv/data/options_model_features/options_data/features/day_aggs_v1/` | Daily stock/date feature outputs. |
| `/srv/data/options_model_features/options_data/features/underlying_close_cache.csv` | Underlying close cache used during option processing. |
| `/srv/data/options_model_features/final_features/` | Split final feature CSVs. |
| `/home/joflay/options_updater/logs/main_pipeline_latest.log` | Symlink to the latest cron run log. |

Run manually:

```bash
/home/joflay/options_updater/data_copy/run_main_to_srv.sh
```
