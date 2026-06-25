# Options Updater

Standalone updater repo for the Options Model data pipeline and cron runner.

This repository intentionally contains code and scheduler helpers only. Runtime data is written to:

```text
/srv/data/options_model_features
```

## Contents

| Path | Purpose |
| --- | --- |
| `Data_Pipeline/` | Pipeline code for constituency snapshots, stock histories, Massive flatfile pulls, IV/Greek calculations, and feature exports. |
| `data_copy/run_main_to_srv.sh` | Cron-safe wrapper for running `Data_Pipeline/main.py` with all mutable outputs directed to `/srv/data/options_model_features`. |
| `data_copy/cron.md` | Example crontab entry and output path notes. |

## Manual Run

```bash
/home/joflay/options_updater/data_copy/run_main_to_srv.sh
```

If the checkout lives somewhere else, run the script from that checkout path instead.

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
/srv/data/options_model_features/logs/main_pipeline_latest.log
```
