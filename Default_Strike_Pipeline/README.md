# Default Strike Daily Pipeline

This stage updates the canonical strike-specific option dataset:

```text
/srv/data/options_model_features/Default_Strike/contracts
/srv/data/options_model_features/Default_Strike/Features
```

It runs after `Data_Pipeline/main.py` has produced ATM IV source rows under
`/srv/data/options_model_features/ATM_Normalized_Options/contracts`.

For every missing or stale source date, the updater:

1. Loads the latest S&P 500 constituency snapshot on or before the trade date.
2. Reuses the ATM pipeline's implied volatility.
3. Computes Greeks at each contract's actual strike.
4. Writes the daily contract CSV atomically.
5. Rebuilds the daily Default Strike feature CSV.
6. Verifies universe membership, required values, dates, and duplicates.

## Manual run

Run only the Default Strike catch-up stage:

```bash
/home/joflay/options_updater/Default_Strike_Pipeline/run_default_strike_daily.sh
```

Run the source updater and Default Strike sequentially:

```bash
/home/joflay/options_updater/Default_Strike_Pipeline/run_daily_pipeline.sh
```

## Cron

Use the combined wrapper instead of scheduling `run_main_to_srv.sh`
separately:

```cron
15 7 * * 1-5 /home/joflay/options_updater/Default_Strike_Pipeline/run_daily_pipeline.sh
```

Latest Default Strike log:

```text
/home/joflay/options_updater/logs/default_strike_latest.log
```

Set `DEFAULT_STRIKE_WORKERS` to change the default four worker processes.
Use `OPTIONS_DEFAULT_STRIKE_ROOT`, `OPTIONS_IV_ROOT`, and
`OPTIONS_CONSTITUENCY_ROOT` to override runtime paths.
