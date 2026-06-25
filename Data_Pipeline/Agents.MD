# Data Pipeline Agents Guide

This document maps the `Options_Model/Data_Pipeline` folder around the project entrypoint: `main.py`.
The cleanup rule for this folder should be:

1. Keep what `main.py` imports or needs at runtime.
2. Keep validation/documentation files that help prove the pipeline output is correct.
3. Treat older standalone scripts, probes, and one-off data artifacts as deletion candidates after confirming they are not used elsewhere.

## Primary Runtime Path

Run target:

```powershell
python Options_Model\Data_Pipeline\main.py
```

`main.py` runs a five-phase pipeline:

1. Load environment variables from `.env` files.
2. Pull or reuse S&P 500 constituency snapshots.
3. Download raw OPRA option flatfiles.
4. Ensure stock price coverage.
5. Compute option IV/Greek outputs and final feature files.

Runtime dependency graph:

```text
main.py
  -> flat_file.py
       -> config.py
  -> flatfile_iv.py
       -> config.py
       -> interest_rate_getter.py
            -> config.py
       -> utils.py
  -> .env
  -> requirements.txt
```

External runtime packages used by this path include `pandas`, `numpy`, `boto3`, `botocore`, `yfinance`, and optionally `lseg-data`. `flatfile_iv.py` also uses the standard library network stack for FRED through `interest_rate_getter.py`.

## Runtime Outputs And Inputs

These paths are outside this folder but are part of the pipeline contract:

| Path | Purpose |
| --- | --- |
| `Options_Model/Constituency/` | S&P 500 constituency snapshot CSVs used to determine eligible symbols by date. |
| `Options_Model/clean stocks/` | Stock history CSVs used for underlying close prices. |
| `/srv/data/options_model_features/FlatFiles/us_options_opra/day_aggs_v1/` | Shared raw compressed OPRA daily aggregate flatfiles. Used when `/srv/data/options_model_features` exists, or override with `OPTIONS_FLATFILES_ROOT`. |
| `Options_Model/options_data/flatfiles/us_options_opra/day_aggs_v1/` | Fallback local raw compressed OPRA daily aggregate flatfiles when the shared `/srv/data` root is unavailable. |
| `Options_Model/options_data/iv/us_options_opra/day_aggs_v1/` | Per-file option outputs with IV and Greeks. |
| `Options_Model/options_data/features/day_aggs_v1/` | Daily stock/date option-feature CSVs aggregated from per-day IV outputs. |
| `Options_Model/options_data/features/underlying_close_cache.csv` | Cache for stock close lookups during option processing. |
| `Options_Model/final_features/` | Final split feature files exported by `flatfile_iv.py`. |
| `Options_Model/lseg_constituency_debug/` | Optional parser/debug dumps when LSEG returns constituency data that cannot be parsed. |

## File Inventory

| File | Main dependency? | Purpose | Cleanup status |
| --- | --- | --- | --- |
| `.env` | Yes | Local secrets/config loaded by `main.py` and `interest_rate_getter.py`. Expected keys include LSEG/Eikon-style keys and FRED API keys. Do not commit or delete without moving required secrets elsewhere. | Keep. |
| `main.py` | Entrypoint | Orchestrates the live pipeline: dotenv loading, LSEG constituency snapshots, option flatfile downloads, yfinance stock downloads, option IV/Greek processing, and final pipeline summary. | Keep. |
| `flat_file.py` | Yes | Downloads missing raw OPRA daily aggregate flatfiles from Massive flatfiles only. Called by `main.py` through `download_flat_files()`. | Keep, but review credential handling before sharing. |
| `flatfile_iv.py` | Yes | Parses OPRA flatfiles, filters by underlyings/date, loads stock closes, loads FRED risk-free rates, computes IV and hypothetical ATM Greeks, writes per-option outputs, daily stock/date feature CSVs, and final feature CSVs. | Keep. |
| `interest_rate_getter.py` | Yes, via `flatfile_iv.py` | Loads a FRED API key and fetches the `DGS3MO` risk-free rate series used by option pricing. Can also run standalone to write an interest-rate CSV. | Keep. |
| `utils.py` | Yes, via `flatfile_iv.py` | Option-pricing math: CRR American option pricing, Black-Scholes helpers, hypothetical ATM Greeks, implied-volatility solver, moneyness, and DTE conversion. | Keep. |
| `config.py` | Yes | Shared constants for API keys, date ranges, stock universe, option filters, label settings, and data paths. Imported by `flat_file.py`, `flatfile_iv.py`, `features.py`, and `interest_rate_getter.py`. | Keep, but review hardcoded default API key before sharing. |
| `requirements.txt` | Yes | Minimal dependency list for the folder. Needed to reproduce the runtime environment. | Keep and update as imports change. |
| `validate_data_completeness.py` | No runtime dependency | Post-run validation script. Imports `main.py` and `flatfile_iv.py` to verify constituency snapshots, stock coverage, option outputs, and final features. Writes `data_validation_report.csv`. | Keep as QA tooling. Note: current code references `Tuple` without importing it. |
| `validate_iv_universe.py` | No runtime dependency | Standalone IV solver quality audit across local OPRA files. Samples contracts, computes IV, reprices, and reports solve rates/failures by ticker. | Keep as QA tooling if IV audit is still useful. |
| `features.py` | No runtime dependency | Older standalone aggregator from a consolidated per-option CSV into one `stock_option_features.csv`. `flatfile_iv.py` now writes daily feature CSVs directly, so this may be redundant. | Deletion candidate after confirming no notebooks/model code use `stock_option_features.csv`. |
| `stock_data_getter.py` | No runtime dependency | Older or alternate LSEG-based stock data pipeline with RIC resolution, batch history fetching, dividend-yield backfill, failed-symbol reporting, and its own CLI. `main.py` now uses yfinance stock download logic directly. | Deletion candidate if LSEG stock backfill/dividend workflow is no longer needed. |
| `flat_file_lambda.py` | No runtime dependency | AWS Lambda-oriented script for copying Massive flatfiles into a destination S3 bucket. Not imported by `main.py`. | Deletion candidate unless cloud backfill/deployment is still active. |
| `eikontest.py` | No runtime dependency | Manual Eikon connectivity probe for `MSFT.O`. Contains a hardcoded app key. | Deletion candidate after moving any useful troubleshooting notes into docs. |
| `lsegtest.py` | No runtime dependency | Minimal LSEG connectivity/history probe for `MSFT.O`. Contains a hardcoded app key. | Deletion candidate after moving any useful troubleshooting notes into docs. |
| `readme.md` | No runtime dependency | Short setup notes and old goal statement. Superseded by this file once this documentation is kept current. | Merge useful notes here, then consider deleting. |
| `2025-11-25.csv.gz` | No runtime dependency in current location | Compressed OPRA daily aggregate data artifact sitting directly in `Data_Pipeline`. `main.py` and `flatfile_iv.py` expect raw flatfiles under `Options_Model/options_data/flatfiles/us_options_opra/day_aggs_v1/YYYY/MM/`. | Move to the expected data tree or delete if duplicate. |

## Suggested Cleanup Order

Do not delete files yet. First confirm usage outside this folder with repo-wide search.

1. Search for imports and command references:

```powershell
rg -n "stock_data_getter|features.py|flat_file_lambda|eikontest|lsegtest|validate_iv_universe|stock_option_features|2025-11-25.csv.gz" Options_Model
```

2. Confirm the primary pipeline still runs with only the keep-set present:

```powershell
python Options_Model\Data_Pipeline\main.py
```

3. Run validation after a pipeline run:

```powershell
python Options_Model\Data_Pipeline\validate_data_completeness.py
```

4. Only then remove deletion candidates one at a time, starting with connectivity probes and the misplaced root-level `.csv.gz` artifact.

## Keep Set For `main.py`

Minimum files this folder appears to need for the main pipeline:

```text
.env
main.py
config.py
flat_file.py
flatfile_iv.py
interest_rate_getter.py
utils.py
requirements.txt
```

Recommended non-runtime support files:

```text
Agents.MD
validate_data_completeness.py
validate_iv_universe.py
```
