activate venv: python3 -m venv .venv

initialize venv: .venv\Scripts\Activate.ps1

linux source source .venv/bin/activate

The goal of this folder should be to pull and clean the relevant options data and store it in options_data. When training our model we should not be pulling data from the api and training at the same time.
We should first focus on making this folder work, in line with how the research paper does it, or our on way if we decide to.

Stock price warning: `clean stocks/` must contain raw stock prices, not dividend-adjusted prices and not split-adjusted historical prices. With Yahoo/yfinance, `auto_adjust=False` prevents dividend adjustment but the `Close` series can still be split-adjusted, so do not write Yahoo `Close` straight into `clean stocks/` without preserving and applying split handling. Split-adjusted underlying closes will corrupt option moneyness, IV, and Greeks around split events.


Execute dividend backfill

python3 Data_Pipeline/Data_Repairs.py --backfill-dividend-yield
