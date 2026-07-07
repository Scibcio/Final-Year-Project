import pandas as pd
import yfinance as yf
import os
import warnings

warnings.filterwarnings('ignore')

INPUT_CSV = "sp500_tickers.csv"
OUTPUT_FOLDER = "../stock_data"
START_DATE = "2010-01-01"
END_DATE = "2026-04-01"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Download Market Baselines
print("Downloading Market Baselines")
baselines = {
    "^GSPC": "SP500_Market",      # Overall Market Benchmark
    "^VIX": "VIX_Retail_Fear",    # Retail Fear / Volatility Index
    "HYG": "HYG_Inst_Panic",      # Institutional Panic (High-Yield/Junk Bonds)
    "^TNX": "TNX_Macro_Stress"    # Macroeconomic Stress (10-Year Interest Rates)
}

for ticker, name in baselines.items():

    print(f"Downloading {name} ({ticker})")

    data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)

    if not data.empty:
        data.reset_index(inplace=True)

        if not data.empty:
            data.reset_index(inplace=True)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

        data = data[["Date", "Open", "High", "Low", "Close", "Volume"]]
        data.to_csv(os.path.join(OUTPUT_FOLDER, f"{ticker}.csv"), index=False)
    else:
        print(f"Warning: Failed to fetch {name}")

# Download Individual Stocks
print("\nDownloading S&P 500 Stocks")
df = pd.read_csv(INPUT_CSV)

for ticker in df["Ticker"]:
    try:
        ticker = ticker.replace(".", "-")
        data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)

        if data.empty:
            print(f"No data for {ticker}")
            continue

        data.reset_index(inplace=True)

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data[["Date", "Open", "High", "Low", "Close", "Volume"]]

        file_path = os.path.join(OUTPUT_FOLDER, f"{ticker}.csv")
        data.to_csv(file_path, index=False)
        print(f"Downloaded {ticker}")

    except Exception as e:
        print(f"Failed for {ticker}: {e}")

print("\nData collection complete.")