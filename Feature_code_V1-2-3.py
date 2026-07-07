import numpy as np
import pandas as pd
import os
import glob
import warnings

warnings.filterwarnings('ignore')

# Coonfig
INPUT_FOLDER = "../stock_data"
OUTPUT_FOLDER = "../engin_features"

# Triple Barrier Method
TIME_OUT_DAYS = 10         # Vertical Barrier (Time Limit)
TAKE_PROFIT_PCT = 0.03     # Upper Barrier (+3%)
STOP_LOSS_PCT = -0.01      # Lower Barrier (-1%)

MA_WINDOWS = [5, 10, 20, 50, 100]
VOL_WINDOWS = [5, 10, 20]
MOM_WINDOWS = [5, 10, 20]
ZSCORE_WINDOWS = [20, 60]
EMA_WINDOWS = [12, 26]
RSI_WINDOW = 14
BOLL_WINDOW = 20
BOLL_STD = 2

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Indicators & Labeling Functions
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_ema(data, window):
    return data.ewm(span=window, adjust=False).mean()

def apply_dynamic_barriers(df, lookahead, tp_multiplier=1.5, sl_multiplier=0.5):
    """
    Implements a Dynamic Triple Barrier.
    """
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    
    # Pull the Volatility Percentage
    volatilities = df['Volatility_Pct'].values 
    
    # Initialize with NaNs
    labels = np.full(len(df), np.nan)
    
    for i in range(len(df) - lookahead):
        entry_price = closes[i]
        current_volatility = volatilities[i]
        
        # Calculate the exact target percentages for THIS specific day.
        # Example: If a stock normally moves 2% a day (0.02)...
        # Take Profit = 0.02 * 1.5 = 0.03 (+3%)
        # Stop Loss = -(0.02 * 0.5) = -0.01 (-1%)
        dynamic_tp_pct = current_volatility * tp_multiplier
        dynamic_sl_pct = -(current_volatility * sl_multiplier)
        
        labels[i] = 0 
        
        # Look ahead day by day (The Time Out Barrier)
        for j in range(1, lookahead + 1):
            future_high = highs[i + j]
            future_low = lows[i + j]
            
            if (future_low - entry_price) / entry_price <= dynamic_sl_pct:
                labels[i] = 0
                break 
                
            elif (future_high - entry_price) / entry_price >= dynamic_tp_pct:
                labels[i] = 1
                break 
                
    df['Target_Label'] = labels
    return df

# Feature Engine
def engineer_features(df, sp500_df=None, vix_df=None, hyg_df=None, tnx_df=None):
    """Generates technical indicators AND merges Macro Data."""

    # Macro Merging
    if sp500_df is not None:
        df = pd.merge(df, sp500_df[['Date', 'Close']], on='Date', how='left')
        df.rename(columns={'Close_y': 'SP500_Close', 'Close_x': 'Close'}, inplace=True)
        df['SP500_Return'] = df['SP500_Close'].pct_change()
        
        # True Relative Strength (Stock return minus Market return)
        df['Relative_Strength'] = df['Close'].pct_change() - df['SP500_Return']
        
        df['SP500_200MA'] = df['SP500_Close'].rolling(window=200).mean()
        df['Market_is_Bullish'] = (df['SP500_Close'] > df['SP500_200MA']).astype(int)

        df.drop(columns=['SP500_Close'], inplace=True)

    if vix_df is not None:
        df = pd.merge(df, vix_df[['Date', 'Close']], on='Date', how='left')
        df.rename(columns={'Close_y': 'VIX_Level', 'Close_x': 'Close'}, inplace=True)
        df['VIX_Change'] = df['VIX_Level'].pct_change()

    if hyg_df is not None:
        df = pd.merge(df, hyg_df[['Date', 'Close']], on='Date', how='left')
        df.rename(columns={'Close_y': 'HYG_Close', 'Close_x': 'Close'}, inplace=True)
        df['HYG_Return'] = df['HYG_Close'].pct_change()
        df.drop(columns=['HYG_Close'], inplace=True)

    if tnx_df is not None:
        df = pd.merge(df, tnx_df[['Date', 'Close']], on='Date', how='left')
        df.rename(columns={'Close_y': 'TNX_Level', 'Close_x': 'Close'}, inplace=True)
        df['TNX_Change'] = df['TNX_Level'].diff()

    df.ffill(inplace=True)
    
    # Institutional Volume Footprint
    df['Vol_20MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Surge_Ratio'] = df['Volume'] / (df['Vol_20MA'] + 1e-8)

    # Normalized Volatility (ATR %)
    df['H-L'] = df['High'] - df['Low']
    df['H-C'] = abs(df['High'] - df['Close'].shift(1))
    df['L-C'] = abs(df['Low'] - df['Close'].shift(1))
    df['True_Range'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    df['ATR_14'] = df['True_Range'].rolling(window=14).mean()
    df['Volatility_Pct'] = df['ATR_14'] / df['Close']
    df.drop(columns=['H-L', 'H-C', 'L-C', 'True_Range', 'ATR_14'], inplace=True)

    # Standard Indicators
    for window in MA_WINDOWS:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        # Rubber Band Effect
        df[f'SMA_{window}_Dist'] = (df['Close'] - df[f'SMA_{window}']) / df[f'SMA_{window}']

    for window in EMA_WINDOWS:
        df[f'EMA_{window}'] = calculate_ema(df['Close'], window)
    df['MACD'] = df[f'EMA_{EMA_WINDOWS[0]}'] - df[f'EMA_{EMA_WINDOWS[1]}']

    for window in VOL_WINDOWS:
        df[f'Vol_{window}'] = df['Close'].pct_change().rolling(window=window).std()

    for window in MOM_WINDOWS:
        df[f'Mom_{window}'] = df['Close'].pct_change(periods=window)

    for window in ZSCORE_WINDOWS:
        rolling_mean = df['Close'].rolling(window=window).mean()
        rolling_std = df['Close'].rolling(window=window).std()
        df[f'ZScore_{window}'] = (df['Close'] - rolling_mean) / (rolling_std + 1e-8)

    df['RSI'] = calculate_rsi(df['Close'], RSI_WINDOW)

    df['Boll_Mid'] = df['Close'].rolling(window=BOLL_WINDOW).mean()
    df['Boll_Std'] = df['Close'].rolling(window=BOLL_WINDOW).std()
    df['Boll_Upper'] = df['Boll_Mid'] + (BOLL_STD * df['Boll_Std'])
    df['Boll_Lower'] = df['Boll_Mid'] - (BOLL_STD * df['Boll_Std'])
    df['Boll_Pos'] = (df['Close'] - df['Boll_Lower']) / (df['Boll_Upper'] - df['Boll_Lower'] + 1e-8)

    # Apply Triple Barrier
    df = apply_dynamic_barriers(df, TIME_OUT_DAYS, TAKE_PROFIT_PCT, STOP_LOSS_PCT)

    # Drop NaNs from moving averages and the label shift
    df.dropna(inplace=True)
    
    return df

# EXECUTION LOOP
def load_baseline(filepath):
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        return df.dropna(subset=['Close'])
    return None

def main():
    print("\nLoading Market Baselines and Emotion Indices...")
    sp500_df = load_baseline(os.path.join(INPUT_FOLDER, "^GSPC.csv"))
    vix_df = load_baseline(os.path.join(INPUT_FOLDER, "^VIX.csv"))
    hyg_df = load_baseline(os.path.join(INPUT_FOLDER, "HYG.csv"))
    tnx_df = load_baseline(os.path.join(INPUT_FOLDER, "^TNX.csv"))

    macro_files = ["^GSPC.csv", "^VIX.csv", "HYG.csv", "^TNX.csv"]
    all_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
    stock_files = [f for f in all_files if os.path.basename(f) not in macro_files]

    if not stock_files:
        print("\nCRITICAL ERROR: No stock CSV files found.")
        return

    print(f"\nEngineering features for {len(stock_files)} stocks using Triple Barrier Labeling...")

    for file_path in stock_files:
        ticker = os.path.basename(file_path).replace(".csv", "")
        
        try:
            df = pd.read_csv(file_path)
            
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            df = df.dropna(subset=['Close'])

            # Safety check
            if len(df) < (max(MA_WINDOWS) + TIME_OUT_DAYS + 10):
                continue

            df_featured = engineer_features(df, sp500_df, vix_df, hyg_df, tnx_df)
            
            output_path = os.path.join(OUTPUT_FOLDER, f"{ticker}_features.csv")
            df_featured.to_csv(output_path, index=False)
            print(f"Processed: {ticker} -> {len(df_featured)} rows")

        except Exception as e:
            print(f"  [Error] {ticker}: {e}")

    print(f"\nAll features engineered and saved to '{OUTPUT_FOLDER}'")

if __name__ == "__main__":
    main()