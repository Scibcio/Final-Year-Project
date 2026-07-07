import pandas as pd
import numpy as np
import os
import glob
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION
RAW_DATA_FOLDER = "../stock_data"
OUTPUT_FOLDER = "../engin_features_V4"
SEQ_LENGTH = 60

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Define the target: Risk 1% to make 3%
TAKE_PROFIT = 0.03
STOP_LOSS = -0.01
HOLD_DAYS = 10

def fetch_spy_baseline():
    print("\nFetching S&P 500 Baseline for Beta Calculations.")
    spy = yf.download('SPY', period="15y", progress=False)
    spy.reset_index(inplace=True)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy['Date'] = pd.to_datetime(spy['Date']).dt.tz_localize(None)
    spy['SPY_Return'] = spy['Close'].pct_change()
    return spy[['Date', 'SPY_Return']].dropna()

def calculate_institutional_features(df, spy_df):
    df = df.sort_values('Date').copy()
    
    # Base Returns
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Merge SPY data for Macro & Beta context
    df = pd.merge(df, spy_df, on='Date', how='left')
    df['SPY_Return'].fillna(0, inplace=True)

    # Rolling Beta (20-Day)
    cov = df['Daily_Return'].rolling(20).cov(df['SPY_Return'])
    var = df['SPY_Return'].rolling(20).var()
    df['Beta_20'] = cov / var
    df['Beta_20'] = df['Beta_20'].replace([np.inf, -np.inf], np.nan).fillna(1.0)

    # Normalized ATR (NATR - Volatility Personality)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    df['NATR_14'] = (atr / df['Close']) * 100

    # Institutional Accumulation (20-Day Rolling VWAP)
    # VWAP = Sum(Price * Volume) / Sum(Volume)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    pv = typical_price * df['Volume']
    rolling_pv = pv.rolling(20).sum()
    rolling_vol = df['Volume'].rolling(20).sum()
    df['VWAP_20'] = rolling_pv / rolling_vol
    df['VWAP_Dist'] = (df['Close'] / df['VWAP_20']) - 1

    # Volatility Squeeze (Bollinger Band Width)
    rolling_mean = df['Close'].rolling(20).mean()
    rolling_std = df['Close'].rolling(20).std()
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    df['BB_Width'] = (upper_band - lower_band) / rolling_mean

    # Volume Surge Ratio
    vol_20ma = df['Volume'].rolling(20).mean()
    df['Vol_Surge'] = df['Volume'] / vol_20ma

    # Short-Term Momentum
    df['EMA_9_Dist'] = (df['Close'] / df['Close'].ewm(span=9, adjust=False).mean()) - 1

    # FORWARD TARGET CALCULATION
    target_labels = []
    prices = df['Close'].values
    
    for i in range(len(prices)):
        if i + HOLD_DAYS >= len(prices):
            target_labels.append(0)
            continue
            
        entry_price = prices[i]
        future_window = prices[i+1 : i+1+HOLD_DAYS]
        
        hit_take_profit = False
        hit_stop_loss = False
        
        for future_price in future_window:
            pct_change = (future_price - entry_price) / entry_price
            if pct_change <= STOP_LOSS:
                hit_stop_loss = True
                break
            elif pct_change >= TAKE_PROFIT:
                hit_take_profit = True
                break
                
        if hit_take_profit and not hit_stop_loss:
            target_labels.append(1)
        else:
            target_labels.append(0)
            
    df['Target_Label'] = target_labels

    # CLEANUP
    features_to_keep = [
        'Date', 'Daily_Return', 'SPY_Return', 'Beta_20', 'NATR_14', 
        'VWAP_Dist', 'BB_Width', 'Vol_Surge', 'EMA_9_Dist', 'Target_Label'
    ]
    
    df_clean = df[features_to_keep].dropna().copy()
    return df_clean

def main():
    print("\n" + "="*50)
    print("INITIALIZING V4 INSTITUTIONAL DATA ENGINE")
    print("="*50)

    spy_df = fetch_spy_baseline()
    
    raw_files = glob.glob(os.path.join(RAW_DATA_FOLDER, "*.csv"))
    if not raw_files:
        print(f"\nCRITICAL ERROR: No raw CSVs found in {RAW_DATA_FOLDER}")
        return
        
    print(f"\nProcessing {len(raw_files)} stocks.")
    
    processed_count = 0
    for f in raw_files:
        ticker = os.path.basename(f).replace(".csv", "")
        df = pd.read_csv(f)
        
        if len(df) < 200 or 'Close' not in df.columns:
            continue
            
        df['Date'] = pd.to_datetime(df['Date'])
        
        try:
            elite_df = calculate_institutional_features(df, spy_df)
            
            if len(elite_df) > SEQ_LENGTH:
                output_path = os.path.join(OUTPUT_FOLDER, f"{ticker}_features.csv")
                elite_df.to_csv(output_path, index=False)
                processed_count += 1
        except Exception as e:
            pass
            
    print(f"\nV4 Engine Complete! Successfully generated {processed_count} institutional feature files in '{OUTPUT_FOLDER}'.")

if __name__ == "__main__":
    main()