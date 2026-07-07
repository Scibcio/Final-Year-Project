import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import glob
import os

# PAGE CONFIGURATION
st.set_page_config(page_title="V4 Institutional AI Dashboard", layout="wide")

# DATA LOADER
@st.cache_data
def load_and_prep_data():
    RESULTS_FOLDER = "../walk_forward_results_V4"
    files = glob.glob(os.path.join(RESULTS_FOLDER, "Fold_*_Results.csv"))
    
    if not files:
        st.error(f"Could not find any CSV files in '{RESULTS_FOLDER}'. Check your paths!")
        return pd.DataFrame()
        
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    
    df['Outcome'] = np.where(df['Actual_Label'] == 1, 'WIN', 'LOSS')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.to_period('M').astype(str)
    
    return df

# RISK SIMULATION ENGINE
def simulate_portfolio(df, start_balance, risk_amount):
    balance = start_balance
    balances = []
    profits = []
    
    # 3:1 Reward to Risk Ratio
    reward_amount = risk_amount * 3.0 
    
    for idx, row in df.iterrows():
        # If we can't afford the next trade, the account is dead
        if balance < risk_amount:
            balances.append(balance)
            profits.append(0)
            continue
            
        if row['Outcome'] == 'WIN':
            trade_profit = reward_amount
            balance += trade_profit
        else:
            trade_profit = -risk_amount
            balance += trade_profit
            
        balances.append(balance)
        profits.append(trade_profit)
        
    df['Net_Profit'] = profits
    df['Running_Balance'] = balances
    return df

# DASHBOARD UI & LOGIC
def main():
    st.title("V4 Risk Management & Wallet Simulator")
    st.markdown("Stress-test the AI using dynamic position sizing and specific investment periods.")
    
    df = load_and_prep_data()
    if df.empty: return

    # SIDEBAR FILTERS
    st.sidebar.header("1. AI Parameters")
    min_conf = st.sidebar.slider("Minimum AI Confidence", min_value=0.50, max_value=0.99, value=0.75, step=0.01)
    
    st.sidebar.header("2. Wallet & Risk")
    start_balance = st.sidebar.number_input("Starting Wallet Balance ($)", min_value=100, max_value=100000, value=1000, step=100)
    risk_amount = st.sidebar.number_input("Risk Amount Per Trade ($)", min_value=10, max_value=5000, value=50, step=10)
    
    # UPGRADED DATE PICKER
    st.sidebar.header("3. Investment Period")
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    
    start_date = st.sidebar.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
    
    duration = st.sidebar.selectbox(
        "Investment Duration", 
        ["1 Year", "2 Years", "3 Years", "5 Years", "To End of Data", "Custom End Date"],
        index=4
    )
    
    # Calculate the End Date based on the user's choice
    if duration == "1 Year":
        end_date = (pd.to_datetime(start_date) + pd.DateOffset(years=1)).date()
    elif duration == "2 Years":
        end_date = (pd.to_datetime(start_date) + pd.DateOffset(years=2)).date()
    elif duration == "3 Years":
        end_date = (pd.to_datetime(start_date) + pd.DateOffset(years=3)).date()
    elif duration == "5 Years":
        end_date = (pd.to_datetime(start_date) + pd.DateOffset(years=5)).date()
    elif duration == "To End of Data":
        end_date = max_date
    else:
        end_date = st.sidebar.date_input("Custom End Date", value=max_date, min_value=start_date, max_value=max_date)

    # Ensure end date doesn't go past our actual data limit
    end_date = min(end_date, max_date)

    # APPLY FILTERS & SIMULATION
    filtered_df = df[
        (df['Predicted_Probability'] >= min_conf) & 
        (df['Date'].dt.date >= start_date) & 
        (df['Date'].dt.date <= end_date)
    ].copy()
        
    # Run the chronological wallet simulation
    filtered_df = simulate_portfolio(filtered_df, start_balance, risk_amount)

    # TOP METRICS
    total_trades = len(filtered_df)
    win_rate = (len(filtered_df[filtered_df['Outcome'] == 'WIN']) / total_trades * 100) if total_trades > 0 else 0
    final_balance = filtered_df['Running_Balance'].iloc[-1] if total_trades > 0 else start_balance
    
    # Check if the account blew up
    is_bankrupt = final_balance < risk_amount

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Total Executed Trades", value=f"{total_trades:,}")
    col2.metric(label="Win Rate", value=f"{win_rate:.1f}%")
    col3.metric(label="Final Wallet Balance", value=f"${final_balance:,.2f}", delta=f"${final_balance - start_balance:,.2f}")
    
    if is_bankrupt:
        col4.error("ACCOUNT BANKRUPT")
    else:
        col4.success("ACCOUNT SURVIVED")

    st.divider()

    if total_trades > 0:
        # WALLET EQUITY CURVE
        st.subheader(f"Wallet Balance ({start_date} to {end_date})")
        
        daily_equity = filtered_df.groupby(filtered_df['Date'].dt.date)['Running_Balance'].last().reset_index()
        daily_equity.rename(columns={'Date': 'Trading Day'}, inplace=True)
        
        fig = px.line(daily_equity, x='Trading Day', y='Running_Balance', title="Chronological Equity Curve (Daily Close)")
        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Bankruptcy Line")
        fig.add_hline(y=start_balance, line_dash="dot", line_color="gray", annotation_text="Starting Balance")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        fig.update_traces(line_color="#00ffcc", line_width=2)
        st.plotly_chart(fig, use_container_width=True)

        # WALLET LEDGER 
        st.subheader("Trade-by-Trade Wallet Ledger")
        
        display_df = filtered_df[['Date', 'Ticker', 'Predicted_Probability', 'Outcome', 'Net_Profit', 'Running_Balance']].copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['Predicted_Probability'] = (display_df['Predicted_Probability'] * 100).round(2).astype(str) + '%'
        display_df.rename(columns={'Predicted_Probability': 'AI Confidence', 'Net_Profit': 'Trade P/L ($)', 'Running_Balance': 'Wallet Balance ($)'}, inplace=True)
        
        def color_logic(val):
            if isinstance(val, str) and val in ['WIN', 'LOSS']:
                return 'color: #00ff00; font-weight: bold' if val == 'WIN' else 'color: #ff4444; font-weight: bold'
            if isinstance(val, (int, float)):
                if val > 0: return 'color: #00ff00'
                if val < 0: return 'color: #ff4444'
            return ''
            
        st.caption(f"Showing chronological ledger (Capped at 500 most recent rows for browser speed).")
        st.dataframe(display_df.tail(500).style.map(color_logic), use_container_width=True, height=400)
    else:
        st.warning("No trades match the selected criteria. Try expanding the dates or lowering the confidence threshold.")

if __name__ == "__main__":
    main()