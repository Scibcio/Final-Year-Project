import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# LOCAL VS CODE CONFIGURATION
RESULTS_FOLDER = "../walk_forward_results_V4" 
FEATURES_FOLDER = "../engin_features_V4"
PLOTS_FOLDER = "../backtest_plots"

CONFIDENCE_THRESHOLD = 0.75

if not os.path.exists(PLOTS_FOLDER): 
    os.makedirs(PLOTS_FOLDER)

plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": "#121212", 
    "figure.facecolor": "#121212", 
    "text.color": "white", 
    "axes.labelcolor": "white", 
    "xtick.color": "white", 
    "ytick.color": "white",
    "grid.color": "#333333"
})

# DATA LOADERS
def load_results():
    """Loads all Walk-Forward Fold CSVs into one massive dataframe."""
    files = glob.glob(os.path.join(RESULTS_FOLDER, "Fold_*_Results.csv"))
    if not files: 
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    return df

# PLOTTING ENGINES
def plot_underwater_drawdown(df):
    """Generates the Risk Analysis Drawdown Curve"""
    print("Generating Underwater Drawdown Curve.")
    trades = df[df['Predicted_Probability'] >= CONFIDENCE_THRESHOLD].copy()
    if len(trades) == 0: return
    
    trades['Profit'] = np.where(trades['Actual_Label'] == 1, 3.0, -1.0)
    daily_profit = trades.groupby('Date')['Profit'].sum().reset_index()
    daily_profit['Cumulative_Return'] = daily_profit['Profit'].cumsum()
    
    daily_profit['Running_Max'] = daily_profit['Cumulative_Return'].cummax()
    daily_profit['Drawdown'] = daily_profit['Cumulative_Return'] - daily_profit['Running_Max']
    
    plt.figure(figsize=(12, 5))
    plt.fill_between(daily_profit['Date'], daily_profit['Drawdown'], 0, color='#ff4444', alpha=0.6)
    plt.plot(daily_profit['Date'], daily_profit['Drawdown'], color='red', linewidth=1)
    
    plt.title('Underwater Drawdown Curve (Risk Analysis)', fontsize=16, fontweight='bold')
    plt.ylabel('Drawdown (Net % Drop)', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.savefig(os.path.join(PLOTS_FOLDER, 'Financial_08_Underwater_Drawdown.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_confidence_curve(df):
    """Generates the Win Rate vs Volume Optimization Chart"""
    print("Generating Confidence Optimization Curve.")
    thresholds = np.arange(0.50, 0.96, 0.05)
    win_rates, trade_counts = [], []
    
    for t in thresholds:
        taken = df[df['Predicted_Probability'] >= t]
        if len(taken) > 0:
            win_rates.append(taken['Actual_Label'].mean() * 100)
            trade_counts.append(len(taken))
        else:
            win_rates.append(0)
            trade_counts.append(0)
            
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = '#00ffcc'
    ax1.set_xlabel('AI Confidence Threshold', fontsize=12)
    ax1.set_ylabel('Win Rate (%)', color=color, fontsize=12)
    ax1.plot(thresholds, win_rates, color=color, marker='o', linewidth=2.5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(25.0, color='red', linestyle='--', alpha=0.5, label='25% Breakeven')
    
    ax2 = ax1.twinx()  
    color = '#ff00ff'
    ax2.set_ylabel('Total Trades Executed', color=color, fontsize=12)  
    ax2.bar(thresholds, trade_counts, width=0.02, color=color, alpha=0.3)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Win Rate vs. Trade Volume Optimization', fontsize=16, fontweight='bold')
    fig.tight_layout()  
    plt.savefig(os.path.join(PLOTS_FOLDER, 'ML_07_Confidence_Curve.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_correlation():
    """Generates the Institutional Feature Heatmap"""
    print("Generating Feature Correlation Matrix.")
    files = glob.glob(os.path.join(FEATURES_FOLDER, "*_features.csv"))
    if not files: 
        print("   -> Skipping: No feature CSVs found.")
        return
    
    sample_files = np.random.choice(files, min(5, len(files)), replace=False)
    df = pd.concat([pd.read_csv(f) for f in sample_files], ignore_index=True)
    
    features_to_plot = ['Daily_Return', 'SPY_Return', 'Beta_20', 'NATR_14', 
                        'VWAP_Dist', 'BB_Width', 'Vol_Surge', 'EMA_9_Dist', 'Target_Label']
    
    corr_matrix = df[features_to_plot].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, 
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title('Institutional Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(PLOTS_FOLDER, 'ML_08_Feature_Correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_equity_trade_overlay(df):
    """Generates the Timeline Dashboard with Trade Markers"""
    print("Generating Equity Curve with Trade Overlays.")
    
    trades = df[df['Predicted_Probability'] >= CONFIDENCE_THRESHOLD].copy()
    if len(trades) == 0: return

    STARTING_CAPITAL = 1000.0
    PROFIT_PER_WIN = 30.0   
    LOSS_PER_LOSS = -10.0   
    
    trades['Date'] = pd.to_datetime(trades['Date'])
    trades['Profit'] = np.where(trades['Actual_Label'] == 1, PROFIT_PER_WIN, LOSS_PER_LOSS)
    
    daily_profit = trades.groupby('Date')['Profit'].sum().reset_index()
    daily_profit['Equity'] = STARTING_CAPITAL + daily_profit['Profit'].cumsum()
    
    winning_days = daily_profit[daily_profit['Profit'] > 0]
    losing_days = daily_profit[daily_profit['Profit'] < 0]
    
    plt.figure(figsize=(14, 7))
    plt.plot(daily_profit['Date'], daily_profit['Equity'], color='#00ffcc', linewidth=2, alpha=0.9, label='Portfolio Equity')
    
    plt.scatter(winning_days['Date'], winning_days['Equity'], color='lime', marker='^', s=60, label='Net Profitable Day', zorder=5)
    plt.scatter(losing_days['Date'], losing_days['Equity'], color='red', marker='v', s=60, label='Net Losing Day', zorder=5)
    
    plt.title(f'AI Trade Ledger & Equity Growth (Threshold > {CONFIDENCE_THRESHOLD})', fontsize=16, fontweight='bold')
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.axhline(STARTING_CAPITAL, color='gray', linestyle='--', alpha=0.5, label='Starting Capital ($1,000)')
    
    plt.legend(loc='upper left', fontsize=12, facecolor='#121212', edgecolor='white')
    plt.savefig(os.path.join(PLOTS_FOLDER, 'Financial_09_Trade_Ledger_Overlay.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_styled_trade_ledger(df):
    """Generates the Static Image Table and the Full CSV Ledger"""
    print("Generating Styled Trade Ledger Image & CSV.")
    
    trades = df[df['Predicted_Probability'] >= CONFIDENCE_THRESHOLD].copy()
    if len(trades) == 0: return

    trades['Date'] = trades['Date'].dt.strftime('%Y-%m-%d')
    trades['Confidence'] = (trades['Predicted_Probability'] * 100).round(2).astype(str) + '%'
    trades['Outcome'] = np.where(trades['Actual_Label'] == 1, 'WIN', 'LOSS')
    trades['Net Profit'] = np.where(trades['Actual_Label'] == 1, '+$30.00', '-$10.00')
    
    # Save the full ledger as CSV
    full_ledger = trades[['Date', 'Ticker', 'Confidence', 'Outcome', 'Net Profit']]
    full_ledger.to_csv(os.path.join(PLOTS_FOLDER, 'Full_Trade_Ledger.csv'), index=False)

    # Plot sample for the image
    table_data = trades[['Date', 'Ticker', 'Confidence', 'Outcome', 'Net Profit']].tail(20).values.tolist()
    columns = ['Execution Date', 'Ticker', 'AI Confidence', 'Trade Outcome', 'Net Profit']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2) 
    
    for (row, col), cell in table.get_celld().items():
        if row == 0: 
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#1e1e1e')
            cell.set_edgecolor('gray')
        else: 
            cell.set_edgecolor('gray')
            cell.set_facecolor('#121212')
            cell.set_text_props(color='white')
            
            if col == 3 or col == 4: 
                outcome_value = table_data[row-1][3]
                if outcome_value == 'WIN':
                    cell.set_text_props(color='lime', weight='bold')
                else:
                    cell.set_text_props(color='#ff4444', weight='bold')

    plt.title(f'V4 Trade Ledger (Sample of Last 20 Trades)', fontsize=16, fontweight='bold', color='white', pad=20)
    plt.savefig(os.path.join(PLOTS_FOLDER, 'Financial_10_Trade_Ledger_Table.png'), dpi=300, bbox_inches='tight', facecolor='#121212')
    plt.close()

# EXECUTION
def main():
    print("\n" + "="*50)
    print("INITIATING ADVANCED PLOTTING PIPELINE.")
    print("="*50)
    
    df = load_results()
    
    if df is not None:
        plot_underwater_drawdown(df)
        plot_confidence_curve(df)
        plot_equity_trade_overlay(df)
        plot_styled_trade_ledger(df)
    else:
        print("WARNING: Could not find results CSVs.")
        
    plot_feature_correlation()
    
    print("="*50)
    print("PIPELINE COMPLETE!")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()