import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import matplotlib.ticker as ticker
from datetime import timedelta
from sklearn.metrics import precision_score, roc_auc_score, f1_score, roc_curve, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

#  CONFIGURATION
MODELS_CONFIG = {
    "V1 (Retail Noisy)": {"folder": "walk_forward_results", "color": "#ff4444"},
    "V2 (Over-Pruned)": {"folder": "walk_forward_results_V2", "color": "#00ffcc"},
    "V3 (Soft-Pruned)": {"folder": "walk_forward_results_V3", "color": "#d4af37"},
    "V4 (Institutional)": {"folder": "walk_forward_results_V4", "color": "#ff00ff"}
}

PLOTS_FOLDER = "../backtest_plots"            

STARTING_CAPITAL = 1000.0  
BET_SIZE = 100.0           
TAKE_PROFIT_PCT = 0.03     
STOP_LOSS_PCT = -0.01      
PROFIT_PER_WIN = BET_SIZE * TAKE_PROFIT_PCT   
LOSS_PER_LOSS = BET_SIZE * STOP_LOSS_PCT      
CONFIDENCE_THRESHOLD = 0.75

if not os.path.exists(PLOTS_FOLDER): os.makedirs(PLOTS_FOLDER)

plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#121212", "figure.facecolor": "#121212", 
                                    "text.color": "white", "axes.labelcolor": "white", 
                                    "xtick.color": "white", "ytick.color": "white"})

#  CORE FUNCTIONS
def load_results(folder):
    files = glob.glob(os.path.join(folder, "Fold_*_Results.csv"))
    if not files: return None
    df_list = [pd.read_csv(f) for f in files]
    master_df = pd.concat(df_list, ignore_index=True)
    master_df['Date'] = pd.to_datetime(master_df['Date'])
    master_df.sort_values('Date', inplace=True)
    return master_df

def plot_ml_metrics(active_models):
    print("\n" + "="*50)
    print("MACHINE LEARNING EVALUATION")
    print("="*50)

    # ROC Curve
    plt.figure(figsize=(8, 6))
    for name, data in active_models.items():
        y_t, y_p = data['df']['Actual_Label'], data['df']['Predicted_Probability']
        fpr, tpr, _ = roc_curve(y_t, y_p)
        auc = roc_auc_score(y_t, y_p)
        plt.plot(fpr, tpr, color=data['color'], lw=2, label=f'{name} (AUC = {auc:.3f})')
        print(f"{name} -> ROC-AUC: {auc:.4f}")
    
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle=':')
    plt.title('Dynamic ROC Curve Comparison')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(PLOTS_FOLDER, 'ML_01_Compare_ROC.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # PR Curve
    plt.figure(figsize=(8, 6))
    for name, data in active_models.items():
        y_t, y_p = data['df']['Actual_Label'], data['df']['Predicted_Probability']
        p, r, _ = precision_recall_curve(y_t, y_p)
        plt.plot(r[::100], p[::100], color=data['color'], lw=2.5, label=name)
        
    plt.title('Dynamic Precision-Recall (Smoothed)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(PLOTS_FOLDER, 'ML_02_Compare_PR.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved ML Plot Overlays (ROC, PR)")

def plot_dynamic_dashboard(active_models, spy_data):
    print("\nSimulating Dynamic Equity Dashboard with REGIME FILTER...")
    
    spy_clean = spy_data[['Date', 'SPY_Close', 'SPY_200MA']].copy()
    spy_clean['Market_Is_Safe'] = spy_clean['SPY_Close'] > spy_clean['SPY_200MA']
    
    metrics_for_bar_chart = {}
    num_models = len(active_models)

    fig, axes = plt.subplots(num_models, 1, figsize=(16, 6 * num_models), sharex=True)
    if num_models == 1: axes = [axes]
    
    fig.suptitle(f'\nRegime-Filtered Portfolio Growth (Risk 1% to Make 3%)', fontsize=20, fontweight='bold', color='white', y=0.92)
    
    for ax, (name, data) in zip(axes, active_models.items()):
        df = pd.merge_asof(data['df'].sort_values('Date'), spy_clean.sort_values('Date'), on='Date', direction='backward')
        
        valid_trades = df[(df['Predicted_Probability'] >= CONFIDENCE_THRESHOLD) & (df['Market_Is_Safe'] == True)].copy()
        
        if len(valid_trades) > 0:
            valid_trades.sort_values('Date', inplace=True)
            valid_trades['Profit'] = np.where(valid_trades['Actual_Label'] == 1, PROFIT_PER_WIN, LOSS_PER_LOSS)
            daily_profit = valid_trades.groupby('Date')['Profit'].sum().reset_index()
            daily_profit['Capital'] = STARTING_CAPITAL + daily_profit['Profit'].cumsum()
            
            final_cap = daily_profit['Capital'].iloc[-1]
            wr = valid_trades['Actual_Label'].mean() * 100
            ev = (wr/100 * PROFIT_PER_WIN) + ((1 - wr/100) * LOSS_PER_LOSS)
            lbl = f"Portfolio (>{int(CONFIDENCE_THRESHOLD*100)}% Conf) | Trades: {len(valid_trades):,} | WR: {wr:.1f}% | EV: ${ev:.2f}"
            
            ax.plot(daily_profit['Date'], daily_profit['Capital'], label=lbl, color=data['color'], linewidth=2.5)
            metrics_for_bar_chart[name] = wr
            print(f"\n{name} -> Final Capital: ${final_cap:,.0f} | Filtered WR: {wr:.2f}% | EV: ${ev:.2f}")
        else:
            ax.text(0.5, 0.5, 'No valid trades', horizontalalignment='center', transform=ax.transAxes)
            metrics_for_bar_chart[name] = 0

        ax.plot(spy_data['Date'], spy_data['SPY_Portfolio'], label=f"S&P 500", color='magenta', linewidth=2, linestyle=':')
        ax.axhline(STARTING_CAPITAL, color='gray', linestyle='-', alpha=0.5)
        ax.set_title(name, fontsize=16, fontweight='bold')
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
        ax.legend(fontsize=12, loc='upper left', frameon=True, facecolor='#1a1a1a')

    axes[-1].set_xlabel('Date', fontsize=14)
    output_img = os.path.join(PLOTS_FOLDER, 'Financial_04_Dynamic_Equity.png')
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    plt.savefig(output_img, dpi=300)
    plt.close() 
    print(f"\nSaved Dynamic Equity Dashboard to {output_img}")

    labels = list(metrics_for_bar_chart.keys())
    precisions = list(metrics_for_bar_chart.values())
    x = np.arange(len(labels))
    colors = [active_models[l]['color'] for l in labels]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, precisions, 0.5, color=colors)
    ax.set_ylabel('Win Rate (%)')
    ax.set_title(f'Trading Performance Comparison (>{int(CONFIDENCE_THRESHOLD*100)}% Conf)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.axhline(25.0, color='red', linestyle='--', label='25% Breakeven Threshold')
    ax.legend()
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f"{yval:.1f}%", ha='center', fontweight='bold')
        
    plt.savefig(os.path.join(PLOTS_FOLDER, 'ML_03_Dynamic_Bar_Metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("\nSaved Dynamic Bar Chart")

def main():
    print("\nLoading Model Predictions.")
    active_models = {}
    
    for name, config in MODELS_CONFIG.items():
        df = load_results(config['folder'])
        if df is not None:
            active_models[name] = {"df": df, "color": config['color']}
        else:
            print(f"\nSkipping '{name}' - Could not load results from {config['folder']}")

    if not active_models:
        print("\nCRITICAL ERROR: No data found. Please check your folder paths.")
        return

    min_date = min([data['df']['Date'].min() for data in active_models.values()])
    max_date = max([data['df']['Date'].max() for data in active_models.values()])
    
    start_date = (min_date - timedelta(days=300)).strftime('%Y-%m-%d')
    end_date = (max_date + timedelta(days=5)).strftime('%Y-%m-%d')
    
    print(f"\nFetching S&P 500 data from {start_date} to {end_date} for Regime Filter.")
    spy_data = yf.download('SPY', start=start_date, end=end_date, progress=False)
    spy_data.reset_index(inplace=True)
    if isinstance(spy_data.columns, pd.MultiIndex): spy_data.columns = spy_data.columns.get_level_values(0)
    spy_data['Date'] = pd.to_datetime(spy_data['Date']).dt.tz_localize(None) 
    
    spy_data['SPY_Close'] = spy_data['Close']
    spy_data['SPY_200MA'] = spy_data['SPY_Close'].rolling(window=200).mean()
    spy_data = spy_data.dropna(subset=['SPY_200MA']) 
    
    spy_start_price = spy_data[spy_data['Date'] >= min_date]['SPY_Close'].iloc[0]
    spy_data['SPY_Portfolio'] = (spy_data['SPY_Close'] / spy_start_price) * STARTING_CAPITAL

    plot_ml_metrics(active_models)
    plot_dynamic_dashboard(active_models, spy_data)
    
    print("\nFull Dynamic Backtest Complete.")

if __name__ == "__main__":
    main()