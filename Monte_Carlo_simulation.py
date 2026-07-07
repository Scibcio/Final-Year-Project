import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.ticker as ticker

# MONTE CARLO CONFIGURATION
# We plug in your exact V4 metrics here
WIN_RATE = 0.365
STARTING_CAPITAL = 1000.0    
BET_SIZE = 100.0             

TAKE_PROFIT_PCT = 0.03       
STOP_LOSS_PCT = -0.01        

PROFIT_PER_WIN = BET_SIZE * TAKE_PROFIT_PCT
LOSS_PER_LOSS = BET_SIZE * STOP_LOSS_PCT

SIMULATIONS = 1000
TRADES_PER_SIM = 500

PLOTS_FOLDER = "../backtest_plots"
if not os.path.exists(PLOTS_FOLDER):
    os.makedirs(PLOTS_FOLDER)

plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#121212", "figure.facecolor": "#121212", 
                                    "text.color": "white", "axes.labelcolor": "white", 
                                    "xtick.color": "white", "ytick.color": "white"})

def run_monte_carlo():
    print("\n" + "="*50)
    print(f"RUNNING MONTE CARLO STRESS TEST ({SIMULATIONS:,} Universes)")
    print("="*50)
    
    # Store the results of all 1000 simulations
    all_equity_curves = np.zeros((SIMULATIONS, TRADES_PER_SIM + 1))
    all_equity_curves[:, 0] = STARTING_CAPITAL
    
    max_drawdowns = []
    ruined_count = 0
    
    for i in range(SIMULATIONS):
        # Generate an entire career of random trades (1 = Win, 0 = Loss) based on your Win Rate
        random_trades = np.random.choice([1, 0], size=TRADES_PER_SIM, p=[WIN_RATE, 1 - WIN_RATE])
        
        # Convert 1s and 0s to actual dollar amounts
        cash_flows = np.where(random_trades == 1, PROFIT_PER_WIN, LOSS_PER_LOSS)
        
        # Calculate the cumulative equity curve
        equity_curve = STARTING_CAPITAL + np.cumsum(cash_flows)
        all_equity_curves[i, 1:] = equity_curve
        
        # Check for Ruin
        if np.any(equity_curve <= 0):
            ruined_count += 1
            
        # Calculate Maximum Drawdown
        running_max = np.maximum.accumulate(all_equity_curves[i])
        drawdown = running_max - all_equity_curves[i]
        max_drawdowns.append(np.max(drawdown))
        
    return all_equity_curves, max_drawdowns, ruined_count

def plot_results(all_equity_curves, max_drawdowns, ruined_count):
    # Calculate key metrics
    risk_of_ruin = (ruined_count / SIMULATIONS) * 100
    avg_drawdown = np.mean(max_drawdowns)
    worst_drawdown = np.max(max_drawdowns)
    median_final_capital = np.median(all_equity_curves[:, -1])
    
    print(f"Risk of Ruin (Blowing Account): {risk_of_ruin:.1f}%")
    print(f"Average Max Drawdown: ${avg_drawdown:.2f}")
    print(f"Absolute Worst Drawdown: ${worst_drawdown:.2f}")
    print(f"Median Final Capital: ${median_final_capital:,.2f}")
    
    # Create a 2-Panel Dashboard
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f'Monte Carlo Stress Test: 36.5% Win Rate | 1:3 R/R\n(Risking $100 per trade)', 
                 fontsize=20, fontweight='bold', color='white')

    # Panel 1: The "Spaghetti" Chart
    # Plotting fewer lines
    for i in range(min(100, SIMULATIONS)): 
        axes[0].plot(all_equity_curves[i], color='#00ffcc', alpha=0.15, linewidth=1)
    
    # Plot the "Average" expectation line in bright magenta
    mean_curve = np.mean(all_equity_curves, axis=0)
    axes[0].plot(mean_curve, color='#ff00ff', linewidth=3, label=f'Average Trajectory (Final: ${mean_curve[-1]:,.0f})')
    axes[0].axhline(STARTING_CAPITAL, color='gray', linestyle='--', linewidth=2)
    
    axes[0].set_title('100 Possible Equity Curves (True Variance)', fontsize=14)
    axes[0].set_xlabel('Number of Trades')
    axes[0].set_ylabel('Portfolio Value ($)')
    axes[0].yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
    axes[0].legend()

    # Panel 2: The Dynamically Scaled Drawdown Histogram
    sns.histplot(max_drawdowns, bins=40, ax=axes[1], color='#ff4444', kde=True)
    axes[1].axvline(avg_drawdown, color='yellow', linestyle='--', linewidth=2, label=f'Avg Drawdown: ${avg_drawdown:.0f}')
    
    axes[1].set_title(f'Risk Profile: Maximum Drawdown Distribution\nRisk of Ruin: {risk_of_ruin:.1f}%', fontsize=14)
    axes[1].set_xlabel('Maximum Drawdown ($)')
    axes[1].set_ylabel('Frequency (Number of Universes)')
    axes[1].legend()

    plt.tight_layout()
    output_img = os.path.join(PLOTS_FOLDER, 'Financial_06_Monte_Carlo.png')
    plt.savefig(output_img, dpi=300)
    print(f"\nSaved Monte Carlo Analysis to {output_img}")

def main():
    curves, drawdowns, ruined = run_monte_carlo()
    plot_results(curves, drawdowns, ruined)

if __name__ == "__main__":
    main()