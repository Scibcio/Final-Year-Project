# Quantitative Trading AI: A CNN-LSTM Approach

## Executive Summary
This repository contains the codebase for a final-year academic dissertation exploring machine learning applications in algorithmic trading. The project demonstrates the chronological evolution of a trading algorithm, starting from a flawed retail-indicator prototype (V1) and iteratively developing into an institutional-grade, hybrid CNN-LSTM deep learning pipeline (V4). 

The final system completely eliminates look-ahead bias using a strict T+1 feature shift, validates robustness through 12 chronological Walk-Forward folds (2014-2026), and incorporates a dynamic Monte Carlo risk-management framework.

## The Iterative Methodology
To prove the necessity of institutional data and rigorous validation, this project was built in four distinct phases:

* V1 (Retail Baseline): Built using standard retail technical indicators (RSI, MACD, etc.). Initial tests showed artificially high win rates due to time-series data leakage.
* V2 (The Correction): Implemented strict Walk-Forward Validation to fix the data leak. The true win rate collapsed, proving that raw retail price data lacks predictive machine learning edge. Highly correlated features were over-pruned.
* V3 (Architecture Upgrade): Upgraded to a hybrid CNN-LSTM architecture with heavy (40%) dropout regularization to prevent neural network hyper-fixation.
* V4 ( Final Pipeline): Discarded standard indicators in favor of Institutional Footprints (20-Day Beta, VWAP Distance, Normalized ATR, and Volume Surges). Added an S&P 500 200-SMA Regime Filter to halt trading during macro bear markets. 

## Repository Structure
**(Note: Due to university Moodle upload limits, the raw CSV data files, generated features, and compiled neural network weights have been hosted on a secure external cloud drive).**

**(https://uogcloud-my.sharepoint.com/:f:/g/personal/ms2709h_gre_ac_uk/IgA-bFAaQsQuQbFSex5zcrTgAUBZrkaqJgH0C3kUHlsoGfM?e=1HwP8W)**

The ZIP file submitted to Moodle contains the pure Python logic engines and interactive dashboards used to process that data.

Final_Year_Project_Code
 |
 |-- 01_Data_Collection
 |    |-- get_sp500.py                 (Scrapes Wikipedia for S&P 500 constituents)
 |    |-- Stock_data.py                (Ingests 15 years of daily OHLCV & Macro Baseline data)
 |    |-- Volume_check.py              (Audits raw data for NaN values and sequence integrity)
 |
 |-- 02_Iterative_Models
 |    |-- Feature_Engine_V1_V3.py      (The deprecated feature engine for early models)
 |    |-- V1_Retail_Baseline_AI.py     (Initial prototype)
 |    |-- V2_OverPruned_AI.py          (Strict walk-forward but data-starved)
 |    |-- V3_Hybrid_Dropout_AI.py      (Introduction of CNN-LSTM spatial/temporal logic)
 |
 |-- 03_Final_V4_Pipeline
 |    |-- Feature_code_V4.py           (Generates pure institutional features + T+1 Shift)
 |    |-- V4_Institutional_AI.py       (Trains the final model across 12 chronological folds)
 |
 |-- 04_Evaluation_and_Dashboards
 |    |-- Backtest_Comparisons.py      (Generates dynamic ROC/PR curves comparing V1-V4)
 |    |-- Feature_Importance.py        (Generates permutation importance heatmaps)
 |    |-- Monte_Carlo_Sim.py           (Runs 1,000 universe simulations to test Risk of Ruin)
 |    |-- Advanced_Plots_Master.py     (Generates static underwater drawdown & trade ledgers)
 |    |-- Quant_Dashboard.py           (Interactive Streamlit app for real-time stress testing)
 |
 |-- Final_Report_Assets               (High-resolution charts and the V4 Trade Ledger CSV)
 |-- README.md                         (Project documentation)


## Key Findings & Results (V4)
Through rigorous Walk-Forward testing spanning over a decade of out-of-sample data:
1. Predictive Edge: The V4 model achieved a 36.5% win rate on a highly asymmetric 1:3 Risk-to-Reward ratio (Risking 1% to make 3%). 
2. Breakeven Threshold: Because winners are 3x larger than losers, the mathematical breakeven point is 25%. The AI consistently performs approximately 11.5% above breakeven.
3. Risk Management: Monte Carlo simulations (1,000 parallel universes) confirmed that filtering trades through a 0.75 AI Confidence Threshold drops the Risk of Ruin to near-zero when risking 1% of portfolio capital per trade.

## Interactive Dashboard
To audit the algorithm's decisions and stress-test the risk parameters dynamically, launch the Streamlit dashboard via your terminal:
> cd 04_Evaluation_and_Dashboards
> streamlit run Quant_Dashboard.py

## Academic Disclaimer
This codebase was developed purely for academic research purposes as part of a university dissertation. It is not financial advice. The models simulate historical market conditions but do not account for live order-book latency, real-world execution slippage, or broker commission friction.