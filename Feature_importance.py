import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import warnings

warnings.filterwarnings('ignore')

plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#121212", "figure.facecolor": "#121212", 
                                    "text.color": "white", "axes.labelcolor": "white", 
                                    "xtick.color": "white", "ytick.color": "white"})

# CONFIGURATION
# Add, remove, or comment out models here
MODELS_CONFIG = {
    "V1 (Retail Noisy)": {
        "model_path": "saved_models/Quant_Model_Fold_12.keras",
        "data_folder": "engin_features",
        "drop_list": []
    },
     "V2 (Over-Pruned)": {
         "model_path": "saved_models_V2/Quant_Model_V2_Fold_12.keras",
         "data_folder": "engin_features",
         "drop_list": ['SMA_20', 'Close', 'SP500_200MA', 'RSI', 'SMA_100', 'Vol_20MA', 'Mom_20', 'Low', 'SMA_20_Dist', 'EMA_12', 'Vol_20', 'Boll_Std', 'ZScore_20', 'Boll_Upper', 'Mom_5', 'Boll_Pos', 'Mom_10', 'SMA_5_Dist', 'SMA_50_Dist', 'SMA_100_Dist', 'Volume_Surge_Ratio', 'MACD', 'SMA_10', 'Boll_Lower', 'Volume', 'Vol_5', 'TNX_Level', 'ZScore_60', 'High', 'Vol_10']
     },
     "V3 (Soft-Pruned)": {
         "model_path": "saved_models_V3/Quant_Model_V3_Fold_12.keras",
         "data_folder": "engin_features",
         "drop_list": ['Close', 'High', 'Low', 'SMA_20', 'SP500_200MA']
     },
    "V4 (Institutional)": {
        "model_path": "saved_models_V4/Quant_Model_V4_Fold_12.keras",
        "data_folder": "engin_features_V4",
        "drop_list": []
    }
}

PLOTS_FOLDER = "../backtest_plots"
SEQ_LENGTH = 60
SAMPLE_STOCKS = 15  

if not os.path.exists(PLOTS_FOLDER): os.makedirs(PLOTS_FOLDER)

# CORE FUNCTIONS
def load_data(data_folder, drop_list):
    files = glob.glob(os.path.join(data_folder, "*_features.csv"))[:SAMPLE_STOCKS]
    if not files: return None, None, None
    
    X_list, y_list, feature_names = [], [], []
    
    for f in files:
        df = pd.read_csv(f)
        df['Date'] = pd.to_datetime(df['Date'])
        test_mask = df['Date'] >= "2025-01-01"
        if test_mask.sum() < SEQ_LENGTH: continue
            
        if drop_list:
            df = df.drop(columns=[col for col in drop_list if col in df.columns])
            
        feature_cols = [c for c in df.columns if c not in ['Date', 'Target_Label']]
        if not feature_names: feature_names = feature_cols
        
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(df[feature_cols])
        labels = df['Target_Label'].values
        
        for i in range(SEQ_LENGTH, len(scaled_features)):
            if test_mask.iloc[i]:
                X_list.append(scaled_features[i - SEQ_LENGTH : i])
                y_list.append(labels[i])
                
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int8), feature_names

def run_permutation(config):
    model_path = config['model_path']
    data_folder = config['data_folder']
    drop_list = config['drop_list']
    
    if not os.path.exists(model_path):
        print(f"  -> Skipping. File not found: {model_path}")
        return {}
        
    model = tf.keras.models.load_model(model_path)
    X, y, names = load_data(data_folder, drop_list)
    
    if X is None or len(X) == 0:
        print(f"  -> Skipping. No data found in {data_folder}")
        return {}
    
    baseline_loss, _ = model.evaluate(X, y, verbose=0, batch_size=2048)
    importances = {}
    
    for idx, name in enumerate(names):
        X_shuffled = X.copy()
        shape = X_shuffled[:, :, idx].shape
        flat = X_shuffled[:, :, idx].flatten()
        np.random.shuffle(flat)
        X_shuffled[:, :, idx] = flat.reshape(shape)
        
        shuffled_loss, _ = model.evaluate(X_shuffled, y, verbose=0, batch_size=2048)
        importances[name] = shuffled_loss - baseline_loss
        
    return importances

def main():
    print("\n" + "="*50)
    print("DYNAMIC FEATURE HEATMAP GENERATOR")
    print("="*50)
    
    all_importances = {}
    
    # Loop through the active models in the config
    for name, config in MODELS_CONFIG.items():
        print(f"\nScanning {name}...")
        imp = run_permutation(config)
        if imp:
            all_importances[name] = imp
    
    if not all_importances:
        print("CRITICAL ERROR: No models were successfully scanned.")
        return

    df_imp = pd.DataFrame(all_importances)
    
    last_model_name = df_imp.columns[-1]
    df_imp = df_imp.sort_values(by=last_model_name, ascending=False)
    
    # Dynamic Figure Sizing based on feature count
    fig_height = max(10, len(df_imp) * 0.4)
    plt.figure(figsize=(12, fig_height))
    
    # Custom colormap: Red (Toxic), Black (Zero), Green (Alpha)
    cmap = sns.diverging_palette(10, 150, as_cmap=True, center="dark")
    
    sns.heatmap(df_imp, cmap=cmap, center=0, annot=True, fmt=".4f", 
                linewidths=.5, cbar_kws={"shrink": .8, "label": "Importance Score"})
    
    active_model_names = " vs ".join(all_importances.keys())
    plt.title(f'{active_model_names}\nFeature Importance Heatmap\n(Green = Drives Profit | Red = Creates Noise | Blank = Not Used)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Features', fontsize=14)
    plt.xlabel('Model Architecture', fontsize=14)
    
    output_img = os.path.join(PLOTS_FOLDER, 'ML_05_Dynamic_Heatmap.png')
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    print(f"\nSaved Dynamic Heatmap to {output_img}")

if __name__ == "__main__":
    main()