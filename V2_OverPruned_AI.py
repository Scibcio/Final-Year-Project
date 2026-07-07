import pandas as pd
import numpy as np
import os
import glob
import warnings
import gc
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Conv1D, MaxPooling1D, Flatten, LSTM, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

warnings.filterwarnings('ignore')

# HARDWARE CHECK & MEMORY GROWTH
print("\n" + "="*50)
print("HARDWARE DIAGNOSTICS:")
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) == 0:
    print("STATUS: Training on CPU.")
else:
    print(f"STATUS: GPU DETECTED! Training on {physical_devices[0].name}")
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("STATUS: GPU Memory Growth Enabled.")
    except:
        pass
print("="*50 + "\n")


# CONFIGURATION

INPUT_FOLDER = "../engin_features"
OUTPUT_FOLDER = "../walk_forward_results_V2"
MODELS_FOLDER = "../saved_models_V2"

SEQ_LENGTH = 60
EPOCHS = 50
BATCH_SIZE = 2048  

if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
if not os.path.exists(MODELS_FOLDER): os.makedirs(MODELS_FOLDER)

# Based directly on the Permutation Feature Importance Chart
TOXIC_FEATURES = [
    'SMA_20', 'Close', 'SP500_200MA', 'RSI', 'SMA_100', 'Vol_20MA', 
    'Mom_20', 'Low', 'SMA_20_Dist', 'EMA_12', 'Vol_20', 'Boll_Std', 
    'ZScore_20', 'Boll_Upper', 'Mom_5', 'Boll_Pos', 'Mom_10', 
    'SMA_5_Dist', 'SMA_50_Dist', 'SMA_100_Dist', 'Volume_Surge_Ratio', 
    'MACD', 'SMA_10', 'Boll_Lower', 'Volume', 'Vol_5', 'TNX_Level', 
    'ZScore_60', 'High', 'Vol_10'
]

FOLDS = [
    {"fold": 1, "train_end": "2013-12-31", "test_start": "2014-04-01", "test_end": "2015-03-31"},
    {"fold": 2, "train_end": "2014-12-31", "test_start": "2015-04-01", "test_end": "2016-03-31"},
    {"fold": 3, "train_end": "2015-12-31", "test_start": "2016-04-01", "test_end": "2017-03-31"},
    {"fold": 4, "train_end": "2016-12-31", "test_start": "2017-04-01", "test_end": "2018-03-31"},
    {"fold": 5, "train_end": "2017-12-31", "test_start": "2018-04-01", "test_end": "2019-03-31"},
    {"fold": 6, "train_end": "2018-12-31", "test_start": "2019-04-01", "test_end": "2020-03-31"},
    {"fold": 7, "train_end": "2019-12-31", "test_start": "2020-04-01", "test_end": "2021-03-31"},
    {"fold": 8, "train_end": "2020-12-31", "test_start": "2021-04-01", "test_end": "2022-03-31"},
    {"fold": 9, "train_end": "2021-12-31", "test_start": "2022-04-01", "test_end": "2023-03-31"},
    {"fold": 10, "train_end": "2022-12-31", "test_start": "2023-04-01", "test_end": "2024-03-31"},
    {"fold": 11, "train_end": "2023-12-31", "test_start": "2024-04-01", "test_end": "2025-03-31"},
    {"fold": 12, "train_end": "2024-12-31", "test_start": "2025-04-01", "test_end": "2026-03-31"}
]


# THE GENERATOR & ARCHITECTURE

class StockDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

def build_quant_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        
        Conv1D(filters=64, kernel_size=5, activation='relu'),
        BatchNormalization(), 
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # The CuDNN bypass is kept intact
        LSTM(64, return_sequences=False, recurrent_dropout=0.01),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=0.001) 
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# WALK-FORWARD ENGINE

def process_stock_for_fold(df, ticker, train_end, test_start, test_end):
    # DROP THE TOXIC FEATURES HERE
    df_clean = df.drop(columns=[col for col in TOXIC_FEATURES if col in df.columns])
    
    feature_cols = [c for c in df_clean.columns if c not in ['Date', 'Target_Label']]
    train_mask = df_clean['Date'] <= train_end
    if train_mask.sum() < SEQ_LENGTH: return None  
    
    scaler = MinMaxScaler()
    scaler.fit(df_clean.loc[train_mask, feature_cols])
    
    scaled_features = scaler.transform(df_clean[feature_cols])
    labels = df_clean['Target_Label'].values
    dates = df_clean['Date'].values
    closes = df['Close'].values
    
    X_train, y_train, X_test, y_test = [], [], [], []
    test_dates, test_tickers, test_closes = [], [], []
    
    for i in range(SEQ_LENGTH, len(scaled_features)):
        window = scaled_features[i - SEQ_LENGTH : i]
        target_date = dates[i]
        label = labels[i]
        
        if target_date <= train_end:
            X_train.append(window)
            y_train.append(label)
        elif target_date >= test_start and target_date <= test_end:
            X_test.append(window)
            y_test.append(label)
            test_dates.append(target_date)
            test_tickers.append(ticker)
            test_closes.append(closes[i])
            
    if not X_train: return None
    
    return (np.array(X_train, dtype=np.float16), np.array(y_train, dtype=np.int8), 
            np.array(X_test, dtype=np.float16), np.array(y_test, dtype=np.int8), 
            test_dates, test_tickers, test_closes)

def main():
    feature_files = glob.glob(os.path.join(INPUT_FOLDER, "*_features.csv"))
    if not feature_files:
        print("\nCRITICAL ERROR: No feature files found.")
        return

    print("\nLoading all stock data into memory...")
    all_stocks = {}
    for f in feature_files:
        ticker = os.path.basename(f).replace("_features.csv", "")
        all_stocks[ticker] = pd.read_csv(f)

    for fold_config in FOLDS:
        fold_num = fold_config['fold']
        train_end = fold_config['train_end']
        test_start = fold_config['test_start']
        test_end = fold_config['test_end']
        
        print(f"\n{'='*50}")
        print(f"RUNNING FOLD {fold_num}/12 (V2 PRUNED ARCHITECTURE)")
        print(f"{'='*50}")
        
        X_train_global, y_train_global = [], []
        X_test_global, y_test_global = [], []
        test_dates_global, test_tickers_global, test_closes_global = [], [], []
        
        for ticker, df in all_stocks.items():
            result = process_stock_for_fold(df, ticker, train_end, test_start, test_end)
            if result:
                X_tr, y_tr, X_te, y_te, t_dates, t_tickers, t_closes = result
                X_train_global.append(X_tr)
                y_train_global.append(y_tr)
                if len(X_te) > 0:
                    X_test_global.append(X_te)
                    y_test_global.append(y_te)
                    test_dates_global.extend(t_dates)
                    test_tickers_global.extend(t_tickers)
                    test_closes_global.extend(t_closes)
                
        if not X_train_global: continue
            
        print("\nAllocating memory-mapped SSD files...")
        total_train = sum(len(x) for x in X_train_global)
        
        train_x_file = os.path.join(OUTPUT_FOLDER, f"temp_x_train_f{fold_num}.dat")
        train_y_file = os.path.join(OUTPUT_FOLDER, f"temp_y_train_f{fold_num}.dat")
        test_x_file = os.path.join(OUTPUT_FOLDER, f"temp_x_test_f{fold_num}.dat")
        test_y_file = os.path.join(OUTPUT_FOLDER, f"temp_y_test_f{fold_num}.dat")

        X_train_arr = np.memmap(train_x_file, dtype=np.float16, mode='w+', shape=(total_train, SEQ_LENGTH, X_train_global[0].shape[2]))
        y_train_arr = np.memmap(train_y_file, dtype=np.int8, mode='w+', shape=(total_train,))

        idx = 0
        while X_train_global:
            chunk_x = X_train_global.pop(0) 
            chunk_y = y_train_global.pop(0)
            size = len(chunk_x)
            X_train_arr[idx:idx+size] = chunk_x
            y_train_arr[idx:idx+size] = chunk_y
            idx += size

        X_train_arr.flush()
        y_train_arr.flush()

        total_ones = np.sum(y_train_arr == 1, dtype=np.int64)
        total_zeros = total_train - total_ones
        weight_0 = (1 / total_zeros) * (total_train / 2.0)
        weight_1 = (1 / total_ones) * (total_train / 2.0)
        class_weights = {0: weight_0, 1: weight_1}
        print(f"\nImbalance: {total_zeros:,} Losses vs {total_ones:,} Wins | Weights -> 0: {weight_0:.2f}, 1: {weight_1:.2f}")

        total_test = sum(len(x) for x in X_test_global) if X_test_global else 0
        if total_test > 0:
            X_test_arr = np.memmap(test_x_file, dtype=np.float16, mode='w+', shape=(total_test, SEQ_LENGTH, X_test_global[0].shape[2]))
            y_test_arr = np.memmap(test_y_file, dtype=np.int8, mode='w+', shape=(total_test,))
            idx = 0
            while X_test_global:
                chunk_x = X_test_global.pop(0)
                chunk_y = y_test_global.pop(0)
                size = len(chunk_x)
                X_test_arr[idx:idx+size] = chunk_x
                y_test_arr[idx:idx+size] = chunk_y
                idx += size
            X_test_arr.flush()
            y_test_arr.flush()
        else:
            X_test_arr, y_test_arr = np.array([]), np.array([])
            
        gc.collect() 
        
        model = build_quant_model(input_shape=(SEQ_LENGTH, X_train_arr.shape[2]))
        
        model_save_path = os.path.join(MODELS_FOLDER, f"Quant_Model_V2_Fold_{fold_num}.keras")
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001, verbose=1),
            ModelCheckpoint(filepath=model_save_path, monitor='val_loss', save_best_only=True, verbose=0)
        ]
        
        split_idx = int(len(X_train_arr) * 0.85)
        train_gen = StockDataGenerator(X_train_arr[:split_idx], y_train_arr[:split_idx], BATCH_SIZE)
        val_gen = StockDataGenerator(X_train_arr[split_idx:], y_train_arr[split_idx:], BATCH_SIZE)

        model.fit(
            train_gen, 
            epochs=EPOCHS, 
            validation_data=val_gen, 
            callbacks=callbacks, 
            class_weight=class_weights,  
            workers=4, 
            use_multiprocessing=False, 
            max_queue_size=10, 
            verbose=1
        )
        
        if len(X_test_arr) > 0:
            print("\nPredicting Test Year.")
            test_gen = StockDataGenerator(X_test_arr, np.zeros(len(X_test_arr)), BATCH_SIZE)
            
            predictions = model.predict(test_gen, workers=4, use_multiprocessing=False, max_queue_size=10).flatten()
            
            results_df = pd.DataFrame({'Date': test_dates_global, 'Ticker': test_tickers_global, 'Close_Price': test_closes_global, 'Actual_Label': y_test_arr, 'Predicted_Probability': predictions})
            output_file = os.path.join(OUTPUT_FOLDER, f"Fold_{fold_num}_Results.csv")
            results_df.to_csv(output_file, index=False)
            print(f"\nSaved to {output_file}")
        
        K.clear_session()
        del X_train_arr, y_train_arr, X_test_arr, y_test_arr
        del train_gen, val_gen
        gc.collect()
        
        print("\nCleaning up SSD temporary files.")
        if os.path.exists(train_x_file): os.remove(train_x_file)
        if os.path.exists(train_y_file): os.remove(train_y_file)
        if os.path.exists(test_x_file): os.remove(test_x_file)
        if os.path.exists(test_y_file): os.remove(test_y_file)

    print(f"\nSaved models are in '{MODELS_FOLDER}'")

if __name__ == "__main__":
    main()