import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import argparse
import os

def preprocess_data(input_path, output_path):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    print("Preprocessing data...")
    # 1. Menghapus Data Kosong
    df = df.dropna()
    
    # 2. Menghapus Data Duplikat
    df = df.drop_duplicates()
    
    # 3. Encoding Label Kategorikal
    le = LabelEncoder()
    if 'class' in df.columns:
        df['class'] = le.fit_transform(df['class'])
        
    # Memisahkan Fitur dan Target
    X = df.drop('class', axis=1)
    y = df['class']
    
    # 4. Standarisasi Fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Menggabungkan kembali untuk hasil akhir
    df_processed = pd.concat([X_scaled_df, y.reset_index(drop=True)], axis=1)
    
    print(f"Saving processed data to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_processed.to_csv(output_path, index=False)
    print("Preprocessing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate Data Preprocessing")
    parser.add_argument("--input", type=str, default="../Heart Attack.csv", help="Path to input dataset")
    parser.add_argument("--output", type=str, default="../dataset_preprocessing/processed_data.csv", help="Path to output processed dataset")
    
    args = parser.parse_args()
    preprocess_data(args.input, args.output)
