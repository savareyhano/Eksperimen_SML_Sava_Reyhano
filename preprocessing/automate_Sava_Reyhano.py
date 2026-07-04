import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def load_data(file_path):
    """Memuat dataset dari path."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Membersihkan data dari nilai kosong dan duplikat."""
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def preprocess_features(df):
    """Melakukan preprocessing pada fitur (encoding dan scaling)."""
    # Encoding target
    le = LabelEncoder()
    df['class'] = le.fit_transform(df['class'])
    
    # Standarisasi fitur numerik
    scaler = StandardScaler()
    numeric_cols = ['age', 'impluse', 'pressurehight', 'pressurelow', 'glucose', 'kcm', 'troponin']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

def run_preprocessing(input_path, output_path):
    """Menjalankan seluruh alur preprocessing."""
    print("Memulai preprocessing data...")
    df = load_data(input_path)
    print(f"Data awal: {df.shape}")
    
    df = clean_data(df)
    print(f"Data setelah cleaning: {df.shape}")
    
    df = preprocess_features(df)
    print("Fitur berhasil di-encode dan di-scale.")
    
    # Pastikan direktori output ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data bersih disimpan ke: {output_path}")

if __name__ == "__main__":
    input_file = "../dataset/Heart Attack.csv"
    output_file = "../dataset/Heart_Attack_Clean.csv"
    
    run_preprocessing(input_file, output_file)
