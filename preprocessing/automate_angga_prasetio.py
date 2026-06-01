import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def preprocessing():
    print("Load data...")

    df = pd.read_csv(r'D:/STUPEND(Pijak)/Eksperimen_SML_Angga Prasetio/Klasifikasi Tingkat Kemiskinan di Indonesia.csv', 
                     sep=';', decimal=',')

    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    print("Kolom setelah cleaning:")
    print(df.columns)

    df = df.drop(columns=['provinsi', 'kab/kota'], errors='ignore')

    print("Cleaning data...")
    df = df.dropna()
    df = df.drop_duplicates()

    print("Split fitur & target...")
    X = df.drop('klasifikasi_kemiskinan', axis=1)
    y = df['klasifikasi_kemiskinan']

    print("Convert data numerik...")

    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].str.replace(',', '.', regex=False)

    X = X.apply(pd.to_numeric, errors='coerce')


    X = X.dropna()
    y = y.loc[X.index]

    print("Scaling data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Gabungkan kembali...")
    df_clean = pd.DataFrame(X_scaled, columns=X.columns)
    df_clean['klasifikasi_kemiskinan'] = y.values

    print("Simpan hasil...")
    os.makedirs('preprocessing', exist_ok=True)
    df_clean.to_csv('preprocessing/dataset_preprocessing.csv', index=False)

    print("Preprocessing selesai!")

if __name__ == "__main__":
    preprocessing()