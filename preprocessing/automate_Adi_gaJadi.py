from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def preprocessing_pipeline(data, target_column, save_path, file_path):

    # 1. Cleaning Awal
    data = data.drop_duplicates()
    if 'student_id' in data.columns:
        data = data.drop(columns=['student_id']) 
    data = data.dropna()

    # 2. Penanganan Outlier (IQR) 
    num_cols = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
    for feature in num_cols:
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[feature] >= lower_bound) & (data[feature] <= upper_bound)] #

    # 3. Binning 'sleep_hours' 
    data['sleep_hours_binned'] = pd.cut(
        data['sleep_hours'], 
        bins=3, 
        labels=['Kurang', 'Cukup', 'Baik']
    ) 
    
    # Drop kolom asli yang sudah di-binning agar tidak masuk ke scaler
    data = data.drop(columns='sleep_hours') 


    # Mendapatkan nama kolom untuk fitur (tanpa target)
    column_names = data.columns.drop(target_column) 

    # Menyimpan nama kolom sebagai header tanpa data (CSV)
    df_header = pd.DataFrame(columns=column_names)
    df_header.to_csv(file_path, index=False) 
    print(f"Nama kolom berhasil disimpan ke: {file_path}")

    # Menentukan fitur numerik dan kategoris secara otomatis
    # Sesuai eksperimenmu: age, study_hours, dan class_attendance masuk scaling
    numeric_features = ['age', 'study_hours', 'class_attendance']
    categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()

    if target_column in categorical_features:
        categorical_features.remove(target_column)

    # Pipeline untuk fitur numerik
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()) 
    ])

    # Pipeline untuk fitur kategorik
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Column Transformer (Menggabungkan num dan cat)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Memisahkan target (y) dan fitur (X)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Membagi data (Splitting 80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    ) #

    # Fitting dan transformasi data 
    # Fit pada training set, transform pada training dan testing set
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Simpan objek preprocessor (Pipeline) ke file .joblib
    dump(preprocessor, save_path)
    print(f"Pipeline preprocessing disimpan ke: {save_path}")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    try:
        df_raw = pd.read_csv('Exam_Score_Prediction_raw.csv')
        # Menjalankan fungsi dengan target 'exam_score'
        X_train, X_test, y_train, y_test = preprocess_data(
            data=df_raw, 
            target_column='exam_score', 
            save_path='preprocessor_adi.joblib', 
            file_path='data_header_adi.csv'
        )
        print("Proses Automasi Preprocessing Selesai!")
    except Exception as e:
        print(f"Gagal menjalankan otomasi: {e}")