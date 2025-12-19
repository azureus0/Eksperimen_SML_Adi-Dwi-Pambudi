import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocessing_pipeline(
    input_path,
    output_dir="Exam_Score_preprocessing",
    test_size=0.1,
    random_state=42
):
    # 1. Load dataset
    df = pd.read_csv(input_path)

    # 2. Drop duplicates
    df = df.drop_duplicates()

    # 3. Drop missing values
    df = df.dropna()

    # 4. Drop kolom ID
    df = df.drop(columns=['student_id'])

    # 5. Drop outlier (IQR)
    features_to_check = ['study_hours', 'sleep_hours']

    for feature in features_to_check:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]

    # 6. Binning sleep_hours
    df['sleep_hours_binned'] = pd.cut(
        df['sleep_hours'],
        bins=3,
        labels=['Kurang', 'Cukup', 'Baik']
    )

    # 7. Drop sleep_hours
    df = df.drop(columns=['sleep_hours'])

    # 8. Split X & y
    X = df.drop(columns=['exam_score'])
    y = df['exam_score']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    # 9. One-Hot Encoding
    categorical_cols = X_train.select_dtypes(
        include=['object', 'category']
    ).columns

    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    ).set_output(transform="pandas")

    X_train_cat = encoder.fit_transform(X_train[categorical_cols])
    X_test_cat  = encoder.transform(X_test[categorical_cols])

    X_train = X_train.drop(columns=categorical_cols)
    X_test  = X_test.drop(columns=categorical_cols)

    X_train = pd.concat([X_train, X_train_cat], axis=1)
    X_test  = pd.concat([X_test,  X_test_cat], axis=1)

    # 10. Scaling numerik
    numerical_cols = ['age', 'study_hours', 'class_attendance']

    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols]  = scaler.transform(X_test[numerical_cols])

    # 11. Gabungkan & save
    train_df = pd.concat(
        [X_train, y_train.reset_index(drop=True)],
        axis=1
    )

    test_df = pd.concat(
        [X_test, y_test.reset_index(drop=True)],
        axis=1
    )

    os.makedirs(output_dir, exist_ok=True)

    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)

    return train_df, test_df


if __name__ == "__main__":
    preprocessing_pipeline("Exam_Score_raw.csv")
