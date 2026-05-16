import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split


def load_heart_disease_data():
    heart = fetch_ucirepo(id=45)

    X = heart.data.features.copy()
    y_raw = heart.data.targets.copy()

    y = (y_raw.iloc[:, 0] > 0).astype(int)

    df = X.copy()
    df["target"] = y
    df = df.replace("?", np.nan)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    X = df.drop(columns=["target"])
    y = df["target"]

    return X, y, df


def make_train_test_split(X, y, test_size=0.25, random_state=42):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )