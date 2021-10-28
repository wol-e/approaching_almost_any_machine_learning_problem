import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler

import sys

sys.path.append("...")  # TODO: to find the config. there must be a better way.
from config import TRAINING_FEATURES


def feature_pipeline(df_train, df_test, encoder=None):
    df_train = df_train[TRAINING_FEATURES]
    df_train = nan_imputation(df_train)
    if df_test is not None:
        df_test = df_test[TRAINING_FEATURES]
        df_test = nan_imputation(df_test)
    if not encoder:
        encoder = Encoder()
        encoder.fit(df_train, df_test)
    df_train = encoder.transform(df_train)
    df_train = scale_features(df_train)
    if df_test is not None:
        df_test = encoder.transform(df_test)
        df_test = scale_features(df_test)

    return df_train, df_test


class Encoder:
    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown="ignore")

    def fit(self, df_train, df_test=None):
        for col in df_train.columns:
            if df_train[col].dtype == "object":
                df_train[col] = df_train[col].astype(str)
                if df_test is not None:
                    df_test[col] = df_test[col].astype(str)

        self.encoder.fit(pd.concat([df_train, df_test]))

    def transform(self, df):
        return self.encoder.transform(df)


def scale_features(df):
    scaler = StandardScaler(with_mean=False)
    return scaler.fit_transform(df)


def nan_imputation(df):
    return df.fillna("NONE")
