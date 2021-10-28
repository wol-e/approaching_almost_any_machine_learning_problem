from sklearn.preprocessing import OrdinalEncoder

import sys

sys.path.append("...")  # TODO: to find the config. there must be a better way.
from config import TRAINING_FEATURES


class FeaturePipeline():
    def __init__(self):
        self.encoder = Encoder()

    def fit(self, df):
        df = df[TRAINING_FEATURES]
        df = nan_imputation(df)
        self.encoder.fit(df)

    def transform(self, df):
        df = df[TRAINING_FEATURES]
        df = nan_imputation(df)
        df = self.encoder.transform(df)

        return df


class Encoder:
    def __init__(self):
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    def fit(self, df):
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype(str)

        self.encoder.fit(df)

    def transform(self, df):
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype(str)
        return self.encoder.transform(df)

        #    encoder.fit(df_train[col].values.reshape(-1,1))
        #    df_train[col] = encoder.transform(df_train[col].values.reshape(-1,1))
        #    df_test[col] = encoder.transform(df_test[col].values.reshape(-1,1))

def nan_imputation(df):
    return df.fillna("NONE")
