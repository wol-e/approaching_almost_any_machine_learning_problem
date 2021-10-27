from sklearn.preprocessing import OrdinalEncoder
import numpy as np


def feature_pipeline(df_train, df_test):
    df_train = nan_imputation(df_train)
    df_test = nan_imputation(df_test)
    df_train, df_test = encode_objects(df_train, df_test)

    return df_train, df_test


def encode_objects(df_train, df_test):
    for col in df_train.columns:
        if df_train[col].dtype == "object":
            df_train[col] = df_train[col].astype(str)
            df_test[col] = df_test[col].astype(str)

            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            encoder.fit(df_train[col].values.reshape(-1,1))
            df_train[col] = encoder.transform(df_train[col].values.reshape(-1,1))
            df_test[col] = encoder.transform(df_test[col].values.reshape(-1,1))

    return df_train, df_test

def nan_imputation(df):
    return df.fillna("NONE")
