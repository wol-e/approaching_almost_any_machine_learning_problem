from sklearn.preprocessing import OneHotEncoder


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

    encoder = OneHotEncoder(handle_unknown="ignore")
    encoder.fit(df_train)
    df_train = encoder.transform(df_train)
    df_test = encoder.transform(df_test)

    return df_train, df_test

def nan_imputation(df):
    return df.fillna("NONE")
