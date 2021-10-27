from sklearn.preprocessing import LabelEncoder


def feature_pipeline(df):
    df = encode_objects(df)
    df = nan_imputation(df)

    return df


def encode_objects(df):
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = LabelEncoder().fit_transform(df[col])

    return df

def nan_imputation(df):
    return df.fillna(-1)
