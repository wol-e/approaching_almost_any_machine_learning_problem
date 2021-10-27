import argparse
import joblib
import importlib
import pandas as pd
from sklearn import metrics

from config import TRAINING_DATA_PATH

def run(model_name, fold):
    """
    runs training on provided fold, i.e. uses the training data with matching fold as test data and trains on remaining
    data.

    :param fold int: fold number to run training for
    :return: None
    """
    df_train = pd.read_csv(TRAINING_DATA_PATH)

    df_test = df_train[df_train.strat_fold == fold].reset_index(drop=True)
    df_train = df_train[df_train.strat_fold != fold].reset_index(drop=True)

    y_train = df_train.target.values
    y_test = df_test.target.values

    df_train = df_train.drop(["id", "target", "strat_fold"], axis=1)
    df_test = df_test.drop(["id", "target", "strat_fold"], axis=1)

    feature_engineering = importlib.import_module(f"models.{model_name}.feature_engineering")
    model = importlib.import_module(f"models.{model_name}.model").model

    df_train = feature_engineering.feature_pipeline(df_train)
    df_test = feature_engineering.feature_pipeline(df_test)

    model.fit(df_train, y_train)

    auc_test = metrics.roc_auc_score(y_true=y_test, y_score=model.predict_proba(df_test)[:, 1])
    auc_train = metrics.roc_auc_score(y_true=y_train, y_score=model.predict_proba(df_train)[:, 1])
    print(f"AUC on fold {fold}: Train: {auc_train}, Test: {auc_test}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    run(model_name=args.model, fold=args.fold)