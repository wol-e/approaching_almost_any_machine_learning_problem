import argparse
import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree

from config import TRAINING_DATA_PATH

def run(fold):
    """
    runs training on provided fold, i.e. uses the training data with matching fold as test data and trains on remaining
    data.

    :param fold int: fold number to run training for
    :return: None
    """
    df_train = pd.read_csv(TRAINING_DATA_PATH)

    ######### TODO: handle cats
    numeric_features = [c for c in df_train.columns if df_train[c].dtype == "float64"]
    df_train = df_train[numeric_features + ["strat_fold", "target"]]
    #########

    df_test = df_train[df_train.strat_fold == fold].reset_index(drop=True)
    df_train = df_train[df_train.strat_fold != fold].reset_index(drop=True)

    y_train = df_train.target.values
    y_test = df_test.target.values

    df_train = df_train.drop(["target", "strat_fold"], axis=1)
    df_test = df_test.drop(["target", "strat_fold"], axis=1)

    # TODO: proper nan imputation
    df_train, df_test = df_train.fillna(0), df_test.fillna(0)

    model = tree.DecisionTreeClassifier()
    model.fit(df_train, y_train)
    predictions = model.predict(df_test)

    auc = metrics.roc_auc_score(y_true=y_test, y_score=predictions)
    print(f"AUC on fold {fold}: {auc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    run(fold=args.fold)