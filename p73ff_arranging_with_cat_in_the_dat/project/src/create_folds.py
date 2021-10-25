import pandas as pd
from sklearn import model_selection

df = pd.read_csv("../data/raw/train.csv")

df["strat_fold"] = -1

df = df.sample(frac=1).reset_index(drop=True)
folds = model_selection.StratifiedKFold(n_splits=5)

for i, (train, test) in enumerate(folds.split(X=df, y=df.target)):
    df.loc[test, "strat_fold"] = i

df.to_csv("../data/processed/train_folds.csv", index=False)