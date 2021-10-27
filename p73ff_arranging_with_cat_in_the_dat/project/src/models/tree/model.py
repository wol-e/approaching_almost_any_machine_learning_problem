from sklearn import tree

model = tree.DecisionTreeClassifier(
    criterion="entropy",
    max_depth=10,
    min_samples_leaf=25
)
