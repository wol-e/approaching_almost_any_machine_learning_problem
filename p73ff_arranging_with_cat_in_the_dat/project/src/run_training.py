from train import run

save_model=True
run(model_name="logistic_regression", fold=0, save_model=save_model)
run(model_name="logistic_regression", fold=1, save_model=save_model)
run(model_name="logistic_regression", fold=2, save_model=save_model)
run(model_name="logistic_regression", fold=3, save_model=save_model)
run(model_name="logistic_regression", fold=4, save_model=save_model)