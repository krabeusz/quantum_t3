# modelling_code.py
import pandas as pd
import numpy as np

import sklearn.metrics
import autosklearn.regression
from sklearn.model_selection import train_test_split

def main():
    dataset_path = "internship_train.csv"
    data_to_label_path = "internship_hidden_test.csv"
    save_predictions_path = "internship_hidden_test_predictions.csv"

    # load dataset
    df = pd.read_csv(dataset_path)
    y = df.target.to_numpy()
    X = df.drop(columns=['target']).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

    # find best regression model
    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=600,
        per_run_time_limit=120,
        metric = autosklearn.metrics.root_mean_squared_error,
        n_jobs=-1
    )

    automl.fit(X_train, y_train, X_test=X_test, y_test=y_test, dataset_name="t3")

    train_predictions = automl.predict(X_train)
    print("Train RMSE score:    ", sklearn.metrics.mean_squared_error(y_train, train_predictions, squared = False))
    test_predictions = automl.predict(X_test)
    print("Test RMSE score:     ", sklearn.metrics.mean_squared_error(y_test, test_predictions, squared = False))

    # retrain using all available labeled data points
    automl.refit(X, y)

    # load data, make predictions for provided data, save labels to file
    X_hidden = pd.read_csv(data_to_label_path).to_numpy()
    y_hidden = automl.predict(X_hidden)
    np.savetxt(save_predictions_path, y_hidden, delimiter=",")

    # extract regressor from ensamble models dict
    regressor = 0
    for key, value in automl.show_models().items() :
        regressor = automl.show_models()[key]['sklearn_regressor']

    return regressor


if __name__ == "__main__":
    main()