import argparse
import pathlib
from typing import Type, Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

models = [LinearRegression, Ridge, RidgeCV, Lasso]

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model')
parser.add_argument('-t', '--train_dataset')
parser.add_argument('-v', '--valid_dataset')
args = parser.parse_args()
_models = {model.__name__: model for model in models}
model_params = {}
if args.model in _models.keys():
    Model = _models[args.model]
    autolog = mlflow.sklearn.autolog
    log_ctx_manager = mlflow.start_run
elif args.model == 'xgb':
    Model = XGBClassifier
    model_params = {'n_estimators': 3, 'max_depth': 3, 'learning_rate': 0.1}
    autolog = mlflow.xgboost.autolog
    log_ctx_manager = mlflow.start_run
else:
    raise Exception('Unknown model')


def read_data(path: str | pathlib.Path) -> pd.DataFrame:
    df = pd.read_parquet(path, engine='pyarrow')
    df['duration'] = df.apply(lambda x: (x['lpep_dropoff_datetime'] - x['lpep_pickup_datetime']).total_seconds(),
                              axis=1)
    print(df.head(10).to_string(index=False))
    return df


def train(
        model: Type[Model],
        model_params: dict[str, Any],
        train_dataset: pd.DataFrame,
        valid_dataset: pd.DataFrame,
) -> Model:
    df_binary = train_dataset[['duration', 'trip_distance']]
    X = np.array(df_binary['duration']).reshape(-1, 1)
    y = np.array(df_binary['trip_distance']).reshape(-1, 1)

    regr = model(*model_params)
    le = LabelEncoder()
    y = le.fit_transform(y)
    regr.fit(X, y)
    # tune(x=X, y=y)
    df_binary_test = valid_dataset[['duration', 'trip_distance']]
    X_test = np.array(df_binary_test['duration']).reshape(-1, 1)
    y_test = np.array(df_binary_test['trip_distance']).reshape(-1, 1)

    print(regr.score(X_test, y_test))

    return regr


def main():
    autolog()
    with log_ctx_manager():
        trained_model = train(
            model=Model,
            model_params=model_params,
            train_dataset=read_data(args.train_dataset),
            valid_dataset=read_data(args.valid_dataset),
        )

        df_binary_test = read_data(args.valid_dataset)[['duration', 'trip_distance']]
        X_test = np.array(df_binary_test['duration']).reshape(-1, 1)
        y_test = np.array(df_binary_test['trip_distance']).reshape(-1, 1)
        y_pred = trained_model.predict(X_test)

        mape = mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred)
        rmse = mean_squared_error(y_true=y_test, y_pred=y_pred, squared=False)
        print(f'MAPE: {mape}, RMSE: {rmse}')
        mlflow.log_metric('MAPE', mape)
        mlflow.log_metric('RMSE', rmse)
        mlflow.sklearn.log_model(
            sk_model=Model,
            artifact_path="sklearn-model",
            registered_model_name=Model.__name__,
        )


if __name__ == "__main__":
    main()
