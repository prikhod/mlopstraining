import numpy as np
import optuna as optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


def tune(
        x: pd.DataFrame,
        y: pd.DataFrame,
        n_trials: int = 10,
        n_jobs: int = 1,
        tracking_uri: str | None = None,
):
    kf = KFold(n_splits=5, shuffle=True, random_state=30)

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=100,
            max_depth=trial.suggest_int("max_depth", low=2, high=20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", low=1, high=100),
            random_state=30,
        )
        metrics = []
        for train_idx, test_idx in kf.split(x, y):
            X_train, y_train = x[train_idx], y[train_idx]
            X_val, y_val = x[test_idx], y[test_idx]
            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            metrics.append(mean_squared_error(y_val, y_pred, squared=False))
        return np.round(np.mean(metrics), decimals=4)

    mlflow_callback = optuna.integration.MLflowCallback(
        tracking_uri=tracking_uri,
        metric_name="rmse",
        mlflow_kwargs={"nested": True},
    )

    study = optuna.create_study()
    study.optimize(
        func=objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        callbacks=[mlflow_callback],
    )
