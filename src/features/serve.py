import os

import mlflow
import pandas as pd
from flask import Flask, request, jsonify

RUN_ID = os.getenv("RUN_ID")
logged_model = f"runs:/{RUN_ID}/model"
model = mlflow.pyfunc.load_model(logged_model)


def prepare_features(ride):
    return ride


def predict(features):
    prediction = model.predict(pd.DataFrame([features['trip_distance']]))
    return prediction


app = Flask("duration-prediction")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    ride = request.get_json()
    features = prepare_features(ride)
    pred = predict(features)[0][0]
    result = {
        "duration": pred,
        "model_version": RUN_ID
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
