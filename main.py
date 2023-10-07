from pandas.core.arrays.arrow import array
from typing import List
import pandas as pd
from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("brainTumorDataset.pkl", "rb"))


@app.route("/predict", methods=["POST"])
def predict():
    json_ = request.json
    query_df = pd.DataFrame(json_)
    prediction = model.predict(query_df)
    return jsonify({"Prediction": list(prediction)})


if __name__ == "__main__":
    app.run(debug=True)
