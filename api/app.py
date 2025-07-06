from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ Add this
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # ✅ Allow CORS from any origin

# Load model and scaler
with open("../model/iris_model.pkl", "rb") as f:
    model, scaler = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array(data["features"]).reshape(1, -1)
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)
    species = ["setosa", "versicolor", "virginica"]
    return jsonify({"prediction": species[prediction[0]]})

if __name__ == "__main__":
    app.run(debug=True)
