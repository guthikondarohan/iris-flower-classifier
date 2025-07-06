# train_model.py

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_scaled, y)

# Create 'model' folder if not exists
os.makedirs("model", exist_ok=True)

# Save the model and scaler together
with open("model/iris_model.pkl", "wb") as f:
    pickle.dump((model, scaler), f)

print("✅ Model and scaler saved to model/iris_model.pkl")
