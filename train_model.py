# train_model.py

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Load sample dataset
X, y = load_iris(return_X_y=True)

# Standard preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier()
model.fit(X_scaled, y)

# Save model and scaler together (standard objects only!)
with open("model.pkl", "wb") as f:
    pickle.dump((model, scaler), f)
