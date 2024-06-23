import os

import joblib
from preprocess import load_and_preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load and preprocess data
X_train, y_train, scaler = load_and_preprocess("data/train.csv")
X_val, y_val, _ = load_and_preprocess("data/val.csv")

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation accuracy: {accuracy:.2f}")

# Save model and scaler
joblib.dump(model, "models/model.joblib")
joblib.dump(scaler, "models/scaler.joblib")
