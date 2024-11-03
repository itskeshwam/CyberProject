# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
data = pd.read_csv("MalwareData.csv", sep="|")  # Use your full dataset file path here

# Separate features and target
X = data.drop(columns=["Name", "md5", "legitimate"])  # Drop unnecessary columns
y = data["legitimate"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model accuracy
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Save the model
joblib.dump(model, "malware_model.pkl")
print("Model saved as 'malware_model.pkl'")
