import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
data = pd.read_csv('MalwareData.csv')

# Drop irrelevant columns for training
X = data.drop(columns=['Name', 'md5', 'legitimate'])  # Dropping non-feature columns
y = data['legitimate']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model and feature columns
joblib.dump(model, 'malware_model.pkl')
selected_features = list(X.columns)
joblib.dump(selected_features, 'selected_features.pkl')
