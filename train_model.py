import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
# Load dataset
df = pd.read_csv("heart.csv")
# Features and Target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]
# Convert categorical columns to numerical
X = pd.get_dummies(X, drop_first=True)
# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)
# Train Random Forest Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)
# Prediction
y_pred = model.predict(X_test)
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
# Save Model
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model Saved Successfully!")
