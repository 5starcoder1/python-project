import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("student_data.csv")

# Features & target
X = df[['study_hours', 'sleep_hours', 'attendance', 'previous_marks']]
y = df['marks']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("🎯 Model Accuracy (R2 Score):", r2_score(y_test, y_pred))

# Save model
joblib.dump(model, "marks_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model Saved!")

# 🔮 Prediction System
def predict_marks():
    print("\nEnter Student Details:")

    study = float(input("Study Hours: "))
    sleep = float(input("Sleep Hours: "))
    attendance = float(input("Attendance (%): "))
    prev = float(input("Previous Marks: "))

    input_data = np.array([[study, sleep, attendance, prev]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    print(f"\n📊 Predicted Marks: {prediction[0]:.2f}")

# Run prediction loop
while True:
    predict_marks()
