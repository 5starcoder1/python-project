import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("house_data.csv")

# Encode categorical feature (location)
le = LabelEncoder()
df['location'] = le.fit_transform(df['location'])

# Features & target
X = df[['area', 'bedrooms', 'bathrooms', 'location']]
y = df['price']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100)

# Train models
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Predictions
lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

# Evaluation
print("📊 Linear Regression R2:", r2_score(y_test, lr_pred))
print("🌳 Random Forest R2:", r2_score(y_test, rf_pred))

# Choose best model
model = rf  # (usually better)

# Save model & scaler
joblib.dump(model, "house_price_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "encoder.pkl")

print("✅ Model Saved!")

# 🔮 Prediction System
def predict_price():
    print("\nEnter House Details:")

    area = float(input("Area (sq ft): "))
    bedrooms = int(input("Bedrooms: "))
    bathrooms = int(input("Bathrooms: "))
    location = input("Location (urban/suburban/rural): ")

    # Encode location
    location_encoded = le.transform([location])[0]

    input_data = np.array([[area, bedrooms, bathrooms, location_encoded]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    print(f"\n🏠 Estimated Price: ₹ {prediction[0]:,.2f}")

# Run loop
while True:
    predict_price()
