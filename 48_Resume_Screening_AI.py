import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("resume_data.csv")

# Features & target
X = df['resume_text']
y = df['category']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluation
y_pred = model.predict(X_test_vec)
print("🎯 Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, "resume_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model Saved!")

# 🔮 Resume Prediction System
def predict_resume():
    print("\n📄 Paste Resume Text:")
    text = input()

    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)

    print(f"\n💼 Predicted Job Role: {prediction[0]}")

# Run loop
while True:
    predict_resume()
