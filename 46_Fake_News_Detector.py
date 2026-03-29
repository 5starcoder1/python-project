import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("news.csv")

# Convert labels
df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluation
y_pred = model.predict(X_test_vec)
print("🎯 Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model Saved!")

# 🔮 Prediction System
def predict_news():
    print("\n📰 Enter News Text:")
    news = input()

    news_vec = vectorizer.transform([news])
    prediction = model.predict(news_vec)

    if prediction[0] == 0:
        print("🚨 FAKE NEWS")
    else:
        print("✅ REAL NEWS")

# Run loop
while True:
    predict_news()
