import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
data = pd.read_csv("dataset/emails.csv")

X = data["email_text"].astype(str)
y = data["label"].astype(str).str.strip().str.lower()

# Extract URL and keyword features
def extract_features(email):
    email = email.lower()

    url_count = len(re.findall(r'https?://|www\.', email))
    urgent_words = ["urgent", "immediately", "blocked", "suspended", "verify", "password", "claim", "prize"]
    keyword_count = sum(email.count(word) for word in urgent_words)

    return f"{email} URLCOUNT{url_count} KEYWORDCOUNT{keyword_count}"

X = X.apply(extract_features)

# Convert text into numerical features
vectorizer = TfidfVectorizer(stop_words="english")
X_vectorized = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.25, random_state=42, stratify=y
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "phishing_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model trained and saved successfully!")