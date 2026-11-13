
import re
import joblib

# Extract the same features used during training
def extract_features(email):
    email = email.lower()

    url_count = len(re.findall(r'https?://|www\.', email))

    urgent_words = [
        "urgent", "immediately", "blocked", "suspended",
        "verify", "password", "claim", "prize", "bank"
    ]

    keyword_count = sum(email.count(word) for word in urgent_words)

    return f"{email} URLCOUNT{url_count} KEYWORDCOUNT{keyword_count}"


# Load trained model and vectorizer
model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

print("=== Phishing Email Detection ===")

email = input("\nEnter an email message: ")

# Apply feature extraction
processed_email = extract_features(email)

# Convert text into numerical features
email_vector = vectorizer.transform([processed_email])

# Predict
prediction = model.predict(email_vector)[0]

print("\nPrediction:", prediction)

if prediction.lower() == "phishing":
    print("WARNING: This email may be a phishing email!")
else:
    print("This email appears to be safe.")
