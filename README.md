# Phishing Email Detection Model

## Project Overview
This project uses Machine Learning and Python to identify phishing emails and distinguish them from safe emails.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Regular Expressions (Regex)

## Key Features
- Train a machine learning model using email data.
- Extract URL and suspicious keyword features.
- Classify emails as Phishing or Safe.
- Display model accuracy.
- Generate a confusion matrix and classification report.
- Predict whether a new email is suspicious.

## Project Files
- train_model.py - Trains and evaluates the machine learning model.
- predict_email.py - Predicts whether an email is phishing or safe.
- emails.csv - Email dataset.
- phishing_model.pkl - Saved trained model.
- tfidf_vectorizer.pkl - Saved text vectorizer.

## How to Run

1. Install the required libraries:

   pip install pandas scikit-learn joblib

2. Train the model:

   python train_model.py

3. Run email prediction:

   python predict_email.py

## Model Performance
The current model achieved 90% accuracy on the test dataset.

Note: This result is based on the current small dataset and may vary with larger and more diverse datasets.

## Author
Mithun Muniraju
