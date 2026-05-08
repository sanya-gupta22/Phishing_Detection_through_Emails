import pandas as pd
import numpy as np
import os 
import joblib

from preprocessing import clean_text, extract_features
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(BASE_DIR, "phishing_email.csv")

df = pd.read_csv(csv_path)

# Apply preprocessing
df['cleaned'] = df['text_combined'].apply(clean_text)
extra_features = np.vstack(
    df['text_combined'].apply(extract_features).values
)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)
X_text = vectorizer.fit_transform(df['cleaned'])

# Combine
X = hstack((X_text, extra_features))
y = df['label']

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=3000, class_weight={0:1, 1:2})
model.fit(X_train, y_train)

# SAVE EVERYTHING
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model saved successfully")