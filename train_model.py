import pandas as pd
import numpy as np
import re
import string
import joblib

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from urllib.parse import urlparse

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("phishing_email.csv")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', 'URL', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# ---------------- FEATURES ----------------
def extract_features(text):
    text = str(text)
    words = text.split()
    text_lower = text.lower()

    urls = re.findall(r'http[s]?://\S+', text, re.I)
    emails = re.findall(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', text, re.I)

    domains = set()
    for url in urls:
        try:
            domain = urlparse(url).netloc
            if domain:
                domains.add(domain)
        except:
            pass

    urgent = ['urgent','immediate','verify','update','confirm','account',
              'security','alert','suspended','click','link','password']

    return [
        len(words),
        len(set(words)),
        len(urls),
        len(domains),
        len(emails),
        sum(text_lower.count(k) for k in urgent)
    ]

# Apply preprocessing
df['cleaned'] = df['text_combined'].apply(clean_text)
extra_features = np.array(df['text_combined'].apply(extract_features).tolist())

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