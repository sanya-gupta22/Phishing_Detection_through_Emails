from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import string
import numpy as np
import os
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from urllib.parse import urlparse
from scipy.sparse import hstack

# ---------------- INIT ----------------
app = Flask(__name__)
CORS(app)

# ---------------- ENV ----------------
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")
VECTORIZER_PATH = os.environ.get("VECTORIZER_PATH", "vectorizer.pkl")
THRESHOLD = float(os.environ.get("THRESHOLD", 0.4))
NLTK_DATA = os.environ.get("NLTK_DATA", "/opt/render/nltk_data")

# ---------------- NLTK SETUP ----------------
# Add NLTK data path before downloading
nltk.data.path.append(NLTK_DATA)

# Create directory if it doesn't exist
os.makedirs(NLTK_DATA, exist_ok=True)

# Download required NLTK data
def download_nltk_resources():
    resources = ['stopwords', 'wordnet', 'omw-1.4']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
        except LookupError:
            print(f"Downloading {resource}...")
            nltk.download(resource, download_dir=NLTK_DATA, quiet=False)

# Download all resources at startup
download_nltk_resources()

# Now import and initialize
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

print("NLTK resources loaded successfully")

# ---------------- LOAD MODEL ----------------
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
print("Model loaded")

# ---------------- CLEAN ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', 'URL', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# ---------------- FEATURES (MATCH TRAINING) ----------------
def extract_features(text):
    if not isinstance(text, str):
        text = ""

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

    misspellings = ['recieve','seperate','definately','occured','untill','wich']
    spelling_errors = sum(text_lower.count(m) for m in misspellings)
    spelling_errors += len(re.findall(r'(.)\1{2,}', text))

    urgent = ['urgent','immediate','verify','update','confirm','account',
              'security','alert','suspended','click','link','password']

    return np.array([[
        len(words),
        len(set(words)),
        sum(1 for w in words if w.lower() in stop_words),
        len(urls),
        len(domains),
        len(emails),
        spelling_errors,
        sum(text_lower.count(k) for k in urgent)
    ]])

# ---------------- API ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        cleaned = clean_text(text)
        text_vec = vectorizer.transform([cleaned])
        extra = extract_features(text)

        final_input = hstack((text_vec, extra))

        prob = model.predict_proba(final_input)[0][1]
        pred = int(prob >= THRESHOLD)

        return jsonify({
            "prediction": pred,
            "label": "Phishing" if pred else "Safe",
            "confidence": round(float(prob), 3)
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "API Running"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)