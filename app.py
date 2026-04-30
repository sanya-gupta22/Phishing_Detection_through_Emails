from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import os
import nltk
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from urllib.parse import urlparse
from scipy.sparse import hstack

app = Flask(__name__)
CORS(app)

# ---------------- ENV VARIABLES ----------------
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")
VECTORIZER_PATH = os.environ.get("VECTORIZER_PATH", "vectorizer.pkl")
THRESHOLD = float(os.environ.get("THRESHOLD", 0.4))

# Optional: for Render NLTK fix
NLTK_DATA_PATH = os.environ.get("NLTK_DATA", "/opt/render/nltk_data")

# ---------------- NLTK SETUP ----------------
nltk.data.path.append(NLTK_DATA_PATH)
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

try:
    lemmatizer = WordNetLemmatizer()
except:
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()

# ---------------- LOAD MODEL ----------------
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print(" Model & Vectorizer loaded successfully")
except Exception as e:
    print(" Error loading model:", e)
    model = None
    vectorizer = None

# ----------------  FUNCTIONS ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', 'URL', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

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

    return np.array([[
        len(words),
        len(set(words)),
        len(urls),
        len(domains),
        len(emails),
        sum(text_lower.count(k) for k in urgent)
    ]])

# ---------------- API ----------------
@app.route("/predict", methods=["POST"])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.json.get("text", "")

    cleaned = clean_text(data)
    text_vec = vectorizer.transform([cleaned])
    extra = extract_features(data)

    final_input = hstack((text_vec, extra))

    prob = model.predict_proba(final_input)[0][1]
    pred = int(prob >= THRESHOLD)

    return jsonify({
        "prediction": pred,
        "label": "Phishing" if pred else "Safe",
        "confidence": round(float(prob), 3)
    })

@app.route("/")
def home():
    return "API Running"

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

    
