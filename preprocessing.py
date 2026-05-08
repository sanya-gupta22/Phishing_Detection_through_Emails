import re
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from urllib.parse import urlparse

# Initialize once
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = str(text).lower()

    text = re.sub(r'http\S+|www\S+', 'URL', text)
    text = re.sub(r'\d+', '', text)

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    words = text.split()

    words = [
        lemmatizer.lemmatize(w)
        for w in words
        if w not in stop_words
    ]

    return " ".join(words)

# ---------------- EXTRA FEATURES ----------------
def extract_features(text):

    if not isinstance(text, str):
        text = ""

    words = text.split()
    text_lower = text.lower()

    urls = re.findall(r'http[s]?://\S+', text, re.I)

    emails = re.findall(
        r'\b[\w\.-]+@[\w\.-]+\.\w+\b',
        text,
        re.I
    )

    domains = set()

    for url in urls:
        try:
            domain = urlparse(url).netloc

            if domain:
                domains.add(domain)

        except:
            pass

    misspellings = [
        'recieve',
        'seperate',
        'definately',
        'occured',
        'untill',
        'wich'
    ]

    spelling_errors = sum(
        text_lower.count(m)
        for m in misspellings
    )

    spelling_errors += len(
        re.findall(r'(.)\1{2,}', text)
    )

    urgent = [
        'urgent',
        'immediate',
        'verify',
        'update',
        'confirm',
        'account',
        'security',
        'alert',
        'suspended',
        'click',
        'link',
        'password'
    ]

    return np.array([[
        len(words),
        len(set(words)),
        sum(
            1 for w in words
            if w.lower() in stop_words
        ),
        len(urls),
        len(domains),
        len(emails),
        spelling_errors,
        sum(text_lower.count(k) for k in urgent)
    ]])