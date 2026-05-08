import nltk
import os

NLTK_DIR = "/opt/render/nltk_data"
os.makedirs(NLTK_DIR, exist_ok=True)

nltk.download("wordnet", download_dir=NLTK_DIR)
nltk.download("omw-1.4", download_dir=NLTK_DIR)
nltk.download("stopwords", download_dir=NLTK_DIR)

print("NLTK data downloaded")