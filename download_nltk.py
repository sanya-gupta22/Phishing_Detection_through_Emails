import nltk
import os

download_path = "/opt/render/nltk_data"

print("Downloading NLTK data to:", download_path)

nltk.download('stopwords', download_dir=download_path)
nltk.download('wordnet', download_dir=download_path)
nltk.download('omw-1.4', download_dir=download_path)

print("Download complete")