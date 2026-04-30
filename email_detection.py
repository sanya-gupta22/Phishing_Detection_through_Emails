import pandas as pd
import numpy as np
import re
import string 
import nltk
nltk.download('wordnet')    # Downloads the WordNet lexical database
nltk.download('stopwords')  # Downloads common stopwords
nltk.download('omw-1.4')    # Downloads Open Multilingual Wordnet (version 1.4) dataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from urllib.parse import urlparse


df = pd.read_csv("phishing_email.csv")
df.head()
df.info()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', 'URL', text )

    # Remove numbers
    text = re.sub(r'\d+', '', text) #\d+ "it will match as many consecutive digits as possible before stopping."

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra spaces
    text = text.strip()

    # Remove stopwords 
    words = text.split()

    # Remove stopwords + lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return " ".join(words)



def extract_features(text):
    if not isinstance(text, str): # Handles non-string inputs (NaN)
        text = ""
    
    words = text.split()    # Splits into words
    text_lower = text.lower()   # Lowercase for consistent matching
    
    # URLs and emails
    urls = re.findall(r'http[s]?://\S+', text, re.I)   # Case-insensitive URL detection
    emails = re.findall(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', text, re.I)  # Case-insensitive email detection
    
    # Unique domains
    domains = set()
    for url in urls:
        try:
            domain = urlparse(url).netloc  
            if domain:
                domains.add(domain) 
        except:
            pass
    
    # Spelling errors
    misspellings = ['recieve', 'seperate', 'definately', 'occured', 'untill', 'wich']
    spelling_errors = sum(text_lower.count(m) for m in misspellings)    
    spelling_errors += len(re.findall(r'(.)\1{2,}', text))  
    
    # Urgent keywords
    urgent = ['urgent', 'immediate', 'verify', 'update', 'confirm', 'account',
              'security', 'alert', 'suspended', 'click', 'link', 'password']
    urgent_count = sum(text_lower.count(k) for k in urgent) 
    return {
        'num_words': len(words),                    # Total word count
        'num_unique_words': len(set(words)),       # Vocabulary richness
        'num_stopwords': sum(1 for w in words if w.lower() in stop_words),  # Stopword count
        'num_links': len(urls),                    # Number of URLs
        'num_unique_domains': len(domains),        # Different domains linked
        'num_email_addresses': len(emails),        # Email addresses mentioned
        'num_spelling_errors': spelling_errors,    # Spelling mistake count
        'num_urgent_keywords': urgent_count        # Urgency word count
    }  


print(df.columns)   

# Clean text
df['cleaned_text'] = df['text_combined'].apply(clean_text)   
# Extract features
df_features = df['text_combined'].apply(lambda x: pd.Series(extract_features(x)))    #

# Combine features with original dataset
df = pd.concat([df, df_features], axis=1)


from sklearn.feature_extraction.text import TfidfVectorizer #Converts text → numerical features
from scipy.sparse import hstack #Means horizontal stacking, Used to combine two feature sets

# TF-IDF
vectorizer = TfidfVectorizer(max_features=3000) #Keeps only top 3000 important words

#Convert Text into Features
X_text = vectorizer.fit_transform(df['cleaned_text'])  

# Combine both features
# Features (X)
X = hstack((X_text, df_features.values))   
 
# Target (y)
y = df['label']

print(X.shape)
print(y.shape) 

#model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = LogisticRegression(max_iter=3000, class_weight={0:1, 1:2})
model.fit(X_train, y_train)

# Predict
#Probability Prediction
y_prob = model.predict_proba(X_test)[:, 1]

#Apply Custom Threshold
threshold = 0.4   # try 0.3–0.5
y_pred = (y_prob >= threshold).astype(int)


#EPredictions
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


#it will create model.pkl, vectorizer.pkl files in the current directory, which can be loaded later for making predictions on new emails without retraining the model.
import joblib

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")