from utils import db_connect
engine = db_connect()

# your code here

# Your code here


import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Step 1: Loading the dataset
url = "https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv"
data = pd.read_csv(url)

# Step 2: Preprocess the links
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_url(url):
    # Segment the URLs into parts according to their punctuation marks
    url_parts = re.split(r'\W+', url)
    # Remove stopwords and lemmatize
    url_parts = [lemmatizer.lemmatize(part.lower()) for part in url_parts if part.lower() not in stop_words]
    return ' '.join(url_parts)

data['processed_url'] = data['url'].apply(preprocess_url)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data['processed_url'], data['is_spam'], test_size=0.2, random_state=42)

# Vectorize the URLs using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 3: Build an SVM with default parameters
svm_model = SVC()
svm_model.fit(X_train_tfidf, y_train)
y_pred = svm_model.predict(X_test_tfidf)

# Analyze the results
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Step 4: Optimize the previous model using Grid Search
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train_tfidf, y_train)
best_model = grid_search.best_estimator_
y_pred_optimized = best_model.predict(X_test_tfidf)

# Analyze the optimized results
print("Optimized Classification Report:\n", classification_report(y_test, y_pred_optimized))
print("Optimized Accuracy Score:", accuracy_score(y_test, y_pred_optimized))

# Step 5: Save the model
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
