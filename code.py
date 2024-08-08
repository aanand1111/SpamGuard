import tkinter as tk
from tkinter import messagebox
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess email text
def preprocess(email):
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    words = word_tokenize(email.lower())
    filtered = [ps.stem(w) for w in words if w.isalnum() and w not in stop_words]
    return ' '.join(filtered)

# Load dataset from CSV file
def load_dataset(mail_data_csv):
    dataset = pd.read_csv(mai_data.csv)
    return dataset

# Train Naive Bayes classifier
def train_classifier(dataset):
    vectorizer = CountVectorizer()
    X_train_counts = vectorizer.fit_transform(dataset['email'])
    clf = MultinomialNB()
    clf.fit(X_train_counts, dataset['label'])
    return clf, vectorizer

# Function to predict email type
def predict_email_type():
    input_email = text_entry.get("1.0", "end-1c")
    preprocessed_input = preprocess(input_email)
    input_counts = vectorizer.transform([preprocessed_input])
    predicted_label = classifier.predict(input_counts)
    messagebox.showinfo("Prediction", f"The email is predicted to be: {predicted_label[0]}")

# Create GUI window
window = tk.Tk()
window.title("Email Spam Detection")
