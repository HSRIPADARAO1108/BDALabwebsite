import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Download required text cleaning resources (Stopwords & Lemmatizer)
print("Downloading text cleaning resources...")
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# 2. Load your local dataset
print("\nLoading 'twitter_tweets.csv'...")
try:
    df = pd.read_csv("twitter_tweets.csv")
    print(f"Successfully loaded dataset! Shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'twitter_tweets.csv' not found in the current directory.")
    print("Please make sure you run the downloader script first in the same folder.")
    exit()

# 3. Text Preprocessing Function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_tweet(text):
    # Ensure text is treated as a string
    text = str(text)
    # Remove Twitter handles (@user)
    text = re.sub(r'@[\w]*', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters, punctuation, and numbers (keeping only alphabets)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase and split into words
    words = text.lower().split()
    # Remove stopwords and lemmatize remaining words (e.g., 'running' -> 'run')
    cleaned_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(cleaned_words)

print("Preprocessing tweets (cleaning noise, links, handles, and stopwords)...")
df['clean_tweet'] = df['tweet'].apply(clean_tweet)

# 4. Define Features (X) and Target Label (y)
X = df['clean_tweet']
y = df['label']  # '0' for normal/non-hate, '1' for hate speech

# Split data: 80% for training and 20% for testing (stratified to balance class ratios)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Feature Extraction (TF-IDF Vectorization)
print("Converting text into TF-IDF numerical vectors...")
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 6. Train the Classifier
# 'class_weight=balanced' is critical here because normal tweets heavily outnumber hate tweets.
print("Training Logistic Regression Classifier...")
model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)

# 7. Evaluate the Trained Model
y_pred = model.predict(X_test_tfidf)

print("\n" + "="*55)
print("              EVALUATION PERFORMANCE              ")
print("="*55)
print(f"Overall Classification Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-Hate (0)', 'Hate Tweet (1)']))
print("="*55)

# 8. Interactive Live Testing
print("\n--- Live Custom Sentence Testing ---")
custom_tests = [
    "I had a wonderful day with my close friends today!",
    "This is a horrible group of people, I absolutely despise them."
]

cleaned_tests = [clean_tweet(t) for t in custom_tests]
test_features = tfidf.transform(cleaned_tests)
predictions = model.predict(test_features)

for original, pred in zip(custom_tests, predictions):
    label_text = "⚠️ Hate Speech (Class 1)" if pred == 1 else "✅ Normal / Non-Hate (Class 0)"
    print(f"👉 Tweet: '{original}'")
    print(f"   Prediction: {label_text}\n")
