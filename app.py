import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ----------------------
# 1. Load dan Preprocessing Dataset
# ----------------------
df = pd.read_csv("IMDB_sample.csv.xls", encoding='utf-8', engine='python')
stopwords = set(ENGLISH_STOP_WORDS)
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stopwords]
    return ' '.join(words)

df['clean_review'] = df['review'].astype(str).apply(preprocess)

# ----------------------
# 2. TF-IDF dan Training Model
# ----------------------
X = df['clean_review']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# ----------------------
# 3. Fungsi Prediksi
# ----------------------
def predict_sentiment(text):
    text = preprocess(text)
    vectorized = tfidf.transform([text])
    pred = model.predict(vectorized)[0]
    return "Positif" if pred == 1 else "Negatif"

# ----------------------
# 4. Streamlit UI
# ----------------------
st.set_page_config(page_title="Sentimen Review Film - IMDB", layout="centered")
st.title("🎬 Analisis Sentimen Review Film IMDB")
st.markdown("Masukkan review film dan sistem akan memprediksi sentimennya (Positif atau Negatif).")

user_input = st.text_area("✍️ Masukkan review film di sini:", height=150)

if st.button("🔍 Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("Silakan masukkan teks review terlebih dahulu.")
    else:
        hasil = predict_sentiment(user_input)
        st.success(f"**Prediksi Sentimen: {hasil}**")
