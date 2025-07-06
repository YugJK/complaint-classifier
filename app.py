import streamlit as st
import numpy as np
import pickle
from keras.models import load_model
import gensim.downloader as api
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Download NLTK resources only once
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# ⏱️ Load Keras Model (cached in memory)
@st.cache_resource
def load_ann_model():
    return load_model('complaint_classifier_ann.h5')

# ⏱️ Load Label Encoder
@st.cache_resource
def load_encoder():
    with open('label_encoder.pkl', 'rb') as f:
        return pickle.load(f)

# ⚡ Load Word2Vec model (lightweight option)
@st.cache_resource
def load_word2vec():
    return api.load("word2vec-google-news-300")  # ⏩ much faster (100D)

# --- Load models ---
model = load_ann_model()
le = load_encoder()
w2v_model = load_word2vec()

# --- Text Preprocessing ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def get_avg_word2vec(tokens, model, k=100):  # 100D for GloVe
    valid_vectors = [model[word] for word in tokens if word in model]
    if not valid_vectors:
        return np.zeros(k)
    return np.mean(valid_vectors, axis=0)

# --- Streamlit App UI ---
st.title("Bank Complaint Classifier (ANN + NLP)")
st.write("Paste a complaint below to classify it into a banking issue category or Product Category.")

user_input = st.text_area("Enter a customer complaint:")

if st.button("🔍 Classify Complaint"):
    if user_input.strip() == "":
        st.warning("Please enter a complaint text.")
    else:
        tokens = clean_text(user_input)
        vec = get_avg_word2vec(tokens, w2v_model).reshape(1, -1)
        pred_probs = model.predict(vec)
        pred_class = le.inverse_transform([np.argmax(pred_probs)])[0]

        st.success(f"**Predicted Complaint Category:** {pred_class}")
