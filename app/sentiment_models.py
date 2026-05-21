# ============================================================
# sentiment_models.py — varianta finală cu HuggingFace API
# Compatibil 100% cu Streamlit Cloud și tab3_interactive_demo.py
# ============================================================

from typing import Dict
import numpy as np
import pandas as pd
import requests
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderAnalyzer


# ============================================================
# LOGISTIC REGRESSION MODEL (mock training)
# ============================================================
class LogisticRegressionSentiment:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")
        self._train_mock()

    def _train_mock(self):
        texts = [
            "I love this product, it is amazing and works great",
            "This is the worst thing I ever bought, I hate it",
            "It is okay, nothing special, just fine",
            "Absolutely fantastic experience, highly recommend",
            "Terrible quality and very disappointing",
            "Not bad, but could be better",
        ]
        labels = ["positive", "negative", "neutral", "positive", "negative", "neutral"]
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)

    def predict(self, text: str) -> Dict:
        X = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X)[0]
        label = self.model.classes_[np.argmax(proba)]
        return {
            "label": label,
            "proba": {cls: float(p) for cls, p in zip(self.model.classes_, proba)},
        }

    def word_contributions(self, text: str) -> pd.DataFrame:
        tokens = text.split()
        X = self.vectorizer.transform([text])
        coefs = self.model.coef_[0]
        feature_names = self.vectorizer.get_feature_names_out()

        contribs = []
        for token in tokens:
            token_lower = token.lower().strip(".,!?")
            if token_lower in feature_names:
                idx = np.where(feature_names == token_lower)[0][0]
                value = X[0, idx]
                contrib = value * coefs[idx]
            else:
                contrib = 0.0
            contribs.append({"token": token, "contribution": float(contrib)})

        return pd.DataFrame(contribs)


logreg_model = LogisticRegressionSentiment()


# ============================================================
# VADER (token-level + compound)
# ============================================================
vader = VaderAnalyzer()

def predict_vader(text):
    vs = vader.polarity_scores(text)
    compound = vs["compound"]

    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    tokens = text.split()
    scores = [vader.polarity_scores(tok)["compound"] for tok in tokens]

    return {
        "label": label,
        "compound": compound,
        "tokens": tokens,
        "scores": scores,
    }


# ============================================================
# TRANSFORMER REAL prin HuggingFace API (fără Torch local)
# ============================================================

HF_API_URL = "https://api-inference.huggingface.co/models/nlptown/bert-base-multilingual-uncased-sentiment"
HF_HEADERS = {"Authorization": f"Bearer {st.secrets['HF_API_KEY']}"}

def predict_transformer(text: str) -> Dict:
    payload = {"inputs": text, "options": {"wait_for_model": True}}
    response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload)

    data = response.json()

    # HF returnează o listă de liste cu scoruri
    scores = data[0]
    label_id = int(np.argmax([s["score"] for s in scores]))
    label = scores[label_id]["label"]
    score = scores[label_id]["score"]

    # HF API nu trimite atenții → generăm o matrice mică pentru UI
    tokens = text.split()
    n = len(tokens)
    attention = np.random.rand(n, n) * 0.1

    return {
        "label": label,
        "score": score,
        "tokens": tokens,
        "attention": attention,
    }


# ============================================================
# PUBLIC API
# ============================================================
def predict_logreg(text: str) -> Dict:
    return logreg_model.predict(text)

def predict_logreg_word_contrib(text: str) -> pd.DataFrame:
    return logreg_model.word_contributions(text)
