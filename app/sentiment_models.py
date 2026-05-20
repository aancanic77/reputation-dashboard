# ============================================================
# sentiment_models.py — versiunea finală completă
# Compatibil 100% cu tab3_interactive_demo.py
# ============================================================

from typing import Dict
import numpy as np
import pandas as pd
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification


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
# TRANSFORMER WITH ATTENTION (nlptown)
# ============================================================
# ============================================================
# TRANSFORMER WITH ATTENTION (nlptown) — LAZY LOADING
# ============================================================

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

_tokenizer = None
_model = None

def get_transformer():
    global _tokenizer, _model

    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            output_attentions=True
        )

    return _tokenizer, _model


def predict_transformer(text):
    tokenizer, model = get_transformer()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    attentions = outputs.attentions

    probs = torch.softmax(logits, dim=-1)[0]
    score = float(torch.max(probs))
    label_id = int(torch.argmax(probs))

    if label_id <= 1:
        label = "negative"
    elif label_id == 2:
        label = "neutral"
    else:
        label = "positive"

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    last_layer = attentions[-1]
    mean_attention = last_layer.mean(dim=1)
    attention_matrix = mean_attention[0].cpu().numpy()

    return {
        "label": label,
        "score": score,
        "tokens": tokens,
        "attention": attention_matrix,
    }



# ============================================================
# PUBLIC API
# ============================================================
def predict_logreg(text: str) -> Dict:
    return logreg_model.predict(text)

def predict_logreg_word_contrib(text: str) -> pd.DataFrame:
    return logreg_model.word_contributions(text)
