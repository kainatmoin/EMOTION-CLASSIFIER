# ================================
# model_wrapper.py
# ================================

import joblib
import numpy as np
import re
from scipy.sparse import hstack

LABELS = ['admiration','anger','disgust','fear','hope',
          'joy','love','pride','sadness']


class MyModel:
    def __init__(self):

        bundle = joblib.load("mindora_ai_team.pkl")  # ✅ FIXED

        self.vectorizer             = bundle["vectorizer"]
        self.classifier             = bundle["classifier"]
        self.tfidf_char             = bundle["tfidf_char"]
        self.tfidf_bigram           = bundle["tfidf_bigram"]
        self.model_lr2              = bundle["model_lr2"]
        self.model_svc              = bundle["model_svc"]
        self.model_sgd              = bundle["model_sgd"]
        self.per_emotion_thresholds = bundle["per_emotion_thresholds"]
        self.weights                = bundle.get("weights", [0.35, 0.30, 0.20, 0.15])

    def predict(self, texts):

        def preprocess(text):
            text = str(text).lower()
            text = text.replace("i'm", "i am")
            text = text.replace("i've", "i have")
            text = text.replace("can't", "cannot")
            text = text.replace("won't", "will not")
            text = text.replace("don't", "do not")
            text = re.sub(r'http\S+|www\S+', '', text)
            text = re.sub(r'@\w+', '', text)
            text = re.sub(r'#(\w+)', r'\1', text)
            text = re.sub(r'[^a-z\s!?]', '', text)
            text = re.sub(r'(.)\1{2,}', r'\1\1', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        texts    = [preprocess(t) for t in texts]
        X_word   = self.vectorizer.transform(texts)
        X_char   = self.tfidf_char.transform(texts)
        X_bigram = self.tfidf_bigram.transform(texts)
        X        = hstack([X_word, X_char, X_bigram])

        p1 = self.classifier.predict_proba(X)
        p2 = self.model_lr2.predict_proba(X)
        p3 = self.model_svc.predict_proba(X)
        p4 = self.model_sgd.predict_proba(X)

        proba = (p1 * self.weights[0] +
                 p2 * self.weights[1] +
                 p3 * self.weights[2] +
                 p4 * self.weights[3])

        preds = np.zeros(proba.shape, dtype=int)
        for i, t in enumerate(self.per_emotion_thresholds):
            preds[:, i] = (proba[:, i] >= t).astype(int)

        preds = np.array(preds)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 9)

        return preds