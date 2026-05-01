# ================================================
# Emotion Classifier
# ================================================

import pandas as pd
import numpy as np
import re
import joblib
from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (f1_score, precision_score,
                             recall_score, hamming_loss,
                             accuracy_score)

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------
df = pd.read_csv('dataset.csv')

all_emotions_raw = df['Emotions (Multi-labeled)'].str.split(',')
all_emotions_raw = all_emotions_raw.apply(
    lambda x: [e.strip().lower() for e in x]
)

all_labels = set()
for emotions in all_emotions_raw:
    all_labels.update(emotions)

emotion_cols = sorted(list(all_labels))
print("Emotions:", emotion_cols)

for emotion in emotion_cols:
    df[emotion] = all_emotions_raw.apply(
        lambda x: 1 if emotion in x else 0
    )

# ------------------------------------------------
# DEFINE X AND y
# ------------------------------------------------
X_raw = df['Tweets (text)'].astype(str)
y = df[emotion_cols].values

# ------------------------------------------------
#  CLEAN TEXT — IMPROVED
# ------------------------------------------------
def preprocess(text):
    text = str(text).lower()
    # Common emotion words expand karo
    text = text.replace("i'm", "i am")
    text = text.replace("i've", "i have")
    text = text.replace("i'd", "i would")
    text = text.replace("i'll", "i will")
    text = text.replace("can't", "cannot")
    text = text.replace("won't", "will not")
    text = text.replace("don't", "do not")
    text = text.replace("doesn't", "does not")
    text = text.replace("didn't", "did not")
    text = text.replace("wasn't", "was not")
    text = text.replace("isn't", "is not")
    text = text.replace("aren't", "are not")
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^a-z\s!?]', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # repeated chars fix
    text = re.sub(r'\s+', ' ', text).strip()
    return text

X_clean = X_raw.apply(preprocess)
print("Sample cleaned:", X_clean[0])

# ------------------------------------------------
#  FEATURES 
# ------------------------------------------------
print("\nExtracting features...")

# Word level — main
tfidf_word = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=40000,
    sublinear_tf=True,
    min_df=1,
    analyzer='word',
    strip_accents='unicode'
)

# Char level — spelling variants
tfidf_char = TfidfVectorizer(
    ngram_range=(2, 5),
    max_features=30000,
    sublinear_tf=True,
    min_df=1,
    analyzer='char_wb'
)

# Bigram focused
tfidf_bigram = TfidfVectorizer(
    ngram_range=(2, 3),
    max_features=20000,
    sublinear_tf=True,
    min_df=1,
    analyzer='word'
)

X_word   = tfidf_word.fit_transform(X_clean)
X_char   = tfidf_char.fit_transform(X_clean)
X_bigram = tfidf_bigram.fit_transform(X_clean)

X_combined = hstack([X_word, X_char, X_bigram])
print("Feature shape:", X_combined.shape)

# ------------------------------------------------
# TRAIN / VAL SPLIT
# ------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_combined, y,
    test_size=0.15,
    random_state=42
)

print("Train size:", X_train.shape)
print("Val size:  ", X_val.shape)

# ------------------------------------------------
#  TRAIN 4 MODELS
# ------------------------------------------------

# Model 1 — LR optimized
print("\nTraining Model 1: Logistic Regression (C=5)...")
lr1 = OneVsRestClassifier(
    LogisticRegression(
        C=5.0,
        max_iter=3000,
        solver='saga',
        random_state=42
    )
)
lr1.fit(X_train, y_train)
print("Done ")

# Model 2 — LR high C
print("Training Model 2: Logistic Regression (C=15)...")
lr2 = OneVsRestClassifier(
    LogisticRegression(
        C=15.0,
        max_iter=3000,
        solver='saga',
        random_state=42
    )
)
lr2.fit(X_train, y_train)
print("Done ")

# Model 3 — LinearSVC
print("Training Model 3: LinearSVC...")
svc = OneVsRestClassifier(
    CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=3000, random_state=42)
    )
)
svc.fit(X_train, y_train)
print("Done ")

# Model 4 — SGD (fast + strong)
print("Training Model 4: SGD Classifier...")
sgd = OneVsRestClassifier(
    CalibratedClassifierCV(
        SGDClassifier(
            loss='modified_huber',
            alpha=0.0001,
            max_iter=1000,
            random_state=42
        )
    )
)
sgd.fit(X_train, y_train)
print("Done ")

# ------------------------------------------------
# FIND BEST ENSEMBLE WEIGHTS
# ------------------------------------------------
print("\nFinding best ensemble weights...")

p1 = lr1.predict_proba(X_val)
p2 = lr2.predict_proba(X_val)
p3 = svc.predict_proba(X_val)
p4 = sgd.predict_proba(X_val)

best_w    = [0.35, 0.30, 0.20, 0.15]
best_f1   = 0

# Try different weight combinations
for w1 in [0.30, 0.35, 0.40]:
    for w2 in [0.25, 0.30, 0.35]:
        for w3 in [0.15, 0.20, 0.25]:
            w4 = round(1.0 - w1 - w2 - w3, 2)
            if w4 < 0.05:
                continue
            proba = (p1*w1 + p2*w2 + p3*w3 + p4*w4)
            preds = (proba >= 0.35).astype(int)
            f1 = f1_score(y_val, preds,
                         average='micro', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_w  = [w1, w2, w3, w4]

print(f"Best weights: {best_w}")
print(f"Best F1 (0.35): {best_f1:.4f}")

# Final ensemble proba
val_proba = (p1 * best_w[0] +
             p2 * best_w[1] +
             p3 * best_w[2] +
             p4 * best_w[3])

# ------------------------------------------------
#  PER EMOTION THRESHOLD TUNE
# ------------------------------------------------
print("\nTuning per-emotion thresholds...")

per_emotion_thresholds = []
for i in range(len(emotion_cols)):
    best_t, best_f = 0.5, 0
    for t in np.arange(0.10, 0.70, 0.05):
        p = (val_proba[:, i] >= t).astype(int)
        f = f1_score(y_val[:, i], p, zero_division=0)
        if f > best_f:
            best_f = f
            best_t = t
    per_emotion_thresholds.append(best_t)

# Final predictions
final_preds = np.zeros_like(val_proba, dtype=int)
for i, t in enumerate(per_emotion_thresholds):
    final_preds[:, i] = (val_proba[:, i] >= t).astype(int)

# ------------------------------------------------
#  COMPLETE EVALUATION
# ------------------------------------------------
print("\n" + "="*57)
print("         COMPLETE EVALUATION REPORT")
print("="*57)

print("\n--- OVERALL METRICS ---")
print(f"F1  Micro         : {f1_score(y_val, final_preds, average='micro', zero_division=0):.4f}")
print(f"F1  Macro         : {f1_score(y_val, final_preds, average='macro', zero_division=0):.4f}")
print(f"Precision (micro) : {precision_score(y_val, final_preds, average='micro', zero_division=0):.4f}")
print(f"Recall    (micro) : {recall_score(y_val, final_preds, average='micro', zero_division=0):.4f}")
print(f"Hamming Loss      : {hamming_loss(y_val, final_preds):.4f}")
exact = accuracy_score(y_val, final_preds)
print(f"Exact Match       : {exact:.4f}  ({round(exact*100,1)}% tweets 100% correct)")

print("\n--- PER EMOTION DETAIL ---")
print(f"{'Emotion':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
print("-" * 57)
for i, col in enumerate(emotion_cols):
    p   = precision_score(y_val[:, i], final_preds[:, i], zero_division=0)
    r   = recall_score(y_val[:, i],    final_preds[:, i], zero_division=0)
    f   = f1_score(y_val[:, i],        final_preds[:, i], zero_division=0)
    sup = int(y_val[:, i].sum())
    print(f"{col:<15} {p:>10.3f} {r:>10.3f} {f:>10.3f} {sup:>10}")
print("-" * 57)

print("\n--- PER EMOTION THRESHOLDS ---")
for col, t in zip(emotion_cols, per_emotion_thresholds):
    print(f"  {col:<15}: {t:.2f}")

print("="*57)

# ------------------------------------------------
#  SAVE MODEL
# ------------------------------------------------
model_dict = {
    'vectorizer'            : tfidf_word,
    'classifier'            : lr1,
    'tfidf_char'            : tfidf_char,
    'tfidf_bigram'          : tfidf_bigram,
    'model_lr2'             : lr2,
    'model_svc'             : svc,
    'model_sgd'             : sgd,
    'emotion_cols'          : emotion_cols,
    'per_emotion_thresholds': per_emotion_thresholds,
    'weights'               : best_w,
}

joblib.dump(model_dict, 'mindora_ai_team.pkl')