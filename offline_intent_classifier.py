#!/usr/bin/env python3
"""
Offline Intent Classifier
Lightweight, fully-offline intent classifier using only numpy/pandas.
No OpenAI/Gemini API calls. Runs on CPU in <200 ms per message.
Model size target: <50 MB (this implementation is <1 MB).

Architecture:
  - TF-IDF-like sparse feature extraction (manual, no sklearn)
  - Softmax logistic regression trained with mini-batch SGD
  - 5 classes: reminder / emotional-support / action-item / small-talk / unknown
"""

import json
import math
import pickle
import time
from collections import Counter
from typing import List, Dict, Tuple, Any
from pathlib import Path

import numpy as np


class TfidfVectorizerLite:
    """Minimal TF-IDF vectorizer — no external dependencies."""

    def __init__(self, max_features: int = 3000, ngram_range: Tuple[int, int] = (1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        self._stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "under", "and", "but", "or", "yet", "so", "if",
            "because", "although", "though", "while", "where", "when",
            "that", "which", "who", "whom", "whose", "what", "this",
            "these", "those", "i", "you", "he", "she", "it", "we", "they",
            "me", "him", "her", "us", "them", "my", "your", "his",
            "its", "our", "their", "am", "just", "really", "very",
        }

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens = []
        current = []
        for ch in text:
            if ch.isalnum():
                current.append(ch)
            else:
                if current:
                    t = "".join(current)
                    if len(t) > 1 and t not in self._stopwords:
                        tokens.append(t)
                    current = []
        if current:
            t = "".join(current)
            if len(t) > 1 and t not in self._stopwords:
                tokens.append(t)
        return tokens

    def _ngrams(self, tokens: List[str]) -> List[str]:
        out = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                out.append(" ".join(tokens[i:i + n]))
        return out

    def fit(self, docs: List[str]) -> "TfidfVectorizerLite":
        doc_freq = Counter()
        tokenized = []
        for d in docs:
            toks = self._tokenize(d)
            grams = self._ngrams(toks)
            tokenized.append(grams)
            doc_freq.update(set(grams))

        # Keep top-k by document frequency
        top = doc_freq.most_common(self.max_features)
        self.vocab = {term: idx for idx, (term, _) in enumerate(top)}
        n_docs = len(docs)
        idf_vals = []
        for term, df in top:
            idf = math.log((n_docs + 1) / (df + 1)) + 1.0
            idf_vals.append(idf)
        self.idf = np.array(idf_vals, dtype=np.float32)
        self._tokenized_cache = tokenized
        return self

    def _tf(self, grams: List[str]) -> np.ndarray:
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        for g in grams:
            if g in self.vocab:
                vec[self.vocab[g]] += 1
        if vec.sum() > 0:
            vec = vec / vec.sum()
        return vec

    def transform(self, docs: List[str]) -> np.ndarray:
        if not self.vocab:
            raise RuntimeError("Vectorizer not fitted yet.")
        out = np.zeros((len(docs), len(self.vocab)), dtype=np.float32)
        for i, d in enumerate(docs):
            grams = self._ngrams(self._tokenize(d))
            tf = self._tf(grams)
            out[i] = tf * self.idf
        # L2 normalize rows
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    def fit_transform(self, docs: List[str]) -> np.ndarray:
        self.fit(docs)
        return self.transform(docs)


class SoftmaxClassifier:
    """Multinomial logistic regression trained with SGD."""

    def __init__(self, n_features: int, n_classes: int, lr: float = 0.5, reg: float = 0.01):
        self.W = np.random.randn(n_features, n_classes).astype(np.float32) * 0.01
        self.b = np.zeros(n_classes, dtype=np.float32)
        self.lr = lr
        self.reg = reg
        self.n_classes = n_classes

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        z_stable = z - np.max(z, axis=1, keepdims=True)
        e = np.exp(z_stable)
        return e / np.sum(e, axis=1, keepdims=True)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X.dot(self.W) + self.b
        return self._softmax(z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 30, batch_size: int = 32):
        n = X.shape[0]
        for epoch in range(epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, batch_size):
                batch_idx = indices[start:start + batch_size]
                Xb = X[batch_idx]
                yb = y[batch_idx]
                probs = self.predict_proba(Xb)
                y_onehot = np.zeros((Xb.shape[0], self.n_classes), dtype=np.float32)
                y_onehot[np.arange(len(yb)), yb] = 1.0
                grad = (probs - y_onehot) / len(yb)
                dW = Xb.T.dot(grad) + self.reg * self.W
                db = np.sum(grad, axis=0)
                self.W -= self.lr * dW
                self.b -= self.lr * db


class OfflineIntentClassifier:
    """End-to-end offline intent classifier."""

    LABELS = ["reminder", "emotional-support", "action-item", "small-talk", "unknown"]

    def __init__(self, model_path: str = "intent_model.pkl"):
        self.model_path = Path(model_path)
        self.vectorizer: TfidfVectorizerLite = TfidfVectorizerLite(max_features=3000)
        self.classifier: SoftmaxClassifier = SoftmaxClassifier(3000, 5)
        self.is_trained = False

    # ------------------------------------------------------------------
    # Training data (synthetic but realistic)
    # ------------------------------------------------------------------
    _TRAIN_DATA: List[Tuple[str, str]] = [
        # reminder
        ("Remind me to call mom at 5 pm", "reminder"),
        ("Set a reminder for my dentist appointment tomorrow", "reminder"),
        ("Don't let me forget to take my pills", "reminder"),
        ("Remind me about the meeting in 10 minutes", "reminder"),
        ("Add a reminder to buy milk", "reminder"),
        ("Ping me at 3 to water the plants", "reminder"),
        ("Can you remind me to submit the report Friday?", "reminder"),
        ("Alert me when it's time to leave", "reminder"),
        ("Remind me to pick up the kids", "reminder"),
        ("Set an alarm for 6 am", "reminder"),
        ("Please remind me about mom's birthday next week", "reminder"),
        ("Don't forget reminder: call the plumber", "reminder"),
        ("Nudge me to finish the slides by tonight", "reminder"),
        ("Remind me to charge my laptop before bed", "reminder"),
        ("Tell me again at noon to eat lunch", "reminder"),

        # emotional-support
        ("I'm feeling really down today", "emotional-support"),
        ("I just had a fight with my best friend and I'm devastated", "emotional-support"),
        ("Nothing seems to go right lately", "emotional-support"),
        ("Can you cheer me up?", "emotional-support"),
        ("I'm so anxious about the interview", "emotional-support"),
        ("I feel lonely", "emotional-support"),
        ("My cat passed away and I can't stop crying", "emotional-support"),
        ("I need someone to talk to", "emotional-support"),
        ("I'm overwhelmed by everything", "emotional-support"),
        ("Do you think things will get better?", "emotional-support"),
        ("I failed my exam and I feel worthless", "emotional-support"),
        ("Just need a virtual hug right now", "emotional-support"),
        ("I'm scared about the surgery", "emotional-support"),
        ("Why does everything hurt so much?", "emotional-support"),
        ("Tell me it's going to be okay", "emotional-support"),

        # action-item
        ("Schedule a meeting with the design team", "action-item"),
        ("Create a task to refactor the auth module", "action-item"),
        ("Send the invoice to accounting", "action-item"),
        ("Add this to my to-do list", "action-item"),
        ("Mark the Jira ticket as done", "action-item"),
        ("Book a flight to New York", "action-item"),
        ("Write an email to the client", "action-item"),
        ("Order more coffee beans", "action-item"),
        ("File the expense report", "action-item"),
        ("Update the roadmap document", "action-item"),
        ("Assign the bug to Sarah", "action-item"),
        ("Push the latest changes to staging", "action-item"),
        ("Call the landlord about the leak", "action-item"),
        ("Reserve a table for two at 8", "action-item"),
        ("Submit the pull request", "action-item"),

        # small-talk
        ("Hey, what's up?", "small-talk"),
        ("Nice weather today, isn't it?", "small-talk"),
        ("Did you see the game last night?", "small-talk"),
        ("How's your day going?", "small-talk"),
        ("Tell me a joke", "small-talk"),
        ("What do you think about pineapple on pizza?", "small-talk"),
        ("Any plans for the weekend?", "small-talk"),
        ("I just got a new puppy!", "small-talk"),
        ("Have you watched any good movies lately?", "small-talk"),
        ("Good morning!", "small-talk"),
        ("What's your favorite color?", "small-talk"),
        ("Random thought: space is huge", "small-talk"),
        ("I love rainy days", "small-talk"),
        ("Coffee or tea?", "small-talk"),
        ("Just saying hi", "small-talk"),

        # unknown
        ("The square root of 144 is 12", "unknown"),
        ("Translate 'hello' to Japanese", "unknown"),
        ("What is the capital of France?", "unknown"),
        ("Solve x + 5 = 12", "unknown"),
        ("How many ounces in a cup?", "unknown"),
        ("Define photosynthesis", "unknown"),
        ("Who wrote Hamlet?", "unknown"),
        ("What is the speed of light?", "unknown"),
        ("Convert 100 USD to EUR", "unknown"),
        ("What does DNA stand for?", "unknown"),
        ("How tall is Mount Everest?", "unknown"),
        ("Explain quantum computing", "unknown"),
        ("When was the moon landing?", "unknown"),
        ("What is blockchain?", "unknown"),
        ("List the planets in order", "unknown"),
    ]

    def train(self):
        texts = [t for t, _ in self._TRAIN_DATA]
        labels = [self.LABELS.index(l) for _, l in self._TRAIN_DATA]
        X = self.vectorizer.fit_transform(texts)
        y = np.array(labels, dtype=np.int32)

        n_features = X.shape[1]
        self.classifier = SoftmaxClassifier(n_features, len(self.LABELS), lr=0.3, reg=0.01)
        self.classifier.fit(X, y, epochs=60, batch_size=16)
        self.is_trained = True

        # Evaluate on training data
        preds = self.classifier.predict(X)
        acc = np.mean(preds == y)
        print(f"[train] accuracy on synthetic data: {acc:.2%}")

    def save(self):
        with open(self.model_path, "wb") as f:
            pickle.dump({
                "vocab": self.vectorizer.vocab,
                "idf": self.vectorizer.idf,
                "W": self.classifier.W,
                "b": self.classifier.b,
            }, f)
        size_mb = self.model_path.stat().st_size / (1024 * 1024)
        print(f"[save] model written to {self.model_path} ({size_mb:.2f} MB)")

    def load(self):
        if not self.model_path.exists():
            return False
        with open(self.model_path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer.vocab = data["vocab"]
        self.vectorizer.idf = data["idf"]
        n_features = len(self.vectorizer.vocab)
        self.classifier = SoftmaxClassifier(n_features, len(self.LABELS))
        self.classifier.W = data["W"]
        self.classifier.b = data["b"]
        self.is_trained = True
        return True

    def classify(self, text: str) -> Dict[str, Any]:
        if not self.is_trained:
            raise RuntimeError("Classifier not trained or loaded.")
        X = self.vectorizer.transform([text])
        probs = self.classifier.predict_proba(X)[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        return {
            "intent": self.LABELS[pred_idx],
            "confidence": round(confidence, 4),
            "scores": {l: round(float(p), 4) for l, p in zip(self.LABELS, probs)}
        }

    def benchmark(self, texts: List[str], runs: int = 3) -> Dict[str, Any]:
        """Measure average per-message inference time on CPU."""
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            for t in texts:
                self.classify(t)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        avg_total = sum(times) / len(times)
        avg_per_msg = avg_total / len(texts) * 1000  # ms
        return {
            "messages": len(texts),
            "runs": runs,
            "avg_total_s": round(avg_total, 4),
            "avg_per_msg_ms": round(avg_per_msg, 4),
            "meets_sla": avg_per_msg < 200.0,
        }


def main():
    clf = OfflineIntentClassifier(model_path="/workspace/app-ck3ijh1si681/tasks/intent_model.pkl")

    # Train & save
    print("=" * 60)
    print("OFFLINE INTENT CLASSIFIER")
    print("=" * 60)
    clf.train()
    clf.save()

    # Demo predictions
    test_messages = [
        "Remind me to call the dentist tomorrow morning",
        "I feel so lost and sad right now",
        "Book a table for dinner at 7",
        "Did you catch the match yesterday?",
        "What is the meaning of life?",
        "Don't forget to send the contract",
        "I'm really stressed about my exam",
        "Push the code to production",
        "How are you doing today?",
        "Convert 50 miles to kilometers",
    ]

    print()
    print("SAMPLE CLASSIFICATIONS")
    print("-" * 60)
    for msg in test_messages:
        res = clf.classify(msg)
        print(f"  [{res['intent']:>18}] {res['confidence']:.2f} | {msg}")

    # Benchmark
    print()
    print("BENCHMARK")
    print("-" * 60)
    bench = clf.benchmark(test_messages * 10, runs=5)
    print(f"  Messages evaluated : {bench['messages']}")
    print(f"  Total time (avg)   : {bench['avg_total_s']} s")
    print(f"  Per-message (avg)  : {bench['avg_per_msg_ms']} ms")
    print(f"  SLA <200ms         : {'PASS' if bench['meets_sla'] else 'FAIL'}")

    # Model size check
    model_mb = Path(clf.model_path).stat().st_size / (1024 * 1024)
    print(f"  Model size         : {model_mb:.2f} MB (<50 MB: {'PASS' if model_mb < 50 else 'FAIL'})")


if __name__ == "__main__":
    main()
