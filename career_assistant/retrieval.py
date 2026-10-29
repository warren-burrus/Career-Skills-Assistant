"""Small keyword search (BM25-style) with light stemming. No external deps."""
from __future__ import annotations

import math
import re
from collections import Counter

STOPWORDS = {
    "a", "an", "and", "the", "to", "of", "in", "into", "on", "for", "with", "at", "by", "from",
    "or", "as", "is", "are", "be", "it", "that", "this", "your", "you", "our", "we", "what",
    "how", "who", "will", "can", "work", "move", "become", "build", "make", "not",
}
_SUFFIXES = ("ement", "ment", "ing", "ers", "er", "es", "s")


def stem(word: str) -> str:
    w = word.lower()
    for _ in range(2):
        for suf in _SUFFIXES:
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                w = w[: -len(suf)]
                break
    if w.endswith("e") and len(w) > 4:
        w = w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class KeywordIndex:
    """Weighted-field keyword index.

    docs: list of dicts with an 'id'. fields: {field_name: weight}. List-valued
    fields (e.g. skills_taught) are joined into text.
    """

    def __init__(self, docs: list[dict], fields: dict[str, float], k1: float = 1.2):
        self.k1 = k1
        self.doc_ids = [d["id"] for d in docs]
        self._tf: dict[str, Counter] = {}
        df: Counter = Counter()
        for d in docs:
            tf: Counter = Counter()
            for field, weight in fields.items():
                value = d.get(field, "")
                text = " ".join(value) if isinstance(value, list) else str(value)
                for tok in tokenize(text):
                    tf[stem(tok)] += weight
            self._tf[d["id"]] = tf
            df.update(tf.keys())
        n = len(docs)
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        q_tokens = tokenize(query)
        results = []
        for doc_id in self.doc_ids:
            tf = self._tf[doc_id]
            score, matched = 0.0, []
            for tok in dict.fromkeys(q_tokens):
                s = stem(tok)
                if s in tf:
                    score += self._idf.get(s, 0) * (tf[s] * (self.k1 + 1)) / (tf[s] + self.k1)
                    matched.append(tok)
            if score > 0:
                results.append({"id": doc_id, "score": round(score, 3), "matched_terms": matched})
        results.sort(key=lambda r: (-r["score"], r["id"]))
        return results[:top_k]
