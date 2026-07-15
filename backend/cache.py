import numpy as np
from typing import Dict, Any

class SemanticCache:
    def __init__(self, embedding_model, threshold: float = 0.90):
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.entries = []

    @staticmethod
    def _cosine(a, b):
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def get(self, query: str):
        if not self.entries:
            return None
        q_emb = self.embedding_model.embed_query(query)
        best_score = 0.0
        best_entry = None
        for entry in self.entries:
            score = self._cosine(q_emb, entry["embedding"])
            if score > best_score:
                best_score = score
                best_entry = entry
        if best_score >= self.threshold:
            return best_entry["response_dict"], best_score
        return None

    def set(self, query: str, response_dict: Dict[str, Any]):
        q_emb = self.embedding_model.embed_query(query)
        self.entries.append({
            "embedding": q_emb,
            "query": query,
            "response_dict": response_dict
        })
