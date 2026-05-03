from sentence_transformers import SentenceTransformer
import numpy as np


class QualityEvaluator:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def semantic_similarity(self, text_a: str, text_b: str) -> float:
        emb1 = self.model.encode(text_a)
        emb2 = self.model.encode(text_b)

        similarity = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2)
        )

        return float(similarity)

    def score_response(self, query: str, response: str) -> float:
        if not response or len(response.strip()) < 5:
            return 0.1

        # Relevance: how semantically close is the response to the query
        relevance = self.semantic_similarity(query, response)

        # Length adequacy: penalize very short or empty responses
        word_count = len(response.split())
        if word_count < 5:
            length_score = 0.2
        elif word_count < 20:
            length_score = 0.6
        else:
            length_score = 1.0

        # Combined quality: 70% relevance + 30% length adequacy
        quality = 0.7 * relevance + 0.3 * length_score

        return round(min(max(quality, 0.0), 1.0), 4)
