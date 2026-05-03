import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from caching.cache_policy import LRUCachePolicy

TIER_COST = {"tiny": 1, "small": 3, "medium": 6}


class SemanticCache:

    def __init__(self, similarity_threshold=0.9, min_threshold=0.8, max_threshold=0.95):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.dimension = 384  # MiniLM embedding size
        self.index = faiss.IndexFlatIP(self.dimension)
        self.cache_store = LRUCachePolicy(max_size=500, ttl_seconds=3600)
        self.keys = []
        self.similarity_threshold = similarity_threshold

        # Adaptive threshold bounds
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.hit_count = 0
        self.miss_count = 0

    def _normalize(self, vec):
        return vec / np.linalg.norm(vec)

    def _adapt_threshold(self):
        total = self.hit_count + self.miss_count
        if total < 20:
            return

        hit_rate = self.hit_count / total

        # If hit rate too low, relax threshold to allow more cache hits
        if hit_rate < 0.1:
            self.similarity_threshold = max(
                self.similarity_threshold - 0.01, self.min_threshold
            )
        # If hit rate too high, tighten threshold for better quality matches
        elif hit_rate > 0.5:
            self.similarity_threshold = min(
                self.similarity_threshold + 0.01, self.max_threshold
            )

    def add(self, query, model_tier, response):
        embedding = self.encoder.encode(query)
        embedding = self._normalize(embedding).astype("float32")

        self.index.add(np.array([embedding]))
        key = f"{model_tier}:{len(self.keys)}"
        self.keys.append(key)

        tier_cost = TIER_COST.get(model_tier, 1)
        self.cache_store.set(key, response, tier_cost=tier_cost)

    def search(self, query, model_tier):
        if len(self.keys) == 0:
            self.miss_count += 1
            return None

        embedding = self.encoder.encode(query)
        embedding = self._normalize(embedding).astype("float32")

        D, I = self.index.search(np.array([embedding]), k=1)

        idx = I[0][0]
        similarity = D[0][0]

        if idx == -1 or similarity < self.similarity_threshold or idx >= len(self.keys):
            self.miss_count += 1
            self._adapt_threshold()
            return None

        key = self.keys[idx]

        # Ensure same model tier
        if not key.startswith(f"{model_tier}:"):
            self.miss_count += 1
            self._adapt_threshold()
            return None

        cached = self.cache_store.get(key)

        if cached:
            self.hit_count += 1
            self._adapt_threshold()
            return cached

        self.miss_count += 1
        self._adapt_threshold()
        return None

    def get_stats(self):
        total = self.hit_count + self.miss_count
        return {
            "hits": self.hit_count,
            "misses": self.miss_count,
            "hit_rate": self.hit_count / total if total > 0 else 0,
            "threshold": round(self.similarity_threshold, 4),
            "cache_size": self.cache_store.size(),
        }
