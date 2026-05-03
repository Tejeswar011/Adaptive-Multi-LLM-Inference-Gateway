import re
import numpy as np


class FeatureExtractor:

    def extract(self, query: str) -> dict:
        words = query.split()
        word_count = len(words)

        token_length = len(query)
        avg_word_length = float(np.mean([len(w) for w in words])) if words else 0.0

        punctuation_count = len(re.findall(r'[^\w\s]', query))
        punctuation_density = punctuation_count / max(token_length, 1)

        question_type = self._detect_question_type(query)

        return {
            "token_length": token_length,
            "word_count": word_count,
            "avg_word_length": avg_word_length,
            "punctuation_density": punctuation_density,
            "question_type": question_type
        }

    def _detect_question_type(self, query: str) -> int:
        query_lower = query.lower()

        if query_lower.startswith(("what", "who", "where", "when")):
            return 0  # factual
        elif query_lower.startswith(("why", "how")):
            return 1  # reasoning
        elif "explain" in query_lower or "describe" in query_lower:
            return 2  # descriptive
        else:
            return 3  # other