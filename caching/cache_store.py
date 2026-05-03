import pickle
import os


class CacheStore:

    def __init__(self, persist_path="cache_data.pkl"):
        self.store = {}
        self.persist_path = persist_path
        self._load()

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value):
        self.store[key] = value
        self._persist()

    def _persist(self):
        with open(self.persist_path, "wb") as f:
            pickle.dump(self.store, f)

    def _load(self):
        if os.path.exists(self.persist_path):
            with open(self.persist_path, "rb") as f:
                self.store = pickle.load(f)