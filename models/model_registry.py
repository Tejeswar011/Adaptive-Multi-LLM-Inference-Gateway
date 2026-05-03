from models.ollama_client import OllamaClient


class ModelRegistry:

    def __init__(self):
        self.models = {
            "tiny": OllamaClient("tinyllama"),
            "small": OllamaClient("phi3"),
            "medium": OllamaClient("mistral"),
        }

    def get(self, tier: str):
        return self.models.get(tier)