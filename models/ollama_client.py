import json
import httpx
from httpx import ReadTimeout
from models.base_model import BaseLLM
from core.config import settings
from typing import AsyncGenerator


class OllamaClient(BaseLLM):

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.base_url = settings.OLLAMA_BASE_URL

    async def generate(self, prompt: str) -> dict:
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7
                        }
                    }
                )

                response.raise_for_status()
                data = response.json()

            text = data.get("response", "")
            tokens = len(text.split())

            return {
                "text": text,
                "tokens": tokens
            }

        except ReadTimeout as e:
            print("OLLAMA ERROR: ReadTimeout — Ollama took too long to respond.")
            return {
                "text": "",
                "tokens": 0,
                "error": "timeout"
            }
        except Exception as e:
            print("OLLAMA ERROR:", repr(e))
            return {
                "text": "",
                "tokens": 0,
                "error": str(e)
            }

    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": True,
                        "options": {
                            "temperature": 0.7
                        }
                    }
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            chunk = json.loads(line)
                            token = chunk.get("response", "")
                            if token:
                                yield token
                            if chunk.get("done", False):
                                break

        except ReadTimeout:
            yield "[ERROR: Ollama timeout]"
        except Exception as e:
            yield f"[ERROR: {e}]"
