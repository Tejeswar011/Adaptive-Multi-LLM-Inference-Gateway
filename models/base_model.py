from abc import ABC, abstractmethod
from typing import AsyncGenerator


class BaseLLM(ABC):

    @abstractmethod
    async def generate(self, prompt: str) -> dict:
        pass

    @abstractmethod
    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        pass
