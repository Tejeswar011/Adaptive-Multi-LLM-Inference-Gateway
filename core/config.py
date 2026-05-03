from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "LLM_Adapter"
    VERSION: str = "0.1.0"
    DEBUG: bool = True

    OLLAMA_BASE_URL: str = "http://localhost:11434"

    DEFAULT_MODEL_TIER: str = "tiny"

    BENCHMARK_MODE: bool = True
    BENCHMARK_SAMPLE_RATE: float = 0.2


settings = Settings()
