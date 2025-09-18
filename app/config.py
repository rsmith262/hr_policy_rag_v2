import os
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseModel):
    aoai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    aoai_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    aoai_deployment: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    aoai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-01-preview")

    embed_deployment: str = os.getenv("AZURE_OPENAI_EMBED_DEP", "text-embedding-3-large")
    embed_api_version: str = os.getenv("AZURE_OPENAI_EMBED_API_VERSION", "2024-10-01-preview")

    search_endpoint: str = os.getenv("AZURE_SEARCH_ENDPOINT", "")
    search_key: str = os.getenv("AZURE_SEARCH_KEY", "")
    search_index: str = os.getenv("AZURE_SEARCH_INDEX", "")

    blob_account: str = os.getenv("BLOB_ACCOUNT_NAME", "")
    blob_container: str = os.getenv("BLOB_CONTAINER", "docs")

    redis_url: str = os.getenv("REDIS_URL", "")
    redis_ttl: int = int(os.getenv("REDIS_TTL_SECONDS", "604800"))

    api_key: str = os.getenv("API_KEY", "")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()
