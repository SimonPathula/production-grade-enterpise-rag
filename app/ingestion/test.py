from app.config import settings
from qdrant_client import QdrantClient

client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY
)
print(f"Using key: {settings.QDRANT_API_KEY[:8]}...")