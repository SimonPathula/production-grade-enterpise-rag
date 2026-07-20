import requests
from typing import List
from nemoguardrails.embeddings.providers.base import EmbeddingModel
from app.config import settings

HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/BAAI/bge-base-en-v1.5"
HEADERS = {"Authorization": f"Bearer {settings.HF_API_TOKEN}"}


class HFAPIEmbeddingModel(EmbeddingModel):
    engine_name = "hf_api"

    def __init__(self, embedding_model: str = "BAAI/bge-base-en-v1.5", **kwargs):
        self.embedding_model = embedding_model

    def encode(self, documents: List[str]) -> List[List[float]]:
        resp = requests.post(
            HF_API_URL,
            headers=HEADERS,
            json={"inputs": documents, "options": {"wait_for_model": True}},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    async def encode_async(self, documents: List[str]) -> List[List[float]]:
        return self.encode(documents)