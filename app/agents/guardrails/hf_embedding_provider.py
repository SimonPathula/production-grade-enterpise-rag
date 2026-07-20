import requests
from typing import List
from nemoguardrails.embeddings.providers.base import EmbeddingModel
from app.config import settings

HF_API_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-base-en-v1.5/pipeline/feature-extraction"


def _auth_headers() -> dict[str, str]:
    if not settings.HF_API_TOKEN:
        raise RuntimeError(
            "HF_API_TOKEN is required for guardrail embeddings. "
            "Set it in the deployment environment before starting the API."
        )
    return {"Authorization": f"Bearer {settings.HF_API_TOKEN}"}


class HFAPIEmbeddingModel(EmbeddingModel):
    engine_name = "hf_api"

    def __init__(self, embedding_model: str = "BAAI/bge-base-en-v1.5", **kwargs):
        self.embedding_model = embedding_model

    def encode(self, documents: List[str]) -> List[List[float]]:
        resp = requests.post(
            HF_API_URL,
            headers=_auth_headers(),
            json={"inputs": documents, "options": {"wait_for_model": True}},
            timeout=60,
        )
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            if resp.status_code == 401:
                raise RuntimeError(
                    "Hugging Face embeddings returned 401 Unauthorized. "
                    "Set a valid HF_API_TOKEN, HF_TOKEN, or HUGGINGFACEHUB_API_TOKEN "
                    "in Render environment variables."
                ) from exc
            raise
        return resp.json()

    async def encode_async(self, documents: List[str]) -> List[List[float]]:
        return self.encode(documents)
