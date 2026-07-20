# # import vertexai
# # from vertexai.language_models import TextEmbeddingModel
# from sentence_transformers import SentenceTransformer
# from app.config import settings

# model = None
# BATCH_SIZE = 50

# def get_embedding_model():
#     global model
#     if model is None:
#         #Initialize the VertexAI before loading the model
#         # vertexai.init(project=settings.PROJECT_ID, location=settings.LOCATION)
#         # model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        
#         # Load HuggingFace model (BAAI/bge-base-en-v1.5 has 768 dimensions, same as text-embedding-004)
#         model = SentenceTransformer("BAAI/bge-base-en-v1.5")

#     return model

# def embed_query(query:str):
#     model = get_embedding_model()
#     # embeddings = model.get_embeddings([query])
#     # return embeddings[0].values
    
#     # Generate HuggingFace embeddings
#     embedding = model.encode([query]).tolist()
#     return embedding[0]

# def embed_texts(texts: list[str]):
#     model = get_embedding_model()
#     all_embeddings = []

#     for i in range(0, len(texts), BATCH_SIZE):
#         batch = texts[i : i + BATCH_SIZE]
#         # embeddings = model.get_embeddings(batch)
#         # all_embeddings.extend([e.values for e in embeddings])
        
#         # Generate HuggingFace embeddings for batch
#         embeddings = model.encode(batch).tolist()
#         all_embeddings.extend(embeddings)

#     return all_embeddings

import requests
from app.config import settings

HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/BAAI/bge-base-en-v1.5"
HEADERS = {"Authorization": f"Bearer {settings.HF_API_TOKEN}"}
BATCH_SIZE = 50


def _call_hf(texts: list[str], retries: int = 3, backoff_base: float = 2.0):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                HF_API_URL,
                headers=HEADERS,
                json={"inputs": texts, "options": {"wait_for_model": True}},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            if attempt == retries:
                raise
            import time
            time.sleep(backoff_base ** attempt)
    raise last_err


def embed_query(query: str):
    return _call_hf([query])[0]


def embed_texts(texts: list[str]):
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        all_embeddings.extend(_call_hf(batch))
    return all_embeddings