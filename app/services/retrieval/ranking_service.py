# import time
# import logfire
# from flashrank import RerankRequest, Ranker

# _ranker = None

# def _get_ranker() -> Ranker:
#     global _ranker
#     if _ranker is None:
#         logfire.info("Initializing FlashRank Model (TinyBERT) locally...")
#         try:
#             # _ranker = Ranker(cache_dir="/tmp/flashrank")
#             _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
#         except Exception:
#             _ranker = Ranker()

#     return _ranker

# def rerank_documents(query:str, documents: list[str], top_n: int = 4) -> list[str]:
#     if not documents:
#         return []

#     start_time = time.time()
#     logfire.info(f"[Reranker] Sending {len(documents)} docs to FlashRank Cross-Encoder...")

#     try:
#         ranker = _get_ranker()
        
#         # FlashRank expects a list of dictionaries with 'id' and 'text'
#         passages = [
#             {"id": i, "text": doc}
#             for i, doc in enumerate(documents)
#         ]

#         request = RerankRequest(query=query, passages=passages)
#         results = ranker.rerank(request)
        
#         reranked_docs = []
#         for res in results[:top_n]:
#             reranked_docs.append(res['text'])

#         duration = time.time() - start_time
#         top_score = results[0]['score'] if results else 'N/A'
#         logfire.info(f"[Reranker] Done in {duration:.2f}s. Top semantic score: {top_score}")
        
#         return reranked_docs

#     except Exception as e:
#         logfire.error(f"[Reranker] Semantic Reranking Failed: {e}")

#         return documents[:top_n]
import json
import time
import logfire
from groq import Groq
from app.config import settings

_client = None

def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client

RERANK_PROMPT = """You are a relevance-ranking engine. Given a QUERY and a numbered list of PASSAGES, return a JSON array ranking the passage indices from MOST to LEAST relevant to the query.

Respond with ONLY a JSON array of integers, e.g. [2, 0, 4, 1, 3]. No prose, no markdown fences.

QUERY: {query}

PASSAGES:
{passages}
"""

def rerank_documents(query: str, documents: list[str], top_n: int = 4, retries: int = 2) -> list[str]:
    if not documents:
        return []

    start_time = time.time()
    logfire.info(f"[Reranker] Sending {len(documents)} docs to Groq LLM reranker...")

    passages_block = "\n".join(f"[{i}] {doc}" for i, doc in enumerate(documents))
    prompt = RERANK_PROMPT.format(query=query, passages=passages_block)

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            client = _get_client()
            response = client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

            ranked_indices = json.loads(raw)

            # Validate indices before trusting them
            valid_indices = [i for i in ranked_indices if isinstance(i, int) and 0 <= i < len(documents)]
            # Append any indices the model omitted, preserving original order
            missing = [i for i in range(len(documents)) if i not in valid_indices]
            final_order = valid_indices + missing

            reranked_docs = [documents[i] for i in final_order[:top_n]]

            duration = time.time() - start_time
            logfire.info(f"[Reranker] Done in {duration:.2f}s via Groq ({settings.GROQ_MODEL}).")
            return reranked_docs

        except Exception as e:
            last_err = e
            if attempt < retries:
                logfire.warning(f"[Reranker] Groq rerank attempt {attempt}/{retries} failed: {e}. Retrying...")
                continue
            logfire.error(f"[Reranker] Groq Reranking Failed after {retries} attempts: {e}")
            return documents[:top_n]

    return documents[:top_n]