import time
import logfire
from flashrank import RerankRequest, Ranker

_ranker = None

def _get_ranker() -> Ranker:
    global _ranker
    if _ranker is None:
        logfire.info("Initializing FlashRank Model (TinyBERT) locally...")
        try:
            # _ranker = Ranker(cache_dir="/tmp/flashrank")
            _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
        except Exception:
            _ranker = Ranker()

    return _ranker

def rerank_documents(query:str, documents: list[str], top_n: int = 5) -> list[str]:
    if not documents:
        return []

    start_time = time.time()
    logfire.info(f"[Reranker] Sending {len(documents)} docs to FlashRank Cross-Encoder...")

    try:
        ranker = _get_ranker()
        
        # FlashRank expects a list of dictionaries with 'id' and 'text'
        passages = [
            {"id": i, "text": doc}
            for i, doc in enumerate(documents)
        ]

        request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(request)
        
        reranked_docs = []
        for res in results[:top_n]:
            reranked_docs.append(res['text'])

        duration = time.time() - start_time
        top_score = results[0]['score'] if results else 'N/A'
        logfire.info(f"[Reranker] Done in {duration:.2f}s. Top semantic score: {top_score}")
        
        return reranked_docs

    except Exception as e:
        logfire.error(f"[Reranker] Semantic Reranking Failed: {e}")

        return documents[:top_n]
