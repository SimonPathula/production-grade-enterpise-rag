import logfire
from qdrant_client import QdrantClient
from app.config import settings
from app.services.retrieval.embedding import embed_query

client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

def search_qdrant_db(query:str, limit: int = 8):
    try:
        query_vector = embed_query(query)

        response = client.query_points(
            collection_name= settings.QDRANT_COLLECTION,
            query = query_vector,
            limit=limit,
            with_payload= True
        )

        results = []
        for res in response.points:
            results.append({
                "content": res.payload.get("text", ""),
                "source": res.payload.get("source", "Unknown"),
                "score": res.score
            })
        return results

    except Exception as e:
        logfire.error(f"Qdrant search failed: {e}")
        raise e
        