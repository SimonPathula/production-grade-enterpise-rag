import os
import time
import json
import pandas as pd
from langchain_openai import ChatOpenAI
from app.services.retrieval.qdrant_service import search_qdrant_db
from app.services.retrieval.ranking_service import rerank_documents

_reasoning_model = None
_documents = []

def get_model():
    global _reasoning_model
    if _reasoning_model is None:
        try:
            _reasoning_model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-5.4-mini")
        except Exception as e:
            raise e
    return _reasoning_model

def get_documents(query:str):
    raw_results = search_qdrant_db(query, 15)
    return [doc["content"] for doc in raw_results]

def precision_without_ranker(query: str, documents: list[str]):
    documents = documents[:5]

    try:
        _reasoning_model = get_model()
    except Exception as e:
        raise e

    prompt = f"""
    You are an expert evaluator for Retrieval-Augmented Generation (RAG) systems.

    Your task is to evaluate the relevance of retrieved document chunks for answering a user query.

    USER QUERY:
    {query}

    RETRIEVED DOCUMENT CHUNKS:
    {documents}

    Evaluation Rules:
    1. Treat each document chunk independently.
    2. A document is RELEVANT only if it contains information that directly helps answer the user's query.
    3. Do NOT count documents that:
    - only mention related keywords,
    - provide vague or generic information,
    - are only partially related,
    - require significant assumptions to answer the query.
    4. Count only documents that contain sufficient and useful information for answering the query.
    5. Ignore duplicate information.

    Return ONLY a single integer between 0 and 5 representing the number of relevant document chunks.

    Output:
    <number>
    """

    answer = _reasoning_model.invoke(prompt)
    result = answer.content

    return result

def precision_with_ranker(query: str, documents: list[str]):
    reranked_docs = rerank_documents(query, documents)
    formatted_docs = [f"CONTENT: {doc}" for doc in reranked_docs]

    try:
        _reasoning_model = get_model()
    except Exception as e:
        raise e

    prompt = f"""
    You are an expert evaluator for Retrieval-Augmented Generation (RAG) systems.

    Your task is to evaluate the relevance of retrieved document chunks for answering a user query.

    USER QUERY:
    {query}

    RETRIEVED DOCUMENT CHUNKS:
    {formatted_docs}

    Evaluation Rules:
    1. Treat each document chunk independently.
    2. A document is RELEVANT only if it contains information that directly helps answer the user's query.
    3. Do NOT count documents that:
    - only mention related keywords,
    - provide vague or generic information,
    - are only partially related,
    - require significant assumptions to answer the query.
    4. Count only documents that contain sufficient and useful information for answering the query.
    5. Ignore duplicate information.

    Return ONLY a single integer between 0 and 5 representing the number of relevant document chunks.

    Output:
    <number>
    """

    answer = _reasoning_model.invoke(prompt)
    result = answer.content

    return result

def llm_reviewer(query:str, documents: list[str]):

    try:
        _reasoning_model = get_model()
    except Exception as e:
        raise e

    reranked_raw = rerank_documents(query, documents)

    reranked_documents = [f"CONTENT: {doc}" for doc in reranked_raw]
    unranked_documents = documents[:5]

    prompt = f"""
    You are an expert evaluator for Retrieval-Augmented Generation (RAG) systems.

    Your task is to compare the relevance of two retrieved document sets for the same user query.

    USER QUERY:
    {query}

    UNRANKED TOP-5 DOCUMENTS:
    {unranked_documents}

    RERANKED TOP-5 DOCUMENTS:
    {reranked_documents}

    Evaluation Rules:

    1. Evaluate the UNRANKED and RERANKED document sets independently.
    2. Treat each document chunk independently.
    3. A document is RELEVANT only if it contains information that directly helps answer the user's query.
    4. Do NOT count documents that:
    - only mention related keywords,
    - provide vague or generic information,
    - are only partially related,
    - require significant assumptions,
    - do not contribute to answering the query.
    5. Count only documents that contain sufficient information to answer the query.
    6. Ignore duplicate information. If two documents contain essentially the same information, count them only once.
    7. Return the number of relevant documents for each list. The value must be an integer between 0 and 5.

    Return ONLY valid JSON.

    Output format:

    {{
        "without_ranker": <integer>,
        "with_ranker": <integer>
    }}
    """

    answer = _reasoning_model.invoke(prompt)
    result = json.loads(answer.content)

    return result

def precision_5(json_file: str):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    print("Starting Evaluation...")

    for item in data["precision"]:
        print(f"Calculating the precision of id:{item['id']}...")
        query = item["question"]

        documents = get_documents(query)

        pre_dict = llm_reviewer(query, documents)

        results.append({
            "id": item["id"],
            "question": item["question"],
            "precision_5": pre_dict["without_ranker"],
            "precision_5_with_ranker": pre_dict["with_ranker"]
        })

        print(f"Completed the calculation of precision of id:{item['id']}")

        time.sleep(2)

    df = pd.DataFrame(results)

    return df

def final_result(json_file:str):
    df = precision_5(json_file)
    df.to_csv("precision_results.csv", index=False)
    avg_without = df["precision_5"].astype(int).mean()
    avg_with = df["precision_5_with_ranker"].astype(int).mean()

    print(f"Average Precision@5 (without ranker): {avg_without:.3f}")
    print(f"Average Precision@5 (with ranker): {avg_with:.3f}")

if __name__ == "__main__":
    final_result(r"tests\test_datasets\dataset.json")