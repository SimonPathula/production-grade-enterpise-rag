import os
import json
import vertexai
import pandas as pd
from vertexai.language_models import TextEmbeddingModel
from app.config import settings
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextRecall,
    LLMContextPrecisionWithReference,
    AnswerCorrectness,
)
from ragas import evaluate

embeddings = None

def get_embedding_model():
    global embeddings
    if embeddings is None:
        
        vertexai.init(project=settings.PROJECT_ID, location=settings.LOCATION)

        embeddings = TextEmbeddingModel.from_pretrained("text-embedding-004")

    return embeddings

judge_llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-5.4-mini",
    temperature=0,
)

with open(
    "tests/test_datasets/evaluation_dataset.json",
    encoding="utf-8",
) as f:
    data = json.load(f)

samples = []
for row in data:

    samples.append(
        SingleTurnSample(
            user_input=row["user_input"],
            response=row["response"],
            reference=row["reference"],
            retrieved_contexts=row["retrieved_contexts"],
        )
    )

dataset = EvaluationDataset(samples=samples)

ragas_llm = LangchainLLMWrapper(judge_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

metrics = [
    Faithfulness(),
    ResponseRelevancy(),
    LLMContextRecall(),
    LLMContextPrecisionWithReference(),
    AnswerCorrectness(),
]

results = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=ragas_llm,
    embeddings=ragas_embeddings,
)

print(results)

df = results.to_pandas()

print(df.head())

df.to_csv(
    "ragas_results.csv",
    index=False,
)