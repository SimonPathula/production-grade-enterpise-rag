import os
import json
import time
import asyncio
from pathlib import Path
import pandas as pd
import vertexai
from openai import AsyncOpenAI
from vertexai.language_models import TextEmbeddingModel
from app.config import settings
from ragas.llms import llm_factory
from ragas.embeddings import GoogleEmbeddings
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextRecall,
    ContextPrecision,
    AnswerCorrectness,
)

INPUT_FILE = Path("tests/test_datasets/evaluation_dataset.json")
OUTPUT_CSV = Path("tests/test_datasets/ragas_results.csv")
MAX_RETRIES = 5

vertexai.init(project=settings.PROJECT_ID, location=settings.LOCATION)
vertex_client = TextEmbeddingModel.from_pretrained("text-embedding-004")

embeddings = GoogleEmbeddings(
    client=vertex_client,
    model="text-embedding-004",
    use_vertex=True,
    project_id=settings.PROJECT_ID,
    location=settings.LOCATION,
)

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
llm = llm_factory(
    "gpt-4o-mini",
    client=openai_client,
    temperature = 0.01
)

faithfulness = Faithfulness(llm=llm)
answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
context_recall = ContextRecall(llm=llm)
context_precision = ContextPrecision(llm=llm)
answer_correctness = AnswerCorrectness(llm=llm, embeddings=embeddings)

def truncate(text: str, limit: int = 100) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."

async def score_row(row: dict) -> dict:
    f, ar, cr, cp, ac = await asyncio.gather(
        faithfulness.ascore(
            user_input=row["user_input"],
            response=row["response"],
            retrieved_contexts=row["retrieved_contexts"],
        ),
        answer_relevancy.ascore(
            user_input=row["user_input"],
            response=row["response"],
        ),
        context_recall.ascore(
            user_input=row["user_input"],
            retrieved_contexts=row["retrieved_contexts"],
            reference=row["reference"],
        ),
        context_precision.ascore(
            user_input=row["user_input"],
            reference=row["reference"],
            retrieved_contexts=row["retrieved_contexts"],
        ),
        answer_correctness.ascore(
            user_input=row["user_input"],
            response=row["response"],
            reference=row["reference"],
        ),
    )
    return {
        "id": row["id"],
        "user_input": truncate(row["user_input"]),
        "reference": truncate(row["reference"]),
        "response": truncate(row["response"]),
        "faithfulness": f.value,
        "answer_relevancy": ar.value,
        "context_recall": cr.value,
        "context_precision": cp.value,
        "answer_correctness": ac.value,
    }


async def main():
    with open(INPUT_FILE, encoding="utf-8") as fh:
        dataset = json.load(fh)

    if OUTPUT_CSV.exists():
        results_df = pd.read_csv(OUTPUT_CSV)
        completed = set(results_df["id"])
        print(f"Resuming... already evaluated: {len(completed)}")
    else:
        results_df = pd.DataFrame()
        completed = set()
        print("Starting fresh...")

    for row in dataset:
        if row["id"] in completed:
            print(f"Skipping {row['id']}")
            continue

        print(f"\nEvaluating {row['id']}")

        success = False
        for attempt in range(MAX_RETRIES):
            try:
                new_row = await score_row(row)
                results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
                results_df.to_csv(OUTPUT_CSV, index=False)

                print(f"✓ Completed {row['id']}")
                print(f"Faithfulness: {new_row['faithfulness']:.3f}")
                print(f"Answer Relevancy: {new_row['answer_relevancy']:.3f}")
                print(f"Context Recall: {new_row['context_recall']:.3f}")
                print(f"Context Precision: {new_row['context_precision']:.3f}")
                print(f"Answer Correctness: {new_row['answer_correctness']:.3f}")

                success = True
                break

            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                wait = 2 ** attempt
                print(e)
                print(f"Retrying in {wait} seconds...")
                await asyncio.sleep(wait)

        if not success:
            break

    # Final summary
    if not results_df.empty:
        print("\n" + "=" * 60)
        print("FINAL RAGAS EVALUATION SUMMARY")
        print("=" * 60)

        metric_columns = [
            "faithfulness",
            "answer_relevancy",
            "context_recall",
            "context_precision",
            "answer_correctness",
        ]

        averages = results_df[metric_columns].mean()

        print(f"Total Samples      : {len(results_df)}")
        print(f"Faithfulness       : {averages['faithfulness']:.3f}")
        print(f"Answer Relevancy   : {averages['answer_relevancy']:.3f}")
        print(f"Context Recall     : {averages['context_recall']:.3f}")
        print(f"Context Precision  : {averages['context_precision']:.3f}")
        print(f"Answer Correctness : {averages['answer_correctness']:.3f}")

        print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())

# ============================================================
# FINAL RAGAS EVALUATION SUMMARY
# ============================================================
# Total Samples      : 100
# Faithfulness       : 0.792
# Answer Relevancy   : 0.915
# Context Recall     : 0.868
# Context Precision  : 0.842
# Answer Correctness : 0.719
# ============================================================