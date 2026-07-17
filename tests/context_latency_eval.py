"""
Latency + Context Reduction Eval — hits your live /evaluate endpoint.
end_to_end_latency_ms = real wall-clock HTTP round trip (includes FastAPI/network overhead).
retrieval/rerank/generation_ms = internal stage timings, returned by the endpoint.
tokens_before / tokens_after = both computed with the SAME tiktoken tokenizer (cl100k_base)
  over (question + chunks), so reduction_pct is an apples-to-apples comparison.
actual_prompt_tokens = real usage.input_tokens from Groq, kept as a separate column —
  not used in reduction_pct, since it's a different tokenizer than tiktoken.
chars_before/chars_after/char_reduction_pct = raw character-count comparison, tokenizer-agnostic.
Resumable: writes each row to CSV as it completes, skips ids already in the CSV on restart.
Retries each question up to MAX_RETRIES with exponential backoff before giving up.
"""

import time, json, httpx, tiktoken
from pathlib import Path
import pandas as pd

DATASET_PATH = Path("tests/test_datasets/evaluation_dataset.json")
OUTPUT_CSV   = Path("tests/test_datasets/eval_results.csv")
ENDPOINT_URL = "http://localhost:8000/evaluate"
TIMEOUT_S    = 60
MAX_RETRIES  = 5

enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(query: str, chunks: list[str]) -> int:
    return len(enc.encode(query + " " + " ".join(chunks)))


def score_row(client: httpx.Client, item: dict) -> dict:
    qid, question = item["id"], item["user_input"]

    t0 = time.perf_counter()
    resp = client.post(ENDPOINT_URL, json={"query": question})

    end_to_end_latency_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    data = resp.json()

    raw_chunks = data["raw_chunks"]
    reranked_chunks = data["reranked_chunks"]

    # Same tokenizer (cl100k_base) on both sides for a consistent reduction %
    tokens_before = count_tokens(question, raw_chunks)
    tokens_after = count_tokens(question, reranked_chunks)
    actual_prompt_tokens = data["usage"]["input_tokens"]   # real Groq count, kept separately

    reduction_tokens = tokens_before - tokens_after
    reduction_pct = (reduction_tokens / tokens_before * 100) if tokens_before else 0.0

    chars_before = len(question + "".join(raw_chunks))
    chars_after = len(question + "".join(reranked_chunks))
    char_reduction_pct = ((chars_before - chars_after) / chars_before * 100) if chars_before else 0.0

    timings = data["timings_ms"]

    return {
        "id": qid,
        "question": question,
        "chunks_before": len(raw_chunks),
        "chunks_after": len(reranked_chunks),
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "actual_prompt_tokens": actual_prompt_tokens,
        "reduction_tokens": reduction_tokens,
        "reduction_pct": round(reduction_pct, 2),
        "chars_before": chars_before,
        "chars_after": chars_after,
        "char_reduction_pct": round(char_reduction_pct, 2),
        "retrieval_time_ms": round(timings["retrieval"], 2) ,
        "rerank_time_ms": round(timings["rerank"], 2),
        "generation_time_ms": round(timings["generation"], 2),
        "end_to_end_latency_ms": round(end_to_end_latency_ms, 2),
        "answer": data["answer"][:100] + "...",
    }


def run_eval(dataset_path: Path) -> pd.DataFrame:
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if OUTPUT_CSV.exists():
        results_df = pd.read_csv(OUTPUT_CSV)
        completed = set(results_df["id"])
        print(f"Resuming... already evaluated: {len(completed)}")
    else:
        results_df = pd.DataFrame()
        completed = set()
        print("Starting fresh...")

    with httpx.Client(timeout=TIMEOUT_S) as client:
        for item in dataset:
            qid = item["id"]
            if qid in completed:
                print(f"Skipping {qid}")
                continue

            print(f"\nEvaluating {qid}")
            success = False
            for attempt in range(MAX_RETRIES):
                try:
                    new_row = score_row(client, item)
                    results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
                    results_df.to_csv(OUTPUT_CSV, index=False)

                    success = True
                    break

                except Exception as e:
                    if attempt == MAX_RETRIES - 1:
                        raise
                    wait = 2 ** attempt
                    print(e)
                    print(f"Retrying in {wait} seconds...")
                    time.sleep(wait)

            if not success:
                break

    return results_df


def summarize(df: pd.DataFrame) -> dict:
    return {
        "Average Chunks Before": round(df["chunks_before"].mean(), 2),
        "Average Chunks After": round(df["chunks_after"].mean(), 2),
        "Average Tokens Before": round(df["tokens_before"].mean(), 1),
        "Average Tokens After": round(df["tokens_after"].mean(), 1),
        "Average Actual Prompt Tokens": round(df["actual_prompt_tokens"].mean(), 1),
        "Average Token Reduction %": round(df["reduction_pct"].mean(), 2),
        "Average Characters Before": round(df["chars_before"].mean(), 1),
        "Average Characters After": round(df["chars_after"].mean(), 1),
        "Average Character Reduction %": round(df["char_reduction_pct"].mean(), 2),
        "Average Retrieval Latency": round(df["retrieval_time_ms"].mean(), 2),
        "Average Rerank Latency": round(df["rerank_time_ms"].mean(), 2),
        "Average Generation Latency": round(df["generation_time_ms"].mean(), 2),
        "Average End-to-End Latency": round(df["end_to_end_latency_ms"].mean(), 2)
    }


if __name__ == "__main__":
    df = run_eval(DATASET_PATH)

    if df.empty:
        print("No results to summarize.")
    else:
        summary = summarize(df)
        print("\n" + "=" * 60)
        print("FINAL LATENCY / CONTEXT REDUCTION SUMMARY")
        print("=" * 60)
        for k, v in summary.items():
            print(f"{k:<24}: {v}")
        print("=" * 60)

        with open("eval_summary.json", "w") as f:
            json.dump(summary, f, indent=2)