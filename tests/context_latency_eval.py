"""
Latency + Context Reduction Eval — hits your live /evaluate endpoint.
total_latency_ms = real wall-clock HTTP round trip (includes FastAPI/network overhead).
retrieval/rerank/generation_ms = internal stage timings, returned by the endpoint.
tokens_before = tiktoken estimate over raw_sources (top-15, never seen by Groq).
tokens_after  = exact usage.input_tokens from Groq (top-5 prompt actually sent).
"""

import time, json, httpx, tiktoken
import pandas as pd

DATASET_PATH = "tests/test_datasets/evaluation_dataset_3.json"   # [{"id": 1, "question": "..."}, ...]
OUTPUT_CSV   = "eval_results.csv"
ENDPOINT_URL = "http://localhost:8000/evaluate"
TIMEOUT_S    = 60

enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(query: str, chunks: list[str]) -> int:
    return len(enc.encode(query + " " + " ".join(chunks)))


def run_eval(dataset_path: str) -> pd.DataFrame:
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    rows = []
    with httpx.Client(timeout=TIMEOUT_S) as client:
        for item in dataset:
            qid, question = item["id"], item["question"]

            t0 = time.perf_counter()
            resp = client.post(ENDPOINT_URL, json={"query": question})
            total_latency_ms = (time.perf_counter() - t0) * 1000
            resp.raise_for_status()
            data = resp.json()

            raw_sources = data["raw_sources"]        # top-15
            sources = data["sources"]                 # top-5
            tokens_before = count_tokens(question, raw_sources)
            tokens_after = data["usage"]["input_tokens"]   # exact, from Groq

            reduction_tokens = tokens_before - tokens_after
            reduction_pct = (reduction_tokens / tokens_before * 100) if tokens_before else 0.0

            timings = data["timings_ms"]

            rows.append({
                "id": qid,
                "question": question,
                "chunks_before": len(raw_sources),
                "chunks_after": len(sources),
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "reduction_tokens": reduction_tokens,
                "reduction_pct": round(reduction_pct, 2),
                "retrieval_time_ms": round(timings["retrieval"], 2),
                "rerank_time_ms": round(timings["rerank"], 2),
                "generation_time_ms": round(timings["generation"], 2),
                "total_latency_ms": round(total_latency_ms, 2),  # real HTTP round trip
                "answer": data["answer"],
            })

            print(f"[{qid}] tokens {tokens_before}->{tokens_after} "
                  f"({reduction_pct:.1f}% cut) | total {total_latency_ms:.0f} ms")

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    return {
        "avg_chunks_before": round(df["chunks_before"].mean(), 2),
        "avg_chunks_after": round(df["chunks_after"].mean(), 2),
        "avg_tokens_before": round(df["tokens_before"].mean(), 1),
        "avg_tokens_after": round(df["tokens_after"].mean(), 1),
        "avg_reduction_pct": round(df["reduction_pct"].mean(), 2),
        "avg_retrieval_ms": round(df["retrieval_time_ms"].mean(), 2),
        "avg_rerank_ms": round(df["rerank_time_ms"].mean(), 2),
        "avg_generation_ms": round(df["generation_time_ms"].mean(), 2),
        "avg_total_latency_ms": round(df["total_latency_ms"].mean(), 2),
    }


if __name__ == "__main__":
    df = run_eval(DATASET_PATH)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows -> {OUTPUT_CSV}")

    summary = summarize(df)
    print("\n--- SUMMARY ---")
    for k, v in summary.items():
        print(f"{k}: {v}")

    with open("eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)