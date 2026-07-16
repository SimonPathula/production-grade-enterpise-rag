import json, time
import httpx
from pathlib import Path

API_URL = "http://localhost:8000/query"
DATASET = Path("tests/test_datasets/dataset.json")
CHECKPOINT = Path("ragas_checkpoint.jsonl")

def main():
    questions = json.loads(DATASET.read_text())["questions"]
    done_ids = set()
    if CHECKPOINT.exists():
        done_ids = {json.loads(l)["id"] for l in CHECKPOINT.read_text().splitlines() if l.strip()}

    with httpx.Client(timeout=120) as client, open(CHECKPOINT, "a", encoding="utf-8") as f:
        for q in questions:
            if q["id"] in done_ids:
                continue
            try:
                resp = client.post(API_URL, json={
                    "query": q["question"],
                    "thread_id": f"ragas_eval_{q['id']}",   # unique -> no MemorySaver bleed
                })
                resp.raise_for_status()
                data = resp.json()
                contexts = [d.removeprefix("CONTENT: ") for d in data.get("sources", [])]

                record = {
                    "id": q["id"],
                    "user_input": q["question"],
                    "response": data.get("answer", ""),
                    "retrieved_contexts": contexts,
                    "reference": q["original_answer"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[OK] {q['id']}")
            except Exception as e:
                print(f"[FAIL] {q['id']}: {e}")
            time.sleep(0.5)  # gentle on Groq rate limits

if __name__ == "__main__":
    main()