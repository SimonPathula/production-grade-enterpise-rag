import json
import time
import uuid
import requests
from pathlib import Path

API_URL = "http://localhost:8000/evaluate"

DATASET = Path(r"D:\projects\enterpriserag\tests\test_datasets\dataset.json")
OUTPUT = Path(r"D:\projects\enterpriserag\tests\test_datasets\evaluation_dataset.json")

MAX_RETRIES = 5
TIMEOUT = 180


# -----------------------------
# Load benchmark dataset
# -----------------------------
with open(DATASET, "r", encoding="utf-8") as f:
    benchmark = json.load(f)


# -----------------------------
# Resume support
# -----------------------------
if OUTPUT.exists():
    with open(OUTPUT, "r", encoding="utf-8") as f:
        evaluation_dataset = json.load(f)

    completed_ids = {item["id"] for item in evaluation_dataset}

    print(f"Resuming...")
    print(f"Already completed: {len(completed_ids)}")
else:
    evaluation_dataset = []
    completed_ids = set()

    print("Starting fresh...")


# -----------------------------
# Generate evaluation dataset
# -----------------------------
for item in benchmark["questions"]:

    if item["id"] in completed_ids:
        print(f"Skipping {item['id']}")
        continue

    print(f"\nGenerating: {item['id']}")

    thread_id = f"eval_{item['id']}_{uuid.uuid4().hex}"

    success = False

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = requests.post(
                API_URL,
                json={
                    "query": item["question"],
                    "thread_id": thread_id,
                },
                timeout=TIMEOUT,
            )

            response.raise_for_status()

            result = response.json()

            evaluation_dataset.append(
                {
                    "id": item["id"],
                    "user_input": item["question"],
                    "reference": item["original_answer"],
                    "response": result["answer"],
                    "retrieved_contexts": result["sources"],
                    "prompt_tokens": result["usage"]["input_tokens"],
                    "completion_tokens": result["usage"]["output_tokens"],
                    "total_tokens": result["usage"]["total_tokens"],
                }
            )

            # Save immediately
            with open(OUTPUT, "w", encoding="utf-8") as f:
                json.dump(
                    evaluation_dataset,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            print(f"✓ Completed {item['id']}")

            success = True
            break

        except requests.exceptions.RequestException as e:

            print(
                f"Attempt {attempt}/{MAX_RETRIES} failed:"
            )
            print(e)

            if attempt == MAX_RETRIES:
                raise

            wait = 2 ** attempt

            print(f"Retrying in {wait} seconds...")
            time.sleep(wait)

    if not success:
        break


print("\n===================================")
print(f"Generated {len(evaluation_dataset)} samples.")
print("===================================")