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
# Token counters
# -----------------------------
prompt_token_sum = sum(
    item.get("prompt_tokens", 0)
    for item in evaluation_dataset
)

completion_token_sum = sum(
    item.get("completion_tokens", 0)
    for item in evaluation_dataset
)

total_token_sum = sum(
    item.get("total_tokens", 0)
    for item in evaluation_dataset
)

TPD_LIMIT = 100_000

print(f"Current token usage: {total_token_sum:,}/{TPD_LIMIT:,}")


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
            prompt_token_sum += result["usage"]["input_tokens"]
            completion_token_sum += result["usage"]["output_tokens"]
            total_token_sum += result["usage"]["total_tokens"]

            # Save immediately
            with open(OUTPUT, "w", encoding="utf-8") as f:
                json.dump(
                    evaluation_dataset,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            remaining = TPD_LIMIT - total_token_sum

            print(
                f"✓ Completed {item['id']} | "
                f"Prompt: {result['usage']['input_tokens']} | "
                f"Completion: {result['usage']['output_tokens']} | "
                f"Total: {result['usage']['total_tokens']}"
            )

            print(
                f"Running Total -> "
                f"Prompt: {prompt_token_sum:,} | "
                f"Completion: {completion_token_sum:,} | "
                f"Total: {total_token_sum:,}/{TPD_LIMIT:,} "
                f"({remaining:,} remaining)"
            )

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

print(f"Prompt Tokens     : {prompt_token_sum:,}")
print(f"Completion Tokens : {completion_token_sum:,}")
print(f"Total Tokens      : {total_token_sum:,}/{TPD_LIMIT:,}")

if len(evaluation_dataset):
    print(f"Average / Question: {total_token_sum / len(evaluation_dataset):.2f}")


# ===================================
# Generated 100 samples.
# ===================================
# Prompt Tokens     : 70,237
# Completion Tokens : 24,509
# Total Tokens      : 94,746/100,000
# Average / Question: 947.46
