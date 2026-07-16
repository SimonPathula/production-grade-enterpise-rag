import json
import uuid
import requests
from pathlib import Path

API_URL = "http://localhost:8000/evaluate"
DATASET = Path(r"D:\projects\enterpriserag\tests\test_datasets\dataset copy.json")

with open(DATASET, "r", encoding="utf-8") as f:
    benchmark = json.load(f)

evaluation_dataset = []

for item in benchmark["questions"]:
    print(f"Generating: {item["id"]}")
    thread_id = f"eval_{item['id']}_{uuid.uuid4().hex}"

    response = requests.post(
        API_URL,
        json={
            "query": item["question"],
            "thread_id": thread_id
        },
        timeout=120
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
            "prompt_tokens":
            result["usage"]["input_tokens"],
            "completion_tokens":
                result["usage"]["output_tokens"],

            "total_tokens":
                result["usage"]["total_tokens"]
        }
    )

with open("evaluation_dataset.json", "w", encoding="utf-8") as f:
    json.dump(evaluation_dataset, f, indent=2, ensure_ascii=False)

print(f"Generated {len(evaluation_dataset)} evaluation samples.")