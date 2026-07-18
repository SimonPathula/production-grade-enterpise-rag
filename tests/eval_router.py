"""
Router/Planner accuracy eval.
Runs planner_node against a labeled dataset, writes results to CSV
incrementally (resumable on rate-limit/timeout failures), then reports
accuracy, confusion matrix, and per-category breakdown.

Usage:
    python eval_router.py --dataset router_eval_dataset_200.json
    # if it stops midway (rate limit / timeout), just re-run the same
    # command — already-scored ids are skipped automatically.
"""
import json, csv, time, sys
from pathlib import Path
from dotenv import load_dotenv

# Make sure the project root (parent of tests/) is importable regardless of
# where this script is run from (double-click, `python tests/eval_router.py`,
# run from inside tests/, etc.)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")
from app.agents.nodes.planner import planner_node

TESTS_DIR    = Path(__file__).resolve().parent
DATASET_PATH = TESTS_DIR / "test_datasets" / "router_eval_dataset_200.json"
OUTPUT_CSV   = TESTS_DIR / "test_datasets" / "router_eval_results.csv"
MAX_RETRIES  = 5

CSV_FIELDS = ["id", "category", "expected_label", "predicted_label", "raw_decision", "question"]


def load_dataset(path):
    with open(path) as f:
        raw = json.load(f)["questions"]
    rows = []
    for r in raw:
        rows.append({
            "id": r["id"],
            "question": r["question"],
            "conversation_history": r.get("conversation_history", []),
            "expected_label": r.get("expected_label", "TECHNICAL"),
            "category": r.get("category", "technical"),
        })
    return rows


def build_state(row):
    messages = list(row["conversation_history"])
    messages.append({"role": "user", "content": row["question"]})
    return {"messages": messages}


def load_completed(out_path: Path):
    """Return {id: row_dict} already written, so we can resume + reuse for the final report.
    Skips (and warns about) any row that's missing a required field -- this happens if a
    previous run was interrupted mid-write and left a malformed line in the CSV. Skipped
    ids are simply re-evaluated on this run."""
    if not out_path.exists():
        return {}
    completed = {}
    skipped = []
    with open(out_path, newline="") as f:
        for row in csv.DictReader(f):
            if any(row.get(k) in (None, "") for k in CSV_FIELDS):
                skipped.append(row.get("id", "?"))
                continue
            try:
                row["id"] = int(row["id"])
            except (TypeError, ValueError):
                skipped.append(row.get("id", "?"))
                continue
            completed[row["id"]] = row
    if skipped:
        print(f"Warning: skipped {len(skipped)} malformed row(s) in {out_path}, will re-evaluate: {skipped}")
    return completed


def append_row(out_path: Path, row: dict, write_header: bool):
    with open(out_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow({k: row[k] for k in CSV_FIELDS})


def score_row(row):
    """One planner_node call -> predicted label. Retries with exponential backoff
    on timeouts / rate limits / any transient exception."""
    for attempt in range(MAX_RETRIES):
        try:
            state = build_state(row)
            out = planner_node(state)
            predicted = "CONVERSATIONAL" if out["current_query"] == "CONVERSATIONAL" else "TECHNICAL"
            return {**row, "predicted_label": predicted, "raw_decision": out["current_query"]}
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt
            print(f"  [id={row['id']}] error: {e}")
            print(f"  Retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})...")
            time.sleep(wait)


def run_eval(rows, out_path: Path):
    completed = load_completed(out_path)
    if completed:
        print(f"Resuming... already evaluated: {len(completed)}/{len(rows)}")
    else:
        print("Starting fresh...")

    write_header = not out_path.exists()
    all_results = list(completed.values())

    for row in rows:
        if row["id"] in completed:
            continue
        print(f"Evaluating id={row['id']} [{row['category']}]")
        result = score_row(row)
        append_row(out_path, result, write_header)
        write_header = False
        all_results.append(result)

    return all_results


def report(results):
    tp = fp = tn = fn = 0  # positive class = TECHNICAL
    for r in results:
        exp, pred = r["expected_label"], r["predicted_label"]
        if exp == "TECHNICAL" and pred == "TECHNICAL": tp += 1
        elif exp == "CONVERSATIONAL" and pred == "TECHNICAL": fp += 1
        elif exp == "CONVERSATIONAL" and pred == "CONVERSATIONAL": tn += 1
        elif exp == "TECHNICAL" and pred == "CONVERSATIONAL": fn += 1

    total = len(results)
    correct = tp + tn
    acc = correct / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print("\n" + "=" * 60)
    print("ROUTING ACCURACY SUMMARY")
    print("=" * 60)
    print(f"Total questions      : {total}")
    print(f"Correctly labelled   : {correct}")
    print(f"Misclassified        : {total - correct}")
    print(f"Routing Accuracy     : {acc:.2%}")
    print(f"Precision (TECHNICAL): {precision:.3f}")
    print(f"Recall (TECHNICAL)   : {recall:.3f}")
    print(f"F1 (TECHNICAL)       : {f1:.3f}")

    print("\n" + "-" * 60)
    print("CONFUSION MATRIX  (rows=expected, cols=predicted)")
    print("-" * 60)
    print(f"{'':20s}{'Pred: TECHNICAL':>18s}{'Pred: CONVERSATIONAL':>22s}")
    print(f"{'True: TECHNICAL':20s}{tp:>18d}{fn:>22d}")
    print(f"{'True: CONVERSATIONAL':20s}{fp:>18d}{tn:>22d}")
    print(f"\n  FP={fp} (conversational -> wrongly retrieved; cost/latency waste)")
    print(f"  FN={fn} (technical -> wrongly skipped retrieval; ungrounded answer risk)")

    print("\n" + "-" * 60)
    print("PER-CATEGORY ACCURACY")
    print("-" * 60)
    cats = {}
    for r in results:
        c = r["category"]
        cats.setdefault(c, {"correct": 0, "total": 0})
        cats[c]["total"] += 1
        if r["expected_label"] == r["predicted_label"]:
            cats[c]["correct"] += 1
    for c, v in sorted(cats.items()):
        print(f"  {c:25s}: {v['correct']}/{v['total']} = {v['correct']/v['total']:.2%}")

    misses = [r for r in results if r["expected_label"] != r["predicted_label"]]
    if misses:
        print("\n" + "-" * 60)
        print(f"MISCLASSIFICATIONS ({len(misses)})")
        print("-" * 60)
        for r in misses:
            print(f"  [{r['category']}] id={r['id']} expected={r['expected_label']} got={r['predicted_label']} | Q: {r['question'][:80]}")

    summary = {
        "total": total, "correct": correct, "misclassified": total - correct,
        "routing_accuracy": round(acc, 4),
        "precision_technical": round(precision, 4),
        "recall_technical": round(recall, 4),
        "f1_technical": round(f1, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "per_category": {c: {"correct": v["correct"], "total": v["total"]} for c, v in cats.items()},
    }
    with open("route_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    rows = load_dataset(DATASET_PATH)
    results = run_eval(rows, OUTPUT_CSV)
    report(results)
    print(f"\nRow-level results: {OUTPUT_CSV}")
    print("Summary: eval_summary.json")

# ============================================================
# ROUTING ACCURACY SUMMARY
# ============================================================
# Total questions      : 200
# Correctly labelled   : 182
# Misclassified        : 18
# Routing Accuracy     : 91.00%
# Precision (TECHNICAL): 0.859
# Recall (TECHNICAL)   : 1.000
# F1 (TECHNICAL)       : 0.924

# ------------------------------------------------------------
# CONFUSION MATRIX  (rows=expected, cols=predicted)
# ------------------------------------------------------------
#                        Pred: TECHNICAL  Pred: CONVERSATIONAL
# True: TECHNICAL                    110                     0
# True: CONVERSATIONAL                18                    72

#   FP=18 (conversational -> wrongly retrieved; cost/latency waste)
#   FN=0 (technical -> wrongly skipped retrieval; ungrounded answer risk)

# ------------------------------------------------------------
# PER-CATEGORY ACCURACY
# ------------------------------------------------------------
#   conversational           : 63/80 = 78.75%
#   edge_case                : 19/20 = 95.00%
#   technical                : 100/100 = 100.00%

# ------------------------------------------------------------
# MISCLASSIFICATIONS (18)
# ------------------------------------------------------------
#   [conversational] id=137 expected=CONVERSATIONAL got=TECHNICAL | Q: So which option should I actually use?
#   [conversational] id=139 expected=CONVERSATIONAL got=TECHNICAL | Q: So which option should I actually use?
#   [conversational] id=142 expected=CONVERSATIONAL got=TECHNICAL | Q: Break that down a bit more for me.
#   [conversational] id=143 expected=CONVERSATIONAL got=TECHNICAL | Q: So which option should I actually use?
#   [conversational] id=145 expected=CONVERSATIONAL got=TECHNICAL | Q: Can you explain that in simpler terms?
#   [conversational] id=146 expected=CONVERSATIONAL got=TECHNICAL | Q: Break that down a bit more for me.
#   [conversational] id=149 expected=CONVERSATIONAL got=TECHNICAL | Q: Break that down a bit more for me.
#   [conversational] id=152 expected=CONVERSATIONAL got=TECHNICAL | Q: Can you explain that in simpler terms?
#   [conversational] id=153 expected=CONVERSATIONAL got=TECHNICAL | Q: So which option should I actually use?
#   [conversational] id=155 expected=CONVERSATIONAL got=TECHNICAL | Q: Can you give me a concrete example of what you just described?
#   [conversational] id=158 expected=CONVERSATIONAL got=TECHNICAL | Q: So which option should I actually use?
#   [conversational] id=159 expected=CONVERSATIONAL got=TECHNICAL | Q: Can you explain that in simpler terms?
#   [conversational] id=161 expected=CONVERSATIONAL got=TECHNICAL | Q: Break that down a bit more for me.
#   [conversational] id=162 expected=CONVERSATIONAL got=TECHNICAL | Q: Can you give me a concrete example of what you just described?
#   [conversational] id=169 expected=CONVERSATIONAL got=TECHNICAL | Q: Are you an AI?
#   [conversational] id=175 expected=CONVERSATIONAL got=TECHNICAL | Q: Can you help with non-technical stuff too?
#   [conversational] id=176 expected=CONVERSATIONAL got=TECHNICAL | Q: What model are you running on?
#   [edge_case] id=181 expected=CONVERSATIONAL got=TECHNICAL | Q: How many pods did you say run at once with parallelism: 3?

# Row-level results: D:\projects\enterpriserag\tests\test_datasets\router_eval_results.csv
# Summary: eval_summary.json