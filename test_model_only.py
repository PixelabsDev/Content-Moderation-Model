#!/usr/bin/env python3
"""
Isolated Model Regression Testing for DistilBERT ONNX Model.

Bypasses FastAPI endpoints and deterministic regex/wordlists completely.
Loads the quantized ONNX model directly with onnxruntime and tokenizers,
measures raw float scores [0.0 - 1.0], tracks CPU inference latency,
identifies pure ML false positives and false negatives,
and dumps model regression failures to JSON.
"""

import argparse
import csv
import json
import os
import sys
import time
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

# ---------------------------------------------------------------------------
# Core Benchmark Datasets
# ---------------------------------------------------------------------------
BATTERY_V3_ORIGINAL = [
    # Benign garbage / figurative (14 cases)
    ("Please put the garbage in the trash", 0, "benign_garbage"),
    ("Please take out the garbage", 0, "benign_garbage"),
    ("Take the garbage out to the curb before sunrise", 0, "benign_garbage"),
    ("Sorting garbage and recycling is everyone's job", 0, "benign_garbage"),
    ("He works as a garbage collector in our city", 0, "benign_garbage"),
    ("The garbage truck comes on Monday", 0, "benign_garbage"),
    ("Trash pickup in our neighborhood is on Fridays", 0, "benign_garbage"),
    ("My phone battery is trash", 0, "benign_garbage"),
    ("The trash can is full, someone empty it", 0, "benign_garbage"),
    ("Garbage collection is on Tuesdays this week", 0, "benign_garbage"),
    ("I love this post, very well written", 0, "benign_clean"),
    ("I could kill for a cup of coffee right now", 0, "benign_figurative_kill"),
    ("Her presentation killed it, everyone loved it", 0, "benign_figurative_kill"),
    ("He was niggardly with his praise", 0, "benign_unusual_vocab"),
    # Toxic direct (10 cases)
    ("You are totally trash", 1, "toxic_insult"),
    ("You are trash", 1, "toxic_insult"),
    ("You are a worthless piece of garbage", 1, "toxic_insult"),
    ("Nobody likes you, just leave", 1, "toxic_harassment"),
    ("Shut up you idiot", 1, "toxic_insult"),
    ("Assholes", 1, "toxic_profanity"),
    ("NIGGER", 1, "toxic_slur"),
    ("You are such a scumbag", 1, "toxic_insult"),
    ("Stop being an assclown", 1, "toxic_insult"),
    ("That is complete horseshit", 1, "toxic_profanity"),
]

def load_all_test_cases(repo_dir: str):
    """Gathers all test cases across the benchmark battery, retest CSV, and hard negatives/positives."""
    cases = []
    seen = set()

    def add_case(text: str, expected: int, category: str, suite: str):
        cleaned = text.strip()
        key = cleaned.lower()
        if key not in seen and len(cleaned) > 0:
            seen.add(key)
            cases.append({
                "text": cleaned,
                "expected": expected,
                "category": category,
                "suite": suite,
            })

    # 1. Historical Battery (24 cases)
    for text, exp, cat in BATTERY_V3_ORIGINAL:
        add_case(text, exp, cat, "historical_battery")

    # 2. Retest cases from CSV
    csv_path = os.path.join(repo_dir, "toxicity_v1_failed_cases_retest.csv")
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("input_text", "").strip()
                exp_label = row.get("expected_label", "").lower()
                exp = 1 if exp_label in ("toxic", "1") else 0
                cat = row.get("category", "retest_case")
                add_case(text, exp, cat, "v1_retest_failures")

    # 3. Curated Hard Dataset (from train-v4 or local copy)
    possible_paths = [
        "/home/ubuntu/kaggle_dataset_train_v4/hard_negatives (1).csv",
        os.path.join(repo_dir, "hard_negatives.csv"),
        os.path.join(repo_dir, "hard_negatives (1).csv"),
    ]
    for hp in possible_paths:
        if os.path.exists(hp):
            with open(hp, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = row.get("text", "").strip()
                    exp = int(row.get("label", 0))
                    kind = row.get("kind", "hard_case")
                    add_case(text, exp, kind, "hard_cases_curated")
            break

    return cases


def evaluate_model(model_path: str, tokenizer_path: str, cases: list, max_len: int = 128):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

    print(f"Loading tokenizer from: {tokenizer_path}")
    tok = Tokenizer.from_file(tokenizer_path)
    # Check if model has dynamic sequence axis
    sess_probe = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    seq_dim = sess_probe.get_inputs()[0].shape[1]
    if isinstance(seq_dim, str) or seq_dim is None:
        tok.no_padding()
    else:
        tok.enable_padding(length=int(seq_dim))
    tok.enable_truncation(max_length=max_len)

    print(f"Loading ONNX model from: {model_path}")
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess_opts.inter_op_num_threads = 1
    session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])

    results = []
    latencies = []

    # Warmup
    warmup_enc = tok.encode("warmup inference test")
    _ = session.run(["score"], {
        "input_ids": np.array([warmup_enc.ids], dtype=np.int64),
        "attention_mask": np.array([warmup_enc.attention_mask], dtype=np.int64),
    })

    for item in cases:
        text = item["text"]
        expected = item["expected"]

        t0 = time.perf_counter()
        enc = tok.encode(text)
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        score = float(session.run(["score"], {"input_ids": ids, "attention_mask": mask})[0].ravel()[0])
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt_ms)

        predicted = 1 if score >= 0.5 else 0
        is_pass = (predicted == expected)

        failure_type = None
        if not is_pass:
            failure_type = "false_positive" if (expected == 0 and predicted == 1) else "false_negative"

        results.append({
            "text": text,
            "expected": expected,
            "score": round(score, 6),
            "predicted": predicted,
            "pass": is_pass,
            "failure_type": failure_type,
            "category": item["category"],
            "suite": item["suite"],
            "latency_ms": round(dt_ms, 3),
        })

    return results, latencies


def main():
    parser = argparse.ArgumentParser(description="Evaluate raw DistilBERT ONNX model without heuristics")
    parser.add_argument("--model", default="model.onnx", help="Path to ONNX model")
    parser.add_argument("--tokenizer", default="tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--output", default="model_regression_failures.json", help="Output JSON path for failures")
    parser.add_argument("--repo-dir", default=".", help="Repository root directory")
    args = parser.parse_args()

    # Fallback paths if not found in current dir
    model_path = args.model
    if not os.path.exists(model_path) and os.path.exists(os.path.join(args.repo_dir, "distilbert_moderation_v4_int8.onnx")):
        model_path = os.path.join(args.repo_dir, "distilbert_moderation_v4_int8.onnx")

    tokenizer_path = args.tokenizer
    if not os.path.exists(tokenizer_path) and os.path.exists(os.path.join(args.repo_dir, "tokenizer.json")):
        tokenizer_path = os.path.join(args.repo_dir, "tokenizer.json")

    cases = load_all_test_cases(args.repo_dir)
    print(f"Total test cases assembled: {len(cases)}")
    pos_count = sum(c["expected"] == 1 for c in cases)
    neg_count = sum(c["expected"] == 0 for c in cases)
    print(f"  Toxic (1): {pos_count} | Benign (0): {neg_count}")

    results, latencies = evaluate_model(model_path, tokenizer_path, cases)

    y_true = np.array([r["expected"] for r in results])
    y_pred = np.array([r["predicted"] for r in results])
    y_scores = np.array([r["score"] for r in results])

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_scores)
    avg_lat = float(np.mean(latencies))
    p95_lat = float(np.percentile(latencies, 95))

    # Benchmark subset
    bat_results = [r for r in results if r["suite"] == "historical_battery"]
    bat_acc = accuracy_score([r["expected"] for r in bat_results], [r["predicted"] for r in bat_results]) if bat_results else 0.0

    failures = [r for r in results if not r["pass"]]
    false_positives = [r for r in failures if r["failure_type"] == "false_positive"]
    false_negatives = [r for r in failures if r["failure_type"] == "false_negative"]

    print("\n" + "=" * 65)
    print("ISOLATED MODEL REGRESSION SUMMARY (RAW ONNX)")
    print("=" * 65)
    print(f"Model Tested             : {model_path}")
    print(f"Total Test Cases         : {len(results)}")
    print(f"Passed                   : {len(results) - len(failures)} / {len(results)} ({acc*100:.2f}%)")
    print(f"Historical Battery (24)  : {sum(r['pass'] for r in bat_results)} / {len(bat_results)} ({bat_acc*100:.2f}%)")
    print(f"False Positives (Clean->1): {len(false_positives)}")
    print(f"False Negatives (Toxic->0): {len(false_negatives)}")
    print(f"Overall Accuracy         : {acc*100:.2f}%")
    print(f"Precision                : {prec:.4f}")
    print(f"Recall                   : {rec:.4f}")
    print(f"Macro F1                 : {f1:.4f}")
    print(f"ROC-AUC                  : {roc_auc:.4f}")
    print(f"Mean CPU Latency         : {avg_lat:.2f} ms (< 25ms gate: {'PASS' if avg_lat < 25 else 'FAIL'})")
    print(f"P95 CPU Latency          : {p95_lat:.2f} ms")
    print("=" * 65)

    if false_positives:
        print("\n[SAMPLE FALSE POSITIVES - Raw Model Blindspots]")
        for fp in false_positives[:10]:
            print(f"  Score: {fp['score']:.4f} | Cat: {fp['category']:<20} | {fp['text']}")

    if false_negatives:
        print("\n[SAMPLE FALSE NEGATIVES - Subtle Abuse Blindspots]")
        for fn in false_negatives[:10]:
            print(f"  Score: {fn['score']:.4f} | Cat: {fn['category']:<20} | {fn['text']}")

    summary = {
        "model_path": model_path,
        "total_cases": len(results),
        "passed": len(results) - len(failures),
        "failed": len(failures),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "macro_f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "avg_cpu_latency_ms": round(avg_lat, 2),
        "p95_cpu_latency_ms": round(p95_lat, 2),
        "historical_battery_accuracy": round(float(bat_acc), 4),
        "false_positives_count": len(false_positives),
        "false_negatives_count": len(false_negatives),
    }

    output_payload = {
        "summary": summary,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "all_failures": failures,
    }

    out_file = args.output
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)
    print(f"\nModel failures dumped successfully to: {out_file}")

if __name__ == "__main__":
    main()
