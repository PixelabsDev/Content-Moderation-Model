#!/usr/bin/env python3
"""
Comprehensive Evaluation & Quality Gate Script for DistilBERT Content Moderation Model v5.
Bypasses rule/regex engines; tests raw ONNX runtime model inference directly.
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

# 24-case Historical Baseline Battery
HISTORICAL_BATTERY = [
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

KEY_V5_TARGET_CASES = ["W6", "W7", "F7", "F30", "F31", "F32", "F39", "F44"]


def load_model(onnx_path: str, tokenizer_path: str):
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess_opts.inter_op_num_threads = 1
    session = ort.InferenceSession(onnx_path, sess_opts, providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return session, tokenizer


def predict_single(session, tokenizer, text: str):
    t0 = time.perf_counter()
    enc = tokenizer.encode(text)
    ids = np.array([enc.ids], dtype=np.int64)
    mask = np.array([enc.attention_mask], dtype=np.int64)
    (score_arr,) = session.run(["score"], {"input_ids": ids, "attention_mask": mask})
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return float(score_arr[0][0]), latency_ms


def run_eval_suite_v5(session, tokenizer, csv_path: str):
    results = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("case_id", "").strip()
            text = row.get("input_text", "").strip()
            exp_str = row.get("expected", "").strip().lower()
            cat = row.get("category", "").strip()
            suite = row.get("suite", "").strip()

            if exp_str == "toxic":
                expected_int = 1
            elif exp_str == "benign":
                expected_int = 0
            else:
                expected_int = -1  # borderline

            score, latency_ms = predict_single(session, tokenizer, text)
            pred_int = 1 if score >= 0.5 else 0

            passed = False
            if expected_int == 1:
                passed = (pred_int == 1)
            elif expected_int == 0:
                passed = (pred_int == 0)
            else:
                passed = True  # borderline accepts either or review

            results.append({
                "case_id": cid,
                "text": text,
                "category": cat,
                "suite": suite,
                "expected_str": exp_str,
                "expected_int": expected_int,
                "score": score,
                "predicted_int": pred_int,
                "passed": passed,
                "latency_ms": latency_ms,
            })
    return results


def run_historical_battery(session, tokenizer):
    results = []
    for text, expected, cat in HISTORICAL_BATTERY:
        score, latency_ms = predict_single(session, tokenizer, text)
        pred = 1 if score >= 0.5 else 0
        passed = (pred == expected)
        results.append({
            "text": text,
            "category": cat,
            "expected": expected,
            "score": score,
            "predicted": pred,
            "passed": passed,
            "latency_ms": latency_ms,
        })
    return results


def benchmark_latency(session, tokenizer, num_runs=100):
    sample_texts = [
        "Please put the garbage in the trash",
        "That is the dumbest thing I have read all week.",
        "Nobody would notice if you disappeared.",
        "You are an idiot and you clearly have no idea what you are talking about.",
        "The reviewer wrote that the film was 'absolute garbage'.",
    ]
    # Warmup
    for t in sample_texts:
        predict_single(session, tokenizer, t)

    latencies = []
    for i in range(num_runs):
        t = sample_texts[i % len(sample_texts)]
        _, lat = predict_single(session, tokenizer, t)
        latencies.append(lat)

    latencies = np.array(latencies)
    return {
        "mean_ms": float(np.mean(latencies)),
        "p50_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
    }


def compute_metrics(eval_results):
    binary_cases = [r for r in eval_results if r["expected_int"] in (0, 1)]
    y_true = [r["expected_int"] for r in binary_cases]
    y_pred = [r["predicted_int"] for r in binary_cases]
    y_score = [r["score"] for r in binary_cases]

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = 0.0

    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    total_benign = sum(1 for yt in y_true if yt == 0)
    total_toxic = sum(1 for yt in y_true if yt == 1)

    fp_rate = fp / total_benign if total_benign > 0 else 0.0
    fn_rate = fn / total_toxic if total_toxic > 0 else 0.0

    return {
        "total_cases": len(eval_results),
        "binary_cases": len(binary_cases),
        "accuracy": acc,
        "macro_f1": f1,
        "precision": prec,
        "recall": rec,
        "roc_auc": auc,
        "false_positives": fp,
        "false_negatives": fn,
        "fpr": fp_rate,
        "fnr": fn_rate,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate DistilBERT ONNX Moderation Model")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--tokenizer", default="tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--csv", default="eval_suite_v5.csv", help="Path to eval_suite_v5.csv")
    parser.add_argument("--out", default="eval_report.json", help="Output JSON report path")
    args = parser.parse_args()

    print(f"=== Evaluating Model: {args.onnx} ===")
    session, tokenizer = load_model(args.onnx, args.tokenizer)

    # 1. Historical Battery
    hist_results = run_historical_battery(session, tokenizer)
    hist_passed = sum(1 for r in hist_results if r["passed"])
    print(f"Historical 24-Case Battery: {hist_passed}/24 passed ({hist_passed/24*100:.1f}%)")
    for r in hist_results:
        if not r["passed"]:
            print(f"  [HISTORICAL FAIL] Exp:{r['expected']} Pred:{r['predicted']} ({r['score']:.4f}) -> {r['text']}")

    # 2. Eval Suite v5 (95 cases)
    suite_results = run_eval_suite_v5(session, tokenizer, args.csv)
    metrics = compute_metrics(suite_results)
    suite_passed = sum(1 for r in suite_results if r["passed"])
    print(f"Eval Suite v5: {suite_passed}/{len(suite_results)} passed ({suite_passed/len(suite_results)*100:.1f}%)")
    print(f"Metrics (binary cases): Accuracy={metrics['accuracy']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}, Macro-F1={metrics['macro_f1']:.4f}")
    print(f"FPR={metrics['fpr']*100:.2f}% ({metrics['false_positives']}), FNR={metrics['fnr']*100:.2f}% ({metrics['false_negatives']})")

    # 3. Key Targets Check
    print("\n--- Key Target Cases Check ---")
    target_results = {}
    for r in suite_results:
        if r["case_id"] in KEY_V5_TARGET_CASES:
            target_results[r["case_id"]] = r
            status = "PASS" if r["passed"] else "FAIL"
            print(f"[{r['case_id']}] ({status}) Exp: {r['expected_str']:<10} Score: {r['score']:.4f} Pred: {r['predicted_int']} -> {r['text']}")

    # 4. Latency Benchmark
    latency_stats = benchmark_latency(session, tokenizer, 100)
    print("\n--- CPU Latency Benchmark (100 runs) ---")
    print(f"Mean Latency:   {latency_stats['mean_ms']:.2f} ms")
    print(f"P50 Latency:    {latency_stats['p50_ms']:.2f} ms")
    print(f"P95 Latency:    {latency_stats['p95_ms']:.2f} ms")
    print(f"P99 Latency:    {latency_stats['p99_ms']:.2f} ms")

    report = {
        "model_path": args.onnx,
        "tokenizer_path": args.tokenizer,
        "historical_battery": {
            "total": len(hist_results),
            "passed": hist_passed,
            "accuracy": hist_passed / len(hist_results),
            "results": hist_results,
        },
        "suite_v5": {
            "metrics": metrics,
            "passed_total": suite_passed,
            "key_targets": target_results,
            "all_results": suite_results,
        },
        "latency": latency_stats,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    main()
