#!/usr/bin/env python3
"""
Prepares the 80:20 Retraining Dataset for DistilBERT.

- 80% Baseline: Sampled from Jigsaw/Civil Comments (SetFit/toxic_conversations)
- 20% Hard Cases: Focused injection of ML failures from model_regression_failures.json
  and curated hard edge-cases (quoted abuse, untargeted profanity, indirect threats).
- Strictly holds out the 24 benchmark battery cases.
- Saves preprocessed train.csv and val.csv ready for Kaggle ingestion.
"""

import csv
import json
import os
import random
import numpy as np
from datasets import load_dataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TOTAL_TRAIN = 24_000   # 19,200 baseline (80%) + 4,800 hard cases (20%)
TOTAL_VAL   = 3_000    # 2,400 baseline (80%) + 600 hard cases (20%)

RATIO_BASELINE = 0.80
RATIO_HARD     = 0.20

TRAIN_BASELINE_COUNT = int(TOTAL_TRAIN * RATIO_BASELINE)  # 19,200
TRAIN_HARD_COUNT     = int(TOTAL_TRAIN * RATIO_HARD)      # 4,800

VAL_BASELINE_COUNT   = int(TOTAL_VAL * RATIO_BASELINE)    # 2,400
VAL_HARD_COUNT       = int(TOTAL_VAL * RATIO_HARD)        # 600

# Benchmark battery cases to STRICTLY exclude from training data
BATTERY_V3_ORIGINAL = [
    "Please put the garbage in the trash",
    "Please take out the garbage",
    "Take the garbage out to the curb before sunrise",
    "Sorting garbage and recycling is everyone's job",
    "He works as a garbage collector in our city",
    "The garbage truck comes on Monday",
    "Trash pickup in our neighborhood is on Fridays",
    "My phone battery is trash",
    "The trash can is full, someone empty it",
    "Garbage collection is on Tuesdays this week",
    "I love this post, very well written",
    "I could kill for a cup of coffee right now",
    "Her presentation killed it, everyone loved it",
    "He was niggardly with his praise",
    "You are totally trash",
    "You are trash",
    "You are a worthless piece of garbage",
    "Nobody likes you, just leave",
    "Shut up you idiot",
    "Assholes",
    "NIGGER",
    "You are such a scumbag",
    "Stop being an assclown",
    "That is complete horseshit",
]
HELD_OUT_BENCHMARK = {t.strip().lower() for t in BATTERY_V3_ORIGINAL}


def load_failures_and_hardcases(repo_dir: str):
    """Gathers all failure cases and curated hard examples."""
    hard_negatives = []
    hard_positives = []
    seen = set()

    def add_hard(text: str, label: int):
        cleaned = text.strip()
        key = cleaned.lower()
        if key in HELD_OUT_BENCHMARK or key in seen or len(cleaned) < 3:
            return
        seen.add(key)
        if label == 0:
            hard_negatives.append(cleaned)
        else:
            hard_positives.append(cleaned)

    # 1. From model_regression_failures.json
    fail_path = os.path.join(repo_dir, "model_regression_failures.json")
    if os.path.exists(fail_path):
        with open(fail_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data.get("false_positives", []):
            add_hard(item["text"], 0)
        for item in data.get("false_negatives", []):
            add_hard(item["text"], 1)

    # 2. From hard_negatives (1).csv
    csv_paths = [
        "/home/ubuntu/kaggle_dataset_train_v4/hard_negatives (1).csv",
        os.path.join(repo_dir, "hard_negatives (1).csv"),
    ]
    for cp in csv_paths:
        if os.path.exists(cp):
            with open(cp, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = row.get("text", "")
                    lbl = int(row.get("label", 0))
                    add_hard(text, lbl)
            break

    # 3. Targeted augmentations for the specific blindspot categories
    targeted_neg = [
        "The original comment said, \"You are an idiot.\" I am quoting it only to document what was written.",
        "The user reported that another person called him an idiot during the discussion. I am only describing what happened.",
        "A moderator removed the comment because it contained the word \"idiot\" directed at another user.",
        "The reported comment contained phrases such as \"you are stupid\" and \"shut up.\" These are quoted as evidence.",
        "Someone wrote \"you are worthless and nobody likes you.\" That statement was abusive and should not have been directed at anyone.",
        "What the hell happened to my computer? It was working perfectly yesterday.",
        "This damn application keeps crashing whenever I upload a large file.",
        "The character shouted \"you bastard\" in the film, and the audience laughed at the scene.",
        "The article quoted him saying 'this is bullshit'.",
        "Where the fuck is my phone?",
        "Black Lives Matter is an important movement.",
        "As a gay man, I support this policy.",
        "Stop calling people idiots, it is not okay.",
        "In his testimony, he quoted the defendant saying 'I will get you'.",
        "The harassment report cited the message: 'you are useless'.",
        "The user was suspended after typing 'shut up you moron'.",
        "Why the hell is this train delayed again?",
        "This damn phone battery drains within two hours.",
        "Damn it, I spilled coffee on my notebook.",
        "He shouted 'what the hell' when the power went out.",
        "Transgender equality is a vital human right.",
        "Muslim and Jewish leaders held an interfaith vigil in our community.",
        "I am disabled and require wheelchair accessibility.",
        "Please refrain from calling other members foolish or dumb.",
        "Harassment is strictly prohibited under our community guidelines.",
        "Mocking someone as worthless is against our rules.",
        "I could kill for a cold glass of lemonade right now.",
        "I could kill for a slice of pizza after work.",
        "His guitar solo totally killed at the concert.",
        "The presentation killed today, client was impressed.",
        "Remember to kill the stale test process on the server.",
        "We killed the deprecated service after migrating users.",
    ]
    targeted_pos = [
        "Keep talking, and you will find out what happens next.",
        "You better sleep with one eye open.",
        "Your family will pay for this.",
        "You picked the wrong person to mess with.",
        "This is your final warning, back off.",
        "Nobody likes you and nobody would miss you.",
        "Watch your back, you will regret this.",
        "I know where you live.",
        "Enjoy your last days, I will find you.",
        "Just wait until we cross paths in person.",
        "You think you are safe behind a keyboard, wait and see.",
        "I will make sure you regret ever posting that.",
        "Keep testing my patience and find out what follows.",
        "I have your address, do not tempt me.",
        "You will be sorry you ever opened your mouth.",
        "One day very soon, someone will teach you a lesson you will not forget.",
        "The world would truly be a much better place without your existence.",
        "Nobody ever wanted you here, you are a complete waste of breath.",
        "You are utterly pathetic and everybody laughs at you behind your back.",
        "Why do you even bother trying, you are completely worthless.",
    ]

    for t in targeted_neg:
        add_hard(t, 0)
    for t in targeted_pos:
        add_hard(t, 1)

    print(f"Loaded {len(hard_negatives)} unique hard negatives and {len(hard_positives)} unique hard positives.")
    return hard_negatives, hard_positives


def sample_baseline_data(total_needed: int):
    """Stream balanced baseline data from Civil Comments / SetFit dataset."""
    print(f"Streaming {total_needed} baseline samples from SetFit/toxic_conversations...")
    ds = load_dataset("SetFit/toxic_conversations", split="train", streaming=True)

    toxic_needed = total_needed // 2
    benign_needed = total_needed - toxic_needed

    toxic_pool = []
    benign_pool = []

    for item in ds:
        text = item["text"].strip()
        label = int(item["label"])
        key = text.lower()
        if key in HELD_OUT_BENCHMARK or len(text) < 5:
            continue

        if label == 1 and len(toxic_pool) < toxic_needed:
            toxic_pool.append((text, 1))
        elif label == 0 and len(benign_pool) < benign_needed:
            benign_pool.append((text, 0))

        if len(toxic_pool) >= toxic_needed and len(benign_pool) >= benign_needed:
            break

    baseline_samples = toxic_pool + benign_pool
    random.shuffle(baseline_samples)
    print(f"Sampled baseline: {len(toxic_pool)} toxic + {len(benign_pool)} benign = {len(baseline_samples)} rows.")
    return baseline_samples


def build_hard_split(hard_neg: list, hard_pos: list, count: int):
    """Generates an oversampled, balanced pool of hard cases."""
    half = count // 2
    # Oversample hard negatives and hard positives
    sampled_neg = [random.choice(hard_neg) for _ in range(half)]
    sampled_pos = [random.choice(hard_pos) for _ in range(count - half)]
    hard_pool = [(t, 0) for t in sampled_neg] + [(t, 1) for t in sampled_pos]
    random.shuffle(hard_pool)
    return hard_pool


def main():
    repo_dir = "/home/ubuntu/Content-Moderation-Model"
    out_dir = os.path.join(repo_dir, "data_v4")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Hard cases
    hard_neg, hard_pos = load_failures_and_hardcases(repo_dir)

    # 2. Baseline cases
    total_baseline_needed = TRAIN_BASELINE_COUNT + VAL_BASELINE_COUNT
    baseline_all = sample_baseline_data(total_baseline_needed)

    train_baseline = baseline_all[:TRAIN_BASELINE_COUNT]
    val_baseline   = baseline_all[TRAIN_BASELINE_COUNT:TRAIN_BASELINE_COUNT + VAL_BASELINE_COUNT]

    # 3. Hard cases splits
    train_hard = build_hard_split(hard_neg, hard_pos, TRAIN_HARD_COUNT)
    val_hard   = build_hard_split(hard_neg, hard_pos, VAL_HARD_COUNT)

    # 4. Combine with 80:20 ratio
    train_data = train_baseline + train_hard
    val_data   = val_baseline + val_hard

    random.shuffle(train_data)
    random.shuffle(val_data)

    print(f"\nFinal Train Set: {len(train_data)} samples")
    print(f"  Baseline (80%): {len(train_baseline)} ({len(train_baseline)/len(train_data)*100:.1f}%)")
    print(f"  Hard Cases (20%): {len(train_hard)} ({len(train_hard)/len(train_data)*100:.1f}%)")
    print(f"  Toxic: {sum(y for _, y in train_data)} | Benign: {sum(1-y for _, y in train_data)}")

    print(f"\nFinal Val Set: {len(val_data)} samples")
    print(f"  Baseline (80%): {len(val_baseline)} ({len(val_baseline)/len(val_data)*100:.1f}%)")
    print(f"  Hard Cases (20%): {len(val_hard)} ({len(val_hard)/len(val_data)*100:.1f}%)")
    print(f"  Toxic: {sum(y for _, y in val_data)} | Benign: {sum(1-y for _, y in val_data)}")

    # 5. Save train.csv and val.csv
    train_csv_path = os.path.join(out_dir, "train.csv")
    val_csv_path   = os.path.join(out_dir, "val.csv")

    with open(train_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(train_data)

    with open(val_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(val_data)

    meta = {
        "train_total": len(train_data),
        "train_baseline": len(train_baseline),
        "train_hard": len(train_hard),
        "train_ratio_baseline": round(len(train_baseline) / len(train_data), 2),
        "train_ratio_hard": round(len(train_hard) / len(train_data), 2),
        "val_total": len(val_data),
        "val_baseline": len(val_baseline),
        "val_hard": len(val_hard),
        "held_out_benchmark_cases": len(HELD_OUT_BENCHMARK),
    }

    with open(os.path.join(out_dir, "data_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved preprocessed splits to {out_dir}:")
    print(f"  - {train_csv_path} ({os.path.getsize(train_csv_path)/1024:.1f} KB)")
    print(f"  - {val_csv_path} ({os.path.getsize(val_csv_path)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
