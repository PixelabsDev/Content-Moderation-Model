#!/usr/bin/env python3
"""
DistilBERT v5 Retraining Pipeline on Kaggle GPU.
Features:
- Focal Loss with class weighting
- AdamW with LR 2e-5 and Linear Warmup (first 10% steps)
- Early Stopping on Validation ROC-AUC (patience=3)
- Restores best ROC-AUC checkpoint
- Section 5 Historical Battery Deploy Gate (Strict 24/24 = 100%)
- Dynamic Sequence ONNX Export & INT8 Dynamic Quantization (~67.4 MB)
- Quantization Precision Gate: PyTorch vs INT8 Max Diff strictly < 0.20
- CPU Latency Gate strictly < 25ms
"""

import os
import sys
import time
import math
import hashlib
import traceback
import subprocess
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

# Helper to install missing packages
def pip(*packages):
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check"] + list(packages)
    subprocess.check_call(cmd)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    pip("torch")
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

try:
    import transformers
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
except ImportError:
    pip("transformers")
    import transformers
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

try:
    import onnx
    import onnxruntime
    import onnxscript
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError:
    pip("onnxruntime", "onnx", "onnxscript")
    import onnx
    import onnxruntime
    from onnxruntime.quantization import quantize_dynamic, QuantType

# ---------------------------------------------------------------------------
# Directories & Paths (v5 Standard)
# ---------------------------------------------------------------------------
OUT_DIR = "v5_out"
os.makedirs(OUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUT_DIR, "train_v5.log")

def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{ts}] {msg}"
    print(formatted, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")

def set_status(stage: str, state="RUNNING"):
    with open(os.path.join(OUT_DIR, "STATUS.txt"), "w") as f:
        f.write(f"STATE: {state}\nSTAGE: {stage}\nWHEN : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

import glob

# Locate Dataset (train-v5 or fallback data_v5)
def locate_dataset_dir():
    patterns = [
        "/kaggle/input/**/train.csv",
        "./data_v5/train.csv",
        "../data_v5/train.csv",
        "data_v5/train.csv",
        "/home/ubuntu/Content-Moderation-Model/data_v5/train.csv",
    ]
    for pat in patterns:
        matches = glob.glob(pat, recursive=True)
        for m in matches:
            d = os.path.dirname(m)
            if os.path.exists(os.path.join(d, "val.csv")):
                return d
    return None

DATA_DIR = locate_dataset_dir()
if DATA_DIR is None:
    raise FileNotFoundError("Could not locate train.csv and val.csv in any candidate directory!")

log(f"Using dataset directory: {DATA_DIR}")

# ---------------------------------------------------------------------------
# Historical 24-Case Battery (Quality Gate)
# ---------------------------------------------------------------------------
HISTORICAL_BATTERY = [
    # Benign garbage / figurative (14 cases)
    ("Please put the garbage in the trash", 0),
    ("Please take out the garbage", 0),
    ("Take the garbage out to the curb before sunrise", 0),
    ("Sorting garbage and recycling is everyone's job", 0),
    ("He works as a garbage collector in our city", 0),
    ("The garbage truck comes on Monday", 0),
    ("Trash pickup in our neighborhood is on Fridays", 0),
    ("My phone battery is trash", 0),
    ("The trash can is full, someone empty it", 0),
    ("Garbage collection is on Tuesdays this week", 0),
    ("I love this post, very well written", 0),
    ("I could kill for a cup of coffee right now", 0),
    ("Her presentation killed it, everyone loved it", 0),
    ("He was niggardly with his praise", 0),
    # Toxic direct (10 cases)
    ("You are totally trash", 1),
    ("You are trash", 1),
    ("You are a worthless piece of garbage", 1),
    ("Nobody likes you, just leave", 1),
    ("Shut up you idiot", 1),
    ("Assholes", 1),
    ("NIGGER", 1),
    ("You are such a scumbag", 1),
    ("Stop being an assclown", 1),
    ("That is complete horseshit", 1),
]

# ---------------------------------------------------------------------------
# Focal Loss Implementation
# ---------------------------------------------------------------------------
class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, alpha=0.55, gamma=1.5, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * ((1 - p_t) ** self.gamma)
        loss = focal_weight * bce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

# ---------------------------------------------------------------------------
# Dataset & Model Setup
# ---------------------------------------------------------------------------
class ModerationTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = float(self.labels[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label": label,
        }

def collate_fn(batch, pad_token_id=0):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    attention_mask = []
    labels = []
    for x in batch:
        pad_len = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(x["attention_mask"] + [0] * pad_len)
        labels.append(x["label"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.float),
    }

def evaluate_pytorch(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].numpy()
            with torch.amp.autocast("cuda"):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                probs = torch.sigmoid(logits).cpu().numpy().ravel()
            all_preds.extend(probs)
            all_labels.extend(labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    binary_preds = (all_preds >= 0.50).astype(int)

    auc = roc_auc_score(all_labels, all_preds)
    f1 = f1_score(all_labels, binary_preds, average="macro")
    acc = accuracy_score(all_labels, binary_preds)
    fp = sum((all_labels == 0) & (binary_preds == 1))
    fn = sum((all_labels == 1) & (binary_preds == 0))

    return {
        "roc_auc": float(auc),
        "macro_f1": float(f1),
        "accuracy": float(acc),
        "fp": int(fp),
        "fn": int(fn),
    }

def main():
    set_status("initializing")
    log("=" * 65)
    log("DISTILBERT v5 RETRAINING PIPELINE STARTING")
    log("=" * 65)

    # 1. Hardware Check
    assert torch.cuda.is_available(), "FATAL: GPU acceleration is strictly required!"
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    log(f"GPU provisioned: {gpu_name}")

    # 2. Ingest Dataset
    set_status("data_ingestion")
    train_path = os.path.join(DATA_DIR, "train.csv")
    val_path = os.path.join(DATA_DIR, "val.csv")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    log(f"Train samples: {len(train_df)} (Toxic: {sum(train_df['label']==1)}, Benign: {sum(train_df['label']==0)})")
    log(f"Val samples  : {len(val_df)} (Toxic: {sum(val_df['label']==1)}, Benign: {sum(val_df['label']==0)})")

    # 3. Tokenizer & Model Initialization
    MODEL_NAME = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
    model.to(device)

    train_ds = ModerationTextDataset(train_df["text"].values, train_df["label"].values, tokenizer)
    val_ds = ModerationTextDataset(val_df["text"].values, val_df["label"].values, tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        num_workers=2,
        pin_memory=True,
    )

    # Seed for exact reproducibility
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # 4. Training Schedule: LR 2e-5 with Linear Warmup (first 10% steps)
    EPOCHS = 4
    MIN_EPOCHS = 3
    PATIENCE = 3
    LEARNING_RATE = 2e-5

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(0.10 * total_steps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    criterion = BinaryFocalLossWithLogits(alpha=0.55, gamma=1.5)
    scaler = torch.amp.GradScaler("cuda")

    log(f"Training parameters: Epochs={EPOCHS}, MinEpochs={MIN_EPOCHS}, LR={LEARNING_RATE}, WarmupSteps={warmup_steps}/{total_steps}")

    # 5. Training Loop with Best Checkpoint Tracking & Early Stopping on ROC-AUC
    best_auc = -1.0
    best_f1 = -1.0
    best_epoch = -1
    best_state_dict = None
    epochs_no_improve = 0

    set_status("training")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            if step % 100 == 0 or step == len(train_loader):
                log(f"Epoch {epoch}/{EPOCHS} | Batch {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

        # Validation
        eval_metrics = evaluate_pytorch(model, val_loader, device)
        val_auc = eval_metrics["roc_auc"]
        val_f1 = eval_metrics["macro_f1"]
        val_acc = eval_metrics["accuracy"]
        dt = (time.time() - t0) / 60.0

        log(f"---> Epoch {epoch} Eval ({dt:.1f}m): Val ROC-AUC: {val_auc:.4f} | Macro F1: {val_f1:.4f} | Acc: {val_acc*100:.2f}% | FP: {eval_metrics['fp']} | FN: {eval_metrics['fn']}")

        # Track improvement: higher AUC, or if equal, higher F1
        is_improved = (val_auc > best_auc) or (abs(val_auc - best_auc) < 1e-4 and val_f1 > best_f1)
        if is_improved:
            log(f"  [+] Best model updated! AUC: {best_auc:.4f} -> {val_auc:.4f}, F1: {best_f1:.4f} -> {val_f1:.4f}")
            best_auc = val_auc
            best_f1 = val_f1
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            # Save checkpoint
            torch.save(best_state_dict, os.path.join(OUT_DIR, "best_checkpoint.pt"))
        else:
            epochs_no_improve += 1
            log(f"  [-] No improvement for {epochs_no_improve} epoch(s).")
            if epoch >= MIN_EPOCHS and epochs_no_improve >= PATIENCE:
                log(f"  [EARLY STOPPING TRIGGERED] Epoch {epoch} terminating training.")
                break

    # 6. Restore Best Checkpoint
    log("=" * 65)
    log(f"Restoring best checkpoint from Epoch {best_epoch} (AUC: {best_auc:.4f}, F1: {best_f1:.4f})")
    log("=" * 65)
    model.load_state_dict(best_state_dict)
    model.to(device)
    model.eval()

    # 7. SECTION 5: Historical Battery Deploy Gate (Strict 24/24 Verification)
    set_status("verify_battery")
    log("=" * 65)
    log("SECTION 5: Held-out Historical Battery Verification (Strict Gate)")
    log("=" * 65)
    bat_texts = [t for t, _ in HISTORICAL_BATTERY]
    bat_expected = [exp for _, exp in HISTORICAL_BATTERY]

    pytorch_bat_probs = []
    bat_passed = 0
    with torch.no_grad():
        for text, exp in HISTORICAL_BATTERY:
            enc = tokenizer(text, return_tensors="pt")
            with torch.amp.autocast("cuda"):
                logits = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device)).logits
                prob = float(torch.sigmoid(logits).cpu().numpy().ravel()[0])
            pytorch_bat_probs.append(prob)
            ok = (prob >= 0.50) == (exp == 1)
            bat_passed += int(ok)
            log(f"  [{'PASS' if ok else 'FAIL'}] Score: {prob:.4f} | Exp: {exp} | {text}")

    log(f"Historical Battery Result: {bat_passed} / {len(HISTORICAL_BATTERY)} ({bat_passed/len(HISTORICAL_BATTERY)*100:.1f}%)")
    assert bat_passed == len(HISTORICAL_BATTERY), f"FATAL GATE FAILURE: PyTorch model failed Historical Battery ({bat_passed}/{len(HISTORICAL_BATTERY)})"

    # 8. Dynamic ONNX Export (<25ms Latency with dynamic sequence axis)
    set_status("onnx_export")
    log("=" * 65)
    log("SECTION 6: ONNX Export & Dynamic INT8 Quantization")
    log("=" * 65)

    class OnnxModelWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, input_ids, attention_mask):
            logits = self.m(input_ids=input_ids, attention_mask=attention_mask).logits
            return torch.sigmoid(logits)

    model_wrapper = OnnxModelWrapper(model).eval()
    dummy_input_ids = torch.ones((1, 16), dtype=torch.int64).to(device)
    dummy_mask = torch.ones((1, 16), dtype=torch.int64).to(device)

    fp32_onnx_path = os.path.join(OUT_DIR, "distilbert_moderation_v5.onnx")
    int8_onnx_path = os.path.join(OUT_DIR, "distilbert_moderation_v5_int8.onnx")

    dynamic_axes_config = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "score": {0: "batch"},
    }

    try:
        torch.onnx.export(
            model_wrapper,
            (dummy_input_ids, dummy_mask),
            fp32_onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["score"],
            dynamic_axes=dynamic_axes_config,
            opset_version=17,
            dynamo=False,
        )
    except Exception:
        torch.onnx.export(
            model_wrapper,
            (dummy_input_ids, dummy_mask),
            fp32_onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["score"],
            dynamic_axes=dynamic_axes_config,
            opset_version=14,
        )
    log(f"  FP32 ONNX Exported: {fp32_onnx_path} ({os.path.getsize(fp32_onnx_path)/1e6:.1f} MB)")

    # INT8 Dynamic Quantization
    quantize_dynamic(
        model_input=fp32_onnx_path,
        model_output=int8_onnx_path,
        weight_type=QuantType.QInt8,
    )
    int8_size_mb = os.path.getsize(int8_onnx_path) / 1e6
    log(f"  INT8 ONNX Quantized: {int8_onnx_path} ({int8_size_mb:.1f} MB)")

    # Save tokenizer artifacts
    tok_dir = os.path.join(OUT_DIR, "tokenizer")
    tokenizer.save_pretrained(tok_dir)
    tokenizer.save_pretrained(OUT_DIR)
    log(f"  Saved tokenizer artifacts to {tok_dir} and {OUT_DIR}")

    # 9. Quantization Precision Gate & CPU Latency
    set_status("quantization_sanity_gate")
    log("=" * 65)
    log("SECTION 7: Quantization Precision Gate & CPU Latency Check")
    log("=" * 65)
    int8_session = onnxruntime.InferenceSession(int8_onnx_path, providers=["CPUExecutionProvider"])

    onnx_bat_probs = []
    latencies = []
    onnx_battery_passed = 0

    for i, (text, exp) in enumerate(HISTORICAL_BATTERY):
        enc = tokenizer(text, return_tensors="np")
        t_start = time.perf_counter()
        sc_i8 = float(int8_session.run(["score"], {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        })[0].ravel()[0])
        dt = (time.perf_counter() - t_start) * 1000.0
        latencies.append(dt)
        onnx_bat_probs.append(sc_i8)

        ok = (sc_i8 >= 0.50) == (exp == 1)
        onnx_battery_passed += int(ok)

    # Calculate differences between PyTorch and Quantized INT8 ONNX
    diffs = [abs(pt - ox) for pt, ox in zip(pytorch_bat_probs, onnx_bat_probs)]
    max_diff = max(diffs)
    mean_lat = np.mean(latencies)

    log(f"  PyTorch vs INT8 Max Diff : {max_diff:.5f}")
    log(f"  INT8 Historical Battery  : {onnx_battery_passed} / {len(HISTORICAL_BATTERY)} ({onnx_battery_passed/len(HISTORICAL_BATTERY)*100:.1f}%)")
    log(f"  Mean CPU Latency per Run : {mean_lat:.2f} ms")

    # Strict Gates:
    assert max_diff < 0.20, f"FATAL GATE FAILURE: PyTorch vs INT8 Max Diff ({max_diff:.5f}) exceeded 0.20 limit!"
    assert onnx_battery_passed == len(HISTORICAL_BATTERY), f"FATAL GATE FAILURE: INT8 ONNX failed battery ({onnx_battery_passed}/{len(HISTORICAL_BATTERY)})"
    assert mean_lat < 25.0, f"FATAL GATE FAILURE: CPU Latency ({mean_lat:.2f} ms) exceeded 25 ms!"

    # Compute SHA256 of final artifact
    final_sha = hashlib.sha256(open(int8_onnx_path, "rb").read()).hexdigest()
    log(f"  Final distilbert_moderation_v5_int8.onnx SHA256: {final_sha}")

    set_status("done", "DONE")
    log("=" * 65)
    log("v5 RETRAINING PIPELINE COMPLETED SUCCESSFULLY WITH ALL GATES PASSED")
    log("=" * 65)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        log("!!!!! TRAINING FAILED !!!!!")
        log(tb)
        set_status("failed", "FAILED")
        sys.exit(1)
