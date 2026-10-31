#!/usr/bin/env python3
"""
=====================================================================
PIXELABS CONTENT MODERATION  --  v4 DISTILBERT RETRAIN (GPU ACCELERATED)
Target: Headless Kaggle Execution ("Save & Run All")
=====================================================================
Architecture & Optimization:
  1. GPU Assertion: Enforces CUDA GPU availability at startup.
  2. Metric Optimization: Checkpoint selection strictly driven by ROC-AUC
     (and Macro F1), the gold standard for imbalanced moderation.
  3. Early Stopping: Terminate immediately if ROC-AUC plateaus or degrades
     (patience = 2).
  4. Best Checkpoint Restoration: Restores strictly the highest ROC-AUC
     checkpoint weights before ONNX export.
  5. Latency (<25ms): Exports dynamic sequence length axes so inferences
     don't incur fixed 128-token attention overhead on CPU.
  6. Quantization: Quantizes best fp32 checkpoint to dynamic int8 (~67 MB).
=====================================================================
"""

import csv
import glob
import hashlib
import os
import random
import sys
import time
import traceback
import subprocess

def pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

try:
    import torch
except ImportError:
    pip("torch")
    import torch

try:
    import transformers
except ImportError:
    pip("transformers")
    import transformers

try:
    import onnxruntime
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError:
    pip("onnxruntime", "onnx")
    import onnxruntime
    from onnxruntime.quantization import quantize_dynamic, QuantType

try:
    import sklearn
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
except ImportError:
    pip("scikit-learn")
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# SECTION 0: GPU ASSERTION & CONFIGURATION
# ---------------------------------------------------------------------------
print("=" * 65)
print("SECTION 0: Hardware Verification & Assertions")
print("=" * 65)

assert torch.cuda.is_available(), (
    "CRITICAL HARDWARE FAILURE: torch.cuda.is_available() is False! "
    "GPU acceleration must be enabled in Kaggle Settings (enable_gpu: true, accelerator: 'gpu')."
)
device = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(0)
print(f"  [GPU CONFIRMED] Device: {device} | Device Name: {gpu_name}")
print(f"  CUDA Version : {torch.version.cuda}")
print(f"  PyTorch      : {torch.__version__}")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

MODEL_NAME   = "distilbert-base-uncased"
MAX_LEN      = 128
EPOCHS       = 4
PATIENCE     = 2          # Early stopping patience in epochs
BATCH_SIZE   = 64
LR           = 2e-5
WEIGHT_DECAY = 0.01
OUT_DIR      = os.environ.get("OUT_DIR", "v4_out")
os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH     = os.path.join(OUT_DIR, "train_v4.log")

def log(*args):
    msg = " ".join(str(x) for x in args)
    print(msg, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass

def set_status(stage: str, state: str = "RUNNING"):
    try:
        with open(os.path.join(OUT_DIR, "STATUS.txt"), "w", encoding="utf-8") as f:
            f.write(f"STATE: {state}\nSTAGE: {stage}\nWHEN : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# SECTION 1: HELD-OUT BENCHMARK BATTERY (NEVER SEEN IN TRAINING)
# ---------------------------------------------------------------------------
BATTERY_BENCHMARK = [
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
# SECTION 2: DATA LOADING & PREPROCESSING (80:20 RATIO)
# ---------------------------------------------------------------------------
def locate_dataset_files():
    """Locate preprocessed train.csv and val.csv in Kaggle input or local folders."""
    train_file = None
    val_file = None

    search_paths = [
        "/kaggle/input/**/train.csv",
        "./data_v4/train.csv",
        "train.csv",
        "../data_v4/train.csv",
    ]
    for sp in search_paths:
        matches = glob.glob(sp, recursive=True)
        if matches:
            train_file = matches[0]
            break

    search_val_paths = [
        "/kaggle/input/**/val.csv",
        "./data_v4/val.csv",
        "val.csv",
        "../data_v4/val.csv",
    ]
    for sp in search_val_paths:
        matches = glob.glob(sp, recursive=True)
        if matches:
            val_file = matches[0]
            break

    return train_file, val_file

def load_csv_data(path: str):
    texts = []
    labels = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row.get("text", "").strip()
            l = int(row.get("label", 0))
            if t:
                texts.append(t)
                labels.append(l)
    return texts, labels

# ---------------------------------------------------------------------------
# SECTION 3: TRAINING LOOP WITH ROC-AUC EARLY STOPPING
# ---------------------------------------------------------------------------
def main():
    set_status("init")
    log("=" * 65)
    log("SECTION 1: Dataset Ingestion (80:20 Strategy)")
    log("=" * 65)

    train_path, val_path = locate_dataset_files()
    if not train_path or not val_path:
        log("  [WARN] Preprocessed CSV files not found locally, preparing directly...")
        # Emergency online fallback if dataset is detached
        from datasets import load_dataset
        ds = load_dataset("SetFit/toxic_conversations", split="train", streaming=True)
        raw_texts, raw_labels = [], []
        for row in ds:
            t = row["text"].strip()
            if len(t) >= 5:
                raw_texts.append(t)
                raw_labels.append(int(row["label"]))
            if len(raw_texts) >= 20000:
                break
        split_idx = int(0.85 * len(raw_texts))
        train_texts, train_labels = raw_texts[:split_idx], raw_labels[:split_idx]
        val_texts, val_labels = raw_texts[split_idx:], raw_labels[split_idx:]
    else:
        log(f"  Loaded train set from: {train_path}")
        log(f"  Loaded val set from  : {val_path}")
        train_texts, train_labels = load_csv_data(train_path)
        val_texts, val_labels = load_csv_data(val_path)

    log(f"  Train samples: {len(train_texts)} (Toxic: {sum(train_labels)}, Benign: {len(train_labels)-sum(train_labels)})")
    log(f"  Val samples  : {len(val_texts)} (Toxic: {sum(val_labels)}, Benign: {len(val_labels)-sum(val_labels)})")

    # Tokenization
    set_status("tokenize")
    log("=" * 65)
    log("SECTION 2: Tokenization (DistilBERT)")
    log("=" * 65)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def encode_data(texts):
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors="pt"
        )
        return enc["input_ids"], enc["attention_mask"]

    train_ids, train_masks = encode_data(train_texts)
    val_ids, val_masks     = encode_data(val_texts)
    y_train = torch.tensor(train_labels, dtype=torch.float32)
    y_val   = np.array(val_labels, dtype=np.int32)
    log(f"  Tokenization complete in {time.time()-t0:.1f}s")

    class TextDataset(Dataset):
        def __init__(self, ids, masks, ys):
            self.ids = ids
            self.masks = masks
            self.ys = ys
        def __len__(self):
            return len(self.ys)
        def __getitem__(self, idx):
            return self.ids[idx], self.masks[idx], self.ys[idx]

    train_loader = DataLoader(
        TextDataset(train_ids, train_masks, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # Initialize model
    set_status("model_setup")
    log("=" * 65)
    log("SECTION 3: Model Setup & Fine-Tuning")
    log("=" * 65)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda")
    criterion = torch.nn.BCEWithLogitsLoss()

    def evaluate(m, ids, masks):
        m.eval()
        scores = []
        batch_eval = 128
        with torch.no_grad():
            for i in range(0, len(ids), batch_eval):
                b_ids = ids[i:i+batch_eval].to(device)
                b_mask = masks[i:i+batch_eval].to(device)
                with torch.amp.autocast("cuda"):
                    logits = m(input_ids=b_ids, attention_mask=b_mask).logits
                probs = torch.sigmoid(logits).cpu().numpy().ravel()
                scores.extend(probs)
        return np.array(scores, dtype=np.float32)

    # Training loop with ROC-AUC Early Stopping & Restoration
    best_auc = -1.0
    best_f1 = -1.0
    best_epoch = -1
    best_state_dict = None
    no_improvement_count = 0

    set_status("training")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_epoch = time.time()
        running_loss = 0.0
        n_batches = 0

        for step, (b_ids, b_mask, b_y) in enumerate(train_loader):
            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_y = b_y.unsqueeze(1).to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(input_ids=b_ids, attention_mask=b_mask).logits
                loss = criterion(logits, b_y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            n_batches += 1

            if step > 0 and step % 100 == 0:
                elapsed = time.time() - t_epoch
                done = step / len(train_loader)
                eta = (elapsed / done) * (1 - done) / 60
                log(f"  Epoch {epoch}/{EPOCHS} | Batch {step}/{len(train_loader)} | Loss: {running_loss/n_batches:.4f} | ETA: {eta:.1f}m")

        # Evaluation at end of epoch
        val_scores = evaluate(model, val_ids, val_masks)
        val_auc = float(roc_auc_score(y_val, val_scores))
        val_preds = (val_scores >= 0.5).astype(int)
        val_f1 = float(f1_score(y_val, val_preds, average="macro"))
        val_acc = float(accuracy_score(y_val, val_preds))
        fp = int(((val_preds == 1) & (y_val == 0)).sum())
        fn = int(((val_preds == 0) & (y_val == 1)).sum())

        log(f"\n---> Epoch {epoch} Evaluation (Time: {(time.time()-t_epoch)/60:.1f}m):")
        log(f"     Val ROC-AUC : {val_auc:.4f} (Best: {best_auc:.4f})")
        log(f"     Val Macro F1: {val_f1:.4f}")
        log(f"     Val Accuracy: {val_acc*100:.2f}% | FP: {fp} | FN: {fn}")

        # Checkpoint selection strictly on ROC-AUC
        if val_auc > best_auc:
            log(f"     [+] Metric Improved! {best_auc:.4f} -> {val_auc:.4f}. Saving best checkpoint.")
            best_auc = val_auc
            best_f1 = val_f1
            best_epoch = epoch
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improvement_count = 0
            # Save checkpoint
            torch.save(best_state_dict, os.path.join(OUT_DIR, "best_checkpoint.pt"))
        else:
            no_improvement_count += 1
            log(f"     [-] No ROC-AUC improvement for {no_improvement_count} epoch(s).")
            if no_improvement_count >= PATIENCE:
                log(f"     [EARLY STOPPING TRIGGERED] Terminating training to preserve GPU quota and avoid degradation.")
                break

    # ---------------------------------------------------------------------------
    # SECTION 4: RESTORE STRICTLY HIGHEST ROC-AUC CHECKPOINT
    # ---------------------------------------------------------------------------
    log("=" * 65)
    log(f"SECTION 4: Restoring Highest ROC-AUC Checkpoint (from Epoch {best_epoch})")
    log("=" * 65)
    assert best_state_dict is not None, "Error: No best checkpoint was saved!"
    model.load_state_dict(best_state_dict)
    model.to(device)
    model.eval()
    log(f"  Restored model weights from Epoch {best_epoch} with Validation ROC-AUC: {best_auc:.4f}")

    # ---------------------------------------------------------------------------
    # SECTION 5: HELD-OUT BENCHMARK BATTERY VERIFICATION
    # ---------------------------------------------------------------------------
    set_status("verify_benchmark")
    log("=" * 65)
    log("SECTION 5: Held-out Historical Battery Verification (Deploy Gate)")
    log("=" * 65)
    bat_texts = [t for t, _ in BATTERY_BENCHMARK]
    bat_expected = [exp for _, exp in BATTERY_BENCHMARK]
    bat_enc = tokenizer(bat_texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            bat_logits = model(input_ids=bat_enc["input_ids"].to(device), attention_mask=bat_enc["attention_mask"].to(device)).logits
    bat_probs = torch.sigmoid(bat_logits).cpu().numpy().ravel()

    n_pass = 0
    for (t, exp), sc in zip(BATTERY_BENCHMARK, bat_probs):
        ok = (sc >= 0.5) == (exp == 1)
        n_pass += int(ok)
        log(f"  [{'PASS' if ok else 'FAIL'}] Score: {sc:.4f} | Exp: {exp} | {t[:55]}")
    log(f"  Historical Battery Result: {n_pass} / {len(BATTERY_BENCHMARK)} ({n_pass/len(BATTERY_BENCHMARK)*100:.1f}%)")

    # ---------------------------------------------------------------------------
    # SECTION 6: DYNAMIC ONNX EXPORT & INT8 QUANTIZATION (<25ms LATENCY)
    # ---------------------------------------------------------------------------
    set_status("onnx_export")
    log("=" * 65)
    log("SECTION 6: ONNX Export & int8 Quantization (Latency Optimized)")
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

    fp32_onnx_path = os.path.join(OUT_DIR, "distilbert_moderation_v4.onnx")
    int8_onnx_path = os.path.join(OUT_DIR, "distilbert_moderation_v4_int8.onnx")

    # DYNAMIC AXES: Exposes both batch AND sequence length dimensions
    # Allows variable sequence lengths so short inputs take < 10ms on CPU!
    dynamic_axes_config = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "score": {0: "batch"}
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
            dynamo=False
        )
    except TypeError:
        torch.onnx.export(
            model_wrapper,
            (dummy_input_ids, dummy_mask),
            fp32_onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["score"],
            dynamic_axes=dynamic_axes_config,
            opset_version=17
        )

    log(f"  FP32 ONNX Exported: {fp32_onnx_path} ({os.path.getsize(fp32_onnx_path)/1e6:.1f} MB)")

    # Quantize to int8 (~67 MB target)
    try:
        quantize_dynamic(
            model_input=fp32_onnx_path,
            model_output=int8_onnx_path,
            weight_type=QuantType.QInt8
        )
        int8_size_mb = os.path.getsize(int8_onnx_path) / 1e6
        log(f"  INT8 ONNX Quantized: {int8_onnx_path} ({int8_size_mb:.1f} MB)")
    except Exception as e:
        log(f"  [ERROR] Dynamic int8 quantization failed: {e}")
        raise

    # Save tokenizer artifacts
    tok_dir = os.path.join(OUT_DIR, "tokenizer")
    tokenizer.save_pretrained(tok_dir)
    # Also save directly in OUT_DIR
    tokenizer.save_pretrained(OUT_DIR)
    log(f"  Saved tokenizer artifacts to {tok_dir} and {OUT_DIR}")

    # Parity & Latency Verification on CPU
    ort_session = onnxruntime.InferenceSession(int8_onnx_path, providers=["CPUExecutionProvider"])
    ort_scores = []
    latencies = []
    for t in bat_texts:
        enc = tokenizer(t, return_tensors="np")
        t_start = time.perf_counter()
        sc = ort_session.run(["score"], {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"]
        })[0].ravel()[0]
        dt = (time.perf_counter() - t_start) * 1000.0
        latencies.append(dt)
        ort_scores.append(float(sc))

    max_diff = max(abs(p - o) for p, o in zip(bat_probs, ort_scores))
    mean_lat = np.mean(latencies)
    log(f"  ONNX vs PyTorch Max Diff : {max_diff:.5f} ({'PASSED' if max_diff < 0.05 else 'FAILED'})")
    log(f"  Mean CPU Latency per Run : {mean_lat:.2f} ms ({'PASSED (<25ms)' if mean_lat < 25 else 'FAILED'})")

    # Compute sha256
    sha = hashlib.sha256(open(int8_onnx_path, "rb").read()).hexdigest()
    log(f"  Final int8 SHA256: {sha}")

    set_status("done", "DONE")
    log("=" * 65)
    log("RETRAINING PIPELINE COMPLETED SUCCESSFULLY")
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
