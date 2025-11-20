#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=========================================================================================
FILE: 3_train_gated_moddrop.py

DESCRIPTION:
    Full training pipeline for the multimodal **Gated GRU + ResNet** architecture
    using the **Modality Dropout** strategy.

    It loads preprocessed text sequences + image paths, constructs the
    multimodal fusion model, trains with validation monitoring, evaluates on the
    held-out test set, and saves both weights and a JSON summary report.

KEY FEATURES (aligned with case-study framework):

    1) Multimodal Gated Fusion (Baseline Architecture)
    ---------------------------------------------------------------
    • Uses a GRU-based text tower and a ResNet18-based image tower.
    • A learned gating network blends TEXT-ONLY logits and FUSED logits.
    • This helps the model rely on text when images are missing or unreliable.

    2) Modality Dropout (ModDrop)
    ---------------------------------------------------------------
    • Randomly drops text or image features during TRAINING only.
    • Improves robustness to incomplete or noisy multimodal inputs.
    • Matches the same philosophy as your case-study experiments for
      robustness-focused feature learning.

    3) Preprocessed Dataset Compatibility
    ---------------------------------------------------------------
    • Uses the exact `train_used.csv`, `valid_used.csv`, `test_used.csv`
      and `*_seq.npy` files produced earlier.
    • Zero extra preprocessing is performed here — consistent with your
      case-study workflow.

    4) Gated Late Fusion with Shared Classifier Heads
    ---------------------------------------------------------------
    • Text-only head for unimodal inference.
    • Fusion head for multimodal inference.
    • Sigmoid gate selects how much to trust each.

    5) Full Metric Reporting (5 metrics)
    ---------------------------------------------------------------
    • loss
    • accuracy
    • precision (macro)
    • recall (macro)
    • F1-score (macro)

=========================================================================================
"""

##################################################################################
############### Imports
##################################################################################

import os
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, models as tvm

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

PIN_MEMORY = True if DEVICE == "cuda" else False

##################################################################################
############### TASK CONFIG — CHANGE ONLY THIS PART
##################################################################################

TRAIN_CAP      = 50000
VALID_CAP      = 5000
TEST_CAP       = 5000

EPOCHS         = 10
BATCH_SIZE     = 64
LR             = 2e-3
WEIGHT_DECAY   = 0.0
NUM_WORKERS    = 8
SEED           = 42
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

TASK = "6way"

if TASK == "2way":
    LABEL_COL      = "2_way_label"
    NUM_CLASSES    = 2
    TEXT_SUBDIR    = "text_proc_2way"
    MODEL_OUT_NAME = "gated_gru_resnet_best_2way.pt"
    SUMMARY_OUT    = "run_summary_gated_gru_2way.json"

elif TASK == "3way":
    LABEL_COL      = "3_way_label"
    NUM_CLASSES    = 3
    TEXT_SUBDIR    = "text_proc_3way"
    MODEL_OUT_NAME = "gated_gru_resnet_best_3way.pt"
    SUMMARY_OUT    = "run_summary_gated_gru_3way.json"

elif TASK == "6way":
    LABEL_COL      = "6_way_label"
    NUM_CLASSES    = 6
    TEXT_SUBDIR    = "text_proc_6way"
    MODEL_OUT_NAME = "gated_gru_resnet_best_6way.pt"
    SUMMARY_OUT    = "run_summary_gated_gru_6way.json"

else:
    raise ValueError("TASK must be '2way', '3way', or '6way'.")

##################################################################################
########################## Config
##################################################################################

DATA_ROOT      = "../project_dataset/fakeddit_data/multimodal_only_samples"
PREPARED_DIR   = "prepared_data"

TRAIN_USED_CSV = "train_used.csv"
VALID_USED_CSV = "valid_used.csv"
TEST_USED_CSV  = "test_used.csv"
TRAIN_SEQ_NPY  = "train_seq.npy"
VALID_SEQ_NPY  = "valid_seq.npy"
TEST_SEQ_NPY   = "test_seq.npy"
META_JSON      = "meta.json"

# Model sizes
EMB_DIM        = 200
GRU_HIDDEN     = 256
GRU_LAYERS     = 1
GRU_BIDIR      = True
TXT_DROPOUT    = 0.2

IMG_SIZE       = 224
IMG_PROJ_DIM   = 512
FREEZE_IMG     = True

FUSION_HIDDEN  = 256

P_DROP_TEXT    = 0.10
P_DROP_IMAGE   = 0.30

#########################################
########################## Data
#########################################

class FusionDataset(Dataset):
    def __init__(self, used_csv_path: Path, seq_npy_path: Path, data_root: Path, cap: int = -1):
        self.df = pd.read_csv(used_csv_path, dtype=str)
        self.seq = np.load(seq_npy_path)
        if cap is not None and cap >= 0 and cap < len(self.df):
            self.df = self.df.iloc[:cap].copy()
            self.seq = self.seq[:cap]

        self.labels = self.df[LABEL_COL].astype(int).to_numpy()

        self.has_img = (self.df.get("hasImage", "FALSE") == "TRUE").to_numpy()
        self.img_paths = self.df.get("image_path", "").astype(str).to_numpy()
        self.data_root = data_root

        self.tf = T.Compose([
            T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(IMG_SIZE),
            T.ToTensor(),
            T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
        ])

    def __len__(self): return len(self.df)

    def _load_img(self, rel: str):
        if not rel:
            return None
        fp = self.data_root / rel
        if not fp.exists():
            return None
        try:
            return self.tf(Image.open(fp).convert("RGBA").convert("RGB"))
        except Exception:
            return None

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        seq = torch.tensor(self.seq[i], dtype=torch.long)
        y   = torch.tensor(self.labels[i], dtype=torch.long)
        has = int(self.has_img[i])

        img = self._load_img(self.img_paths[i]) if has else None
        if img is None:
            img = torch.zeros(3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
            has = 0

        return {"seq": seq, "image": img, "has_img": torch.tensor(has), "label": y}

#########################################
########################## Models
#########################################

class TextTowerGRU(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden: int, layers: int, bidir: bool, dropout: float):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(
            emb_dim, hidden, num_layers=layers, batch_first=True,
            bidirectional=bidir, dropout=0.0 if layers == 1 else dropout
        )
        out_dim = hidden * (2 if bidir else 1)
        self.out_dim = out_dim
        self.post = nn.Dropout(dropout)

    def forward(self, x):
        e = self.emb(x)
        out, h = self.gru(e)
        feat = out[:, -1, :]
        return self.post(feat)

class ImageTowerResNet18(nn.Module):
    def __init__(self, out_dim: int, freeze_backbone: bool = True):
        super().__init__()
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
        feat_dim = m.fc.in_features
        self.backbone = nn.Sequential(*list(m.children())[:-1])

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.proj = nn.Linear(feat_dim, out_dim)

    def forward(self, x):
        f = self.backbone(x).flatten(1)
        return self.proj(f)

class GatedLateFusion(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int):
        super().__init__()
        self.txt = TextTowerGRU(vocab_size, EMB_DIM, GRU_HIDDEN, GRU_LAYERS, GRU_BIDIR, TXT_DROPOUT)
        self.img = ImageTowerResNet18(IMG_PROJ_DIM, freeze_backbone=FREEZE_IMG)

        self.text_head = nn.Sequential(
            nn.LayerNorm(self.txt.out_dim),
            nn.Linear(self.txt.out_dim, num_classes),
        )

        self.fuse_head = nn.Sequential(
            nn.Linear(self.txt.out_dim + IMG_PROJ_DIM, FUSION_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(FUSION_HIDDEN, num_classes),
        )

        self.gate = nn.Sequential(
            nn.Linear(self.txt.out_dim + IMG_PROJ_DIM + 1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, seq, image, has_img, p_drop_text=0.0, p_drop_image=0.0, train_mode=False):
        B = seq.size(0)
        device = seq.device

        drop_text = torch.zeros(B, dtype=torch.bool, device=device)
        drop_img  = torch.zeros(B, dtype=torch.bool, device=device)

        if train_mode and p_drop_text > 0:
            drop_text = (torch.rand(B, device=device) < p_drop_text)
        if train_mode and p_drop_image > 0:
            drop_img = (torch.rand(B, device=device) < p_drop_image)

        drop_img = torch.logical_or(drop_img, has_img == 0)

        seq_masked = seq.clone()
        seq_masked[drop_text] = 0
        tfeat = self.txt(seq_masked)

        image = torch.where(drop_img.view(B,1,1,1), torch.zeros_like(image), image)
        ifeat = self.img(image)

        text_logits  = self.text_head(tfeat)
        fused_logits = self.fuse_head(torch.cat([tfeat, ifeat], dim=1))
        g = self.gate(torch.cat([tfeat, ifeat, has_img.float().unsqueeze(1)], dim=1))

        logits = g * text_logits + (1 - g) * fused_logits
        return logits, g.squeeze(1)

#########################################
########################## Train / Eval
#########################################

def set_seed(seed:int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_epoch(model, loader, optimizer, criterion, train: bool):
    """
    Returns:
        avg_loss, acc, prec, rec, f1
    """
    model.train(train)
    total_loss = 0.0
    ys, ps = [], []

    for batch in loader:
        seq = batch["seq"].to(DEVICE)
        img = batch["image"].to(DEVICE)
        has = batch["has_img"].to(DEVICE)
        y   = batch["label"].to(DEVICE)

        with torch.set_grad_enabled(train):
            logits, gate = model(seq, img, has, P_DROP_TEXT, P_DROP_IMAGE, train_mode=train)
            loss = criterion(logits, y)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.size(0)

        pred = torch.argmax(logits, 1).detach().cpu().numpy()
        ys.extend(y.detach().cpu().numpy().tolist())
        ps.extend(pred.tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc  = accuracy_score(ys, ps)
    prec = precision_score(ys, ps, average="macro", zero_division=0)
    rec  = recall_score(ys, ps, average="macro", zero_division=0)
    f1   = f1_score(ys, ps, average="macro", zero_division=0)

    return avg_loss, acc, prec, rec, f1

#########################################
########################## Main
#########################################

def main():
    set_seed(SEED)

    base = Path(DATA_ROOT) / PREPARED_DIR / TEXT_SUBDIR
    meta = json.loads((base / META_JSON).read_text())
    vocab_size = meta["actual_vocab_size"] + 1

    tr = FusionDataset(base / TRAIN_USED_CSV, base / TRAIN_SEQ_NPY, Path(DATA_ROOT), cap=TRAIN_CAP)
    va = FusionDataset(base / VALID_USED_CSV, base / VALID_SEQ_NPY, Path(DATA_ROOT), cap=VALID_CAP)
    te = FusionDataset(base / TEST_USED_CSV,  base / TEST_SEQ_NPY,  Path(DATA_ROOT), cap=TEST_CAP)

    dl_tr = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    dl_va = DataLoader(va, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    dl_te = DataLoader(te, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    print(f"Sizes -> train {len(tr)} | valid {len(va)} | test {len(te)}")

    model = GatedLateFusion(vocab_size=vocab_size, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_f1, best_state = -1.0, None

    for ep in range(1, EPOCHS+1):
        tr_loss, tr_acc, tr_prec, tr_rec, tr_f1 = run_epoch(model, dl_tr, optimizer, criterion, train=True)
        va_loss, va_acc, va_prec, va_rec, va_f1 = run_epoch(model, dl_va, optimizer, criterion, train=False)

        print(
            f"[Epoch {ep:02d}] "
            f"train: loss {tr_loss:.4f} acc {tr_acc:.4f} prec {tr_prec:.4f} rec {tr_rec:.4f} f1 {tr_f1:.4f} | "
            f"valid: loss {va_loss:.4f} acc {va_acc:.4f} prec {va_prec:.4f} rec {va_rec:.4f} f1 {va_f1:.4f}"
        )

        if va_f1 > best_f1:
            best_f1, best_state = va_f1, {k:v.cpu() for k,v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # ===========================
    # TEST METRICS (ALL 5)
    # ===========================
    te_loss, te_acc, te_prec, te_rec, te_f1 = run_epoch(
        model, dl_te, optimizer, criterion, train=False
    )

    print(
        f"[TEST] loss {te_loss:.4f} | "
        f"acc {te_acc:.4f} | "
        f"prec {te_prec:.4f} | "
        f"rec {te_rec:.4f} | "
        f"f1 {te_f1:.4f}"
    )

    out_dir = Path("./training_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / MODEL_OUT_NAME)

    summary = {
        "sizes": {"train": len(tr), "valid": len(va), "test": len(te)},
        "caps": {"train_cap": TRAIN_CAP, "valid_cap": VALID_CAP, "test_cap": TEST_CAP},
        "hyperparams": {
            "emb_dim": EMB_DIM,
            "gru_hidden": GRU_HIDDEN,
            "bidir": GRU_BIDIR,
            "img_proj_dim": IMG_PROJ_DIM,
            "fusion_hidden": FUSION_HIDDEN,
            "p_drop_text": P_DROP_TEXT,
            "p_drop_image": P_DROP_IMAGE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR
        },
        "metrics": {
            "test_loss": te_loss,
            "test_acc": te_acc,
            "test_precision": te_prec,
            "test_recall": te_rec,
            "test_f1_macro": te_f1
        }
    }

    (out_dir / SUMMARY_OUT).write_text(json.dumps(summary, indent=2))
    print(f"[saved] weights + summary to {out_dir}")

if __name__ == "__main__":
    main()