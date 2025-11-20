#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================
FILE: 4_caseStudy1_gatedGRU_resnet_CMA_SA_proj.py

    DESCRIPTION:
        Enhanced version of the multimodal Gated GRU + ResNet architecture.
        This variant introduces THREE targeted structural upgrades designed to
        improve cross-modal alignment, representation quality, and fusion stability
        while keeping the original backbone intact.

    STRUCTURAL IMPROVEMENTS (with baseline reference):

        1) Cross-Modal Attention (CMA)
        ---------------------------------------------------------------
        • NEW: Text feature attends to image feature before fusion.
        • BASELINE: Text and image were only concatenated — no interaction occurred before gating or classification.
        • WHY: CMA lets the text embedding incorporate visual context, improving multimodal coherence and fake-news detection.

        2) GRU + Self-Attention Layer (Hybrid Text Tower)
        ---------------------------------------------------------------
        • NEW: Added a lightweight multi-head self-attention layer after the GRU.
        • BASELINE: Text tower relied solely on the GRU final hidden state.
        • WHY: GRU models sequential flow but lacks global reasoning. Self-attention adds long-range semantic understanding.

        3) Bottleneck Projection Layer for Text and Image (→ 256)
        ---------------------------------------------------------------
        • NEW: Both text and image features are projected into a shared 256-dimensional bottleneck space.
        • BASELINE: Text (GRU output) and image (ResNet features) had mismatched dimensional scales before fusion.
        • WHY: Normalizing both modalities improves fusion stability and helps the gating network operate on balanced feature representations.

    OVERALL IMPACT:
        These upgrades increase multimodal synergy, reduce modality imbalance,
        and improve classification robustness without requiring large architectural
        rewrites or additional data.

=========================================================================================
"""

##################################################################################
################################## Imports #######################################
##################################################################################
import os
import json
import math
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, models as tvm

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


##################################################################################
################################## Config ########################################
##################################################################################
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

# Dataset limits
TRAIN_CAP      = 50000
VALID_CAP      = 5000
TEST_CAP       = 5000

# Training
EPOCHS         = 10
BATCH_SIZE     = 64
LR             = 2e-3
WEIGHT_DECAY   = 0.0
NUM_WORKERS    = 8
SEED           = 42

# Task type
TASK = "3way"

if TASK == "2way":
    LABEL_COL      = "2_way_label"
    NUM_CLASSES    = 2
    TEXT_SUBDIR    = "text_proc_2way"
    MODEL_OUT_NAME = "caseStudy1_gated_gru_resnet_plus_cma_sa_proj_2way.pt"
    SUMMARY_OUT    = "caseStudy1_run_summary_plus_model_2way.json"

elif TASK == "3way":
    LABEL_COL      = "3_way_label"
    NUM_CLASSES    = 3
    TEXT_SUBDIR    = "text_proc_3way"
    MODEL_OUT_NAME = "caseStudy1_gated_gru_resnet_plus_cma_sa_proj_3way.pt"
    SUMMARY_OUT    = "caseStudy1_run_summary_plus_model_3way.json"

elif TASK == "6way":
    LABEL_COL      = "6_way_label"
    NUM_CLASSES    = 6
    TEXT_SUBDIR    = "text_proc_6way"
    MODEL_OUT_NAME = "caseStudy1_gated_gru_resnet_plus_cma_sa_proj_6way.pt"
    SUMMARY_OUT    = "caseStudy1_run_summary_plus_model_6way.json"

##################################################################################
################################## Paths #########################################
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

##################################################################################
############################ Model Hyperparameters ###############################
##################################################################################
EMB_DIM        = 200
GRU_HIDDEN     = 256
GRU_LAYERS     = 1
GRU_BIDIR      = True
TXT_DROPOUT    = 0.2

IMG_SIZE       = 224
IMG_PROJ_DIM   = 512
FREEZE_IMG     = True

# New bottleneck dim for both modalities
BOTTLENECK_DIM = 256

FUSION_HIDDEN  = 256

# Modality Dropout
P_DROP_TEXT    = 0.10
P_DROP_IMAGE   = 0.30


##################################################################################
################################ Dataset #########################################
##################################################################################
class FusionDataset(Dataset):
    def __init__(self, used_csv_path: Path, seq_npy_path: Path, data_root: Path, cap: int = -1):
        self.df = pd.read_csv(used_csv_path, dtype=str)
        self.seq = np.load(seq_npy_path)

        if cap >= 0 and cap < len(self.df):
            self.df  = self.df.iloc[:cap].copy()
            self.seq = self.seq[:cap]

        self.labels    = self.df[LABEL_COL].astype(int).to_numpy()
        self.has_img   = (self.df.get("hasImage", "FALSE") == "TRUE").to_numpy()
        self.img_paths = self.df.get("image_path", "").astype(str).to_numpy()
        self.data_root = data_root

        self.tf = T.Compose([
            T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(IMG_SIZE),
            T.ToTensor(),
            T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ])

    def __len__(self):
        return len(self.df)

    def _load_img(self, rel):
        if not rel: return None
        fp = self.data_root / rel
        if not fp.exists(): return None
        try:
            return self.tf(Image.open(fp).convert("RGBA").convert("RGB"))
        except:
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


##################################################################################
################################ New Blocks #######################################
##################################################################################

# ------------------------------------------------------------------------
# 1) GRU + SELF-ATTENTION (Hybrid Text Module)
# ------------------------------------------------------------------------
class TextTowerGRU_SA(nn.Module):
    """GRU encoder + small MultiHead Self-Attention layer."""
    def __init__(self, vocab_size):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, EMB_DIM, padding_idx=0)
        self.gru = nn.GRU(
            EMB_DIM, GRU_HIDDEN,
            num_layers=GRU_LAYERS,
            bidirectional=GRU_BIDIR,
            batch_first=True
        )

        self.out_dim = GRU_HIDDEN * (2 if GRU_BIDIR else 1)

        self.self_att = nn.MultiheadAttention(
            embed_dim=self.out_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )

        self.dropout = nn.Dropout(TXT_DROPOUT)

    def forward(self, seq):
        e = self.emb(seq)
        out, _ = self.gru(e)

        # Apply self-attention
        att_out, _ = self.self_att(out, out, out)

        feat = att_out[:, -1]      # last token representation
        return self.dropout(feat)


# ------------------------------------------------------------------------
# 2) Cross-Modal Attention (CMA)
# ------------------------------------------------------------------------
class CrossModalAttention(nn.Module):
    """
    Text attends to Image features to enhance text vector.
    """
    def __init__(self, txt_dim, img_dim, hidden=256):
        super().__init__()
        self.q = nn.Linear(txt_dim, hidden)
        self.k = nn.Linear(img_dim, hidden)
        self.v = nn.Linear(img_dim, hidden)
        self.out = nn.Linear(hidden, txt_dim)

    def forward(self, tfeat, ifeat):
        # B x DIM -> B x 1 x DIM
        q = self.q(tfeat).unsqueeze(1)
        k = self.k(ifeat).unsqueeze(1)
        v = self.v(ifeat).unsqueeze(1)

        att = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1)), dim=-1)
        cm = (att @ v).squeeze(1)
        return self.out(cm)


##################################################################################
################################ Main Model #######################################
##################################################################################
class ImageTowerResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
        feat_dim = m.fc.in_features
        self.backbone = nn.Sequential(*list(m.children())[:-1])

        if FREEZE_IMG:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.proj = nn.Linear(feat_dim, IMG_PROJ_DIM)

    def forward(self, x):
        f = self.backbone(x).flatten(1)
        return self.proj(f)


class GatedLateFusion(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()

        # Text & Image Towers
        self.txt = TextTowerGRU_SA(vocab_size)
        self.img = ImageTowerResNet18()

        # 3) New Bottleneck Projections
        self.txt_proj = nn.Linear(self.txt.out_dim, BOTTLENECK_DIM)
        self.img_proj = nn.Linear(IMG_PROJ_DIM, BOTTLENECK_DIM)

        # 1) Cross-Modal Attention
        self.cma = CrossModalAttention(BOTTLENECK_DIM, BOTTLENECK_DIM)

        # Classification heads
        self.text_head = nn.Sequential(
            nn.LayerNorm(BOTTLENECK_DIM),
            nn.Linear(BOTTLENECK_DIM, num_classes)
        )

        self.fuse_head = nn.Sequential(
            nn.Linear(BOTTLENECK_DIM * 2, FUSION_HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(FUSION_HIDDEN, num_classes),
        )

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(BOTTLENECK_DIM * 2 + 1, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, seq, image, has_img, p_drop_text=0.0, p_drop_image=0.0, train_mode=False):
        B = seq.size(0)
        device = seq.device

        # Modality dropout
        drop_text = (torch.rand(B, device=device) < p_drop_text) if train_mode else torch.zeros(B, dtype=torch.bool, device=device)
        drop_img  = (torch.rand(B, device=device) < p_drop_image) if train_mode else torch.zeros(B, dtype=torch.bool, device=device)
        drop_img  = drop_img | (has_img == 0)

        seq_masked = seq.clone()
        seq_masked[drop_text] = 0
        tfeat = self.txt(seq_masked)

        image = torch.where(drop_img.view(B,1,1,1), torch.zeros_like(image), image)
        ifeat = self.img(image)

        # Bottleneck projections
        tfeat = self.txt_proj(tfeat)
        ifeat = self.img_proj(ifeat)

        # Cross-modal attention
        cm = self.cma(tfeat, ifeat)
        tfeat = tfeat + cm  # residual enhancement

        # Heads
        text_logits  = self.text_head(tfeat)
        fused_input  = torch.cat([tfeat, ifeat], dim=1)
        fused_logits = self.fuse_head(fused_input)

        gate_in = torch.cat([tfeat, ifeat, has_img.float().unsqueeze(1)], dim=1)
        g = self.gate(gate_in)

        logits = g * text_logits + (1 - g) * fused_logits
        return logits, g.squeeze(1)


##################################################################################
################################ Training #########################################
##################################################################################
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, optimizer, criterion, train=False):
    model.train(train)
    total_loss = 0.0
    ys, ps = [], []

    for batch in loader:
        seq = batch["seq"].to(DEVICE)
        img = batch["image"].to(DEVICE)
        has = batch["has_img"].to(DEVICE)
        y   = batch["label"].to(DEVICE)

        with torch.set_grad_enabled(train):
            logits, _ = model(seq, img, has,
                               p_drop_text=P_DROP_TEXT,
                               p_drop_image=P_DROP_IMAGE,
                               train_mode=train)
            loss = criterion(logits, y)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.size(0)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        ys.extend(y.cpu().numpy().tolist())
        ps.extend(pred.tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(ys, ps)
    f1  = f1_score(ys, ps, average="macro")
    prec = precision_score(ys, ps, average="macro", zero_division=0)
    rec  = recall_score(ys, ps, average="macro", zero_division=0)

    return avg_loss, acc, prec, rec, f1


##################################################################################
################################### Main #########################################
##################################################################################
def main():
    set_seed(SEED)

    base = Path(DATA_ROOT) / PREPARED_DIR / TEXT_SUBDIR
    meta = json.loads((base / META_JSON).read_text())
    vocab_size = meta["actual_vocab_size"] + 1

    tr = FusionDataset(base / TRAIN_USED_CSV, base / TRAIN_SEQ_NPY, Path(DATA_ROOT), cap=TRAIN_CAP)
    va = FusionDataset(base / VALID_USED_CSV, base / VALID_SEQ_NPY, Path(DATA_ROOT), cap=VALID_CAP)
    te = FusionDataset(base / TEST_USED_CSV,  base / TEST_SEQ_NPY,  Path(DATA_ROOT), cap=TEST_CAP)

    dl_tr = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    dl_va = DataLoader(va, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    dl_te = DataLoader(te, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    print(f"Dataset sizes -> train={len(tr)}, valid={len(va)}, test={len(te)}")

    model = GatedLateFusion(vocab_size=vocab_size, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_f1 = -1
    best_state = None

    for ep in range(1, EPOCHS+1):
        tr_loss, tr_acc, tr_prec, tr_rec, tr_f1 = run_epoch(
            model, dl_tr, optimizer, criterion, train=True
        )
        va_loss, va_acc, va_prec, va_rec, va_f1 = run_epoch(
            model, dl_va, optimizer, criterion, train=False
        )

        print(
            f"[Epoch {ep:02d}] "
            f"Train: loss={tr_loss:.4f} acc={tr_acc:.4f} prec={tr_prec:.4f} rec={tr_rec:.4f} f1={tr_f1:.4f} | "
            f"Valid: loss={va_loss:.4f} acc={va_acc:.4f} prec={va_prec:.4f} rec={va_rec:.4f} f1={va_f1:.4f}"
        )
        # ↑ ONLY CHANGE: added prec + rec for TRAIN + VAL

        if va_f1 > best_f1:
            best_f1 = va_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    # TEST METRICS (ALL 5)
    te_loss, te_acc, te_prec, te_rec, te_f1 = run_epoch(model, dl_te, optimizer, criterion, train=False)

    print(
        f"[TEST] loss={te_loss:.4f} acc={te_acc:.4f} prec={te_prec:.4f} rec={te_rec:.4f} f1={te_f1:.4f}"
    )
    # ↑ ONLY CHANGE: added precision + recall

    out_dir = Path("./training_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_dir / MODEL_OUT_NAME)

    summary = {
        "sizes": {"train": len(tr), "valid": len(va), "test": len(te)},
        "metrics": {
            "test_loss": te_loss,
            "test_acc": te_acc,
            "test_precision": te_prec,
            "test_recall": te_rec,
            "test_f1": te_f1
        },
        "notes": [
            "This model includes 3 improvements:",
            "1) Cross-Modal Attention (CMA)",
            "2) GRU + Self-Attention",
            "3) Bottleneck projection for text/image"
        ]
    }

    (out_dir / SUMMARY_OUT).write_text(json.dumps(summary, indent=2))

    print(f"Saved model + summary to: {out_dir}")


if __name__ == "__main__":
    main()