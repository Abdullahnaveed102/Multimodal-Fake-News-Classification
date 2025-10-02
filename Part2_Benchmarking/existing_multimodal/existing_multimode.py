#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multimodal CNN (text + images) — cleaned & updated
- Fixes torch AMP deprecations (uses torch.amp.autocast / torch.amp.GradScaler)
- Handles sklearn UndefinedMetricWarning via zero_division=0
- Deduplicates Dataset definition (one collate-safe class)
- NLTK downloads are guarded + quiet
- Optional class weights computed from actual train batches
- Small perf tweaks (cudnn.benchmark, pin_memory, num_workers configurable)
"""

# =========================================
# Imports
# =========================================
import os, re, time, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Union, Tuple

# Images
import cv2

# Text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Torch
import torch
from torch import nn, optim, amp
from torch.utils.data import Dataset, DataLoader

# Perf hint for CNNs
torch.backends.cudnn.benchmark = True

print("==> Imports loaded")

# =========================================
# NLTK: ensure resources (quiet, one-time)
# =========================================
def _ensure(pkg_path: str, key: str):
    try:
        nltk.data.find(pkg_path)
    except LookupError:
        nltk.download(key, quiet=True)

_ensure("tokenizers/punkt", "punkt")
_ensure("corpora/stopwords", "stopwords")
_ensure("corpora/wordnet", "wordnet")
_ensure("corpora/omw-1.4", "omw-1.4")

_STOPWORDS = set(stopwords.words("english"))
_LEMMA = WordNetLemmatizer()

# =========================================
# Paths & loading TSVs
# =========================================
BASE_ROOT = "/home/abdullahnaveed/blue_li_abdullah/pattern_project/project_dataset/fakeddit_data/multimodal_only_samples"

TSV_TRAIN = os.path.join(BASE_ROOT, "multimodal_train.tsv")
TSV_VALID = os.path.join(BASE_ROOT, "multimodal_validate.tsv")
TSV_TEST  = os.path.join(BASE_ROOT, "multimodal_test_public.tsv")

IMG_DIRS = {
    "train": os.path.join(BASE_ROOT, "train_images"),
    "valid": os.path.join(BASE_ROOT, "valid_images"),
    "test":  os.path.join(BASE_ROOT, "test_images"),
}

use_cols = ["id", "title", "2_way_label", "3_way_label", "6_way_label"]
train_all = pd.read_csv(TSV_TRAIN, sep="\t", usecols=lambda c: True)
valid_all = pd.read_csv(TSV_VALID, sep="\t", usecols=lambda c: True)
test_all  = pd.read_csv(TSV_TEST,  sep="\t", usecols=lambda c: True)

train_all = train_all[[c for c in use_cols if c in train_all.columns]]
valid_all = valid_all[[c for c in use_cols if c in valid_all.columns]]
test_all  = test_all [[c for c in use_cols if c in test_all.columns]]

train_df = train_all.dropna(subset=["title"]).copy()
valid_df = valid_all.dropna(subset=["title"]).copy()
test_df  = test_all.dropna(subset=["title"]).copy()

for lab in ["2_way_label", "3_way_label", "6_way_label"]:
    for df in (train_df, valid_df, test_df):
        if lab in df.columns:
            df[lab] = df[lab].astype(int)

# Image paths
for split, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
    df["img_file"] = df["id"].astype(str) + ".jpg"
    df["img_path"] = df["img_file"].apply(lambda f: os.path.join(IMG_DIRS[split], f))
    df["img_exists"] = df["img_path"].apply(os.path.exists)
    before = len(df)
    df = df[df["img_exists"]].copy()
    if split == "train": train_df = df
    if split == "valid": valid_df = df
    if split == "test":  test_df  = df
    print(f"[{split}] kept {len(df):,} / {before:,} rows with existing images")

print("train_df columns:", train_df.columns.tolist())
print("valid_df shape:", valid_df.shape, "test_df shape:", test_df.shape)

def to_lists(df, label_col="6_way_label"):
    texts  = df["title"].tolist()
    labels = df[label_col].tolist() if label_col in df.columns else None
    imgs   = df["img_path"].tolist()
    return texts, labels, imgs

train_texts, train_labels6, train_imgs = to_lists(train_df, "6_way_label")
valid_texts, valid_labels6, valid_imgs = to_lists(valid_df, "6_way_label")
test_texts,  test_labels6,  test_imgs  = to_lists(test_df,  "6_way_label")

# =========================================
# Text preprocessing → tokenization/padding
# =========================================
def word_tokenize_fast(s: str):
    return [w for w in re.split(r"[^a-zA-Z]+", s.lower()) if w]

def preprocess_text_basic(s: str) -> str:
    s = re.sub(r"[^a-zA-Z]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def remove_stopwords_and_lemmatize(s: str) -> str:
    toks = word_tokenize_fast(s)
    toks = [w for w in toks if w not in _STOPWORDS]
    toks = [_LEMMA.lemmatize(w) for w in toks]
    return " ".join(toks)

def clean_corpus(texts: List[str]) -> List[str]:
    return [remove_stopwords_and_lemmatize(preprocess_text_basic(t)) for t in texts]

train_clean = clean_corpus(train_texts)
valid_clean = clean_corpus(valid_texts)
test_clean  = clean_corpus(test_texts)

MAX_VOCAB = 120_000
tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token=None)
tokenizer.fit_on_texts(train_clean + valid_clean + test_clean)
print("Vocabulary length:", len(tokenizer.word_index))

train_tok = tokenizer.texts_to_sequences(train_clean)
valid_tok = tokenizer.texts_to_sequences(valid_clean)
test_tok  = tokenizer.texts_to_sequences(test_clean)

SEQ_LEN = 15
train_pad = pad_sequences(train_tok, maxlen=SEQ_LEN, truncating="post", padding="post")
valid_pad = pad_sequences(valid_tok, maxlen=SEQ_LEN, truncating="post", padding="post")
test_pad  = pad_sequences(test_tok,  maxlen=SEQ_LEN, truncating="post", padding="post")
print("Shapes -> train:", train_pad.shape, "| valid:", valid_pad.shape, "| test:", test_pad.shape)

# =========================================
# Word embeddings (GloVe unchanged)
# =========================================
EMB_DIM   = 300
FREEZE_EMB = True
EMB_PATH = os.path.expanduser("~/data/glove/glove.6B.300d.txt")

def load_glove_embeddings(glove_path: str, vocab_set: set) -> dict:
    embeddings = {}
    if not os.path.isfile(glove_path):
        print(f"[WARN] GloVe file not found at: {glove_path}")
        print("       Embedding matrix will be randomly initialized.")
        return embeddings
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < EMB_DIM + 1:
                continue
            word = parts[0]
            if word in vocab_set:
                vec = np.asarray(parts[1:], dtype=np.float32)
                if vec.shape[0] == EMB_DIM:
                    embeddings[word] = vec
    print(f"[GloVe] Loaded {len(embeddings):,} vectors (matched in-vocab words).")
    return embeddings

word_index = tokenizer.word_index
num_words = min(MAX_VOCAB, len(word_index) + 1)
invocab_words = {w for w, idx in word_index.items() if idx < num_words}
glove = load_glove_embeddings(EMB_PATH, invocab_words)

rng = np.random.default_rng(seed=1234)
emb_matrix = rng.normal(loc=0.0, scale=0.05, size=(num_words, EMB_DIM)).astype(np.float32)
emb_matrix[0] = np.zeros((EMB_DIM,), dtype=np.float32)
hits = 0
for w, idx in word_index.items():
    if idx >= num_words: continue
    vec = glove.get(w)
    if vec is not None:
        emb_matrix[idx] = vec
        hits += 1
print(f"[EmbMatrix] num_words={num_words:,} | dim={EMB_DIM} | hits={hits:,} | OOV={num_words - 1 - hits:,}")

embedding_layer = nn.Embedding(num_embeddings=num_words, embedding_dim=EMB_DIM, padding_idx=0)
embedding_layer.weight.data = torch.from_numpy(emb_matrix)
embedding_layer.weight.requires_grad = not FREEZE_EMB

# =========================================
# Image preprocessing
# =========================================
IM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IM_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE = 224
RESIZE_TO = 256

def resize_short_side(img: np.ndarray, short=RESIZE_TO) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0 or w == 0: return img
    if h < w:
        new_h, new_w = short, int(round(w * (short / h)))
    else:
        new_w, new_h = short, int(round(h * (short / w)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def center_crop(img: np.ndarray, size=IMG_SIZE) -> np.ndarray:
    h, w = img.shape[:2]
    y0 = max(0, (h - size) // 2)
    x0 = max(0, (w - size) // 2)
    return img[y0:y0+size, x0:x0+size]

def img_to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    img = img_rgb.astype(np.float32) / 255.0
    img = (img - IM_MEAN) / IM_STD
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img)

def load_and_process_image(path: str) -> torch.Tensor:
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        im = np.full((IMG_SIZE, IMG_SIZE, 3), 128, dtype=np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = resize_short_side(im, RESIZE_TO)
    if im.shape[0] < IMG_SIZE or im.shape[1] < IMG_SIZE:
        im = cv2.resize(im, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    else:
        im = center_crop(im, IMG_SIZE)
    return img_to_tensor(im)

# =========================================
# Dataset & loaders (single, collate-safe class)
# =========================================
class MultimodalFakedditDataset(Dataset):
    """
    Returns a dict with collate-safe fields:
      - image: Tensor [3,224,224]
      - text_ids: Tensor [SEQ_LEN] (int64)
      - label: LongTensor [] (if available)
      - id: str
      - img_path: str
    """
    def __init__(self,
                 img_paths: List[str],
                 text_ids: np.ndarray,
                 labels: Optional[Union[List[int], np.ndarray]],
                 ids: Optional[List[Union[str,int]]] = None):
        assert len(img_paths) == len(text_ids)
        self.img_paths = img_paths
        self.text_ids  = np.asarray(text_ids, dtype=np.int64)
        if labels is not None:
            lbl = np.asarray(labels)
            lbl = np.where(np.isfinite(lbl.astype(float, copy=False)), lbl, -1)
            self.labels = lbl.astype(np.int64, copy=False)
        else:
            self.labels = None
        self.ids = [("" if (ids is None or ids[i] is None) else str(ids[i])) for i in range(len(img_paths))]

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, i):
        img_t  = load_and_process_image(self.img_paths[i])
        text_t = torch.from_numpy(self.text_ids[i])
        item = {"image": img_t, "text_ids": text_t, "id": self.ids[i], "img_path": self.img_paths[i]}
        if self.labels is not None:
            li = int(self.labels[i])
            if li >= 0:
                item["label"] = torch.tensor(li, dtype=torch.long)
        return item

def subset_arrays(
    imgs: List[str], text_pad: np.ndarray, labels: Optional[Union[List[int], np.ndarray]],
    max_n: Optional[int] = None, frac: Optional[float] = None, seed: int = 42, stratify: bool = True
) -> Tuple[List[str], np.ndarray, Optional[np.ndarray]]:
    N = len(imgs)
    if max_n is None and (frac is None or frac >= 1.0):
        return imgs, text_pad, (np.asarray(labels) if labels is not None else None)
    target = max_n if max_n is not None else int(np.ceil(N * float(frac)))
    target = max(1, min(target, N))
    idx_all = np.arange(N)
    if stratify and labels is not None:
        y = np.asarray(labels)
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=target, random_state=seed)
        sub_idx, _ = next(splitter.split(idx_all.reshape(-1, 1), y))
        chosen = sub_idx
    else:
        rng = np.random.default_rng(seed); chosen = np.sort(rng.choice(idx_all, size=target, replace=False))
    imgs_sub   = [imgs[i] for i in chosen]
    text_sub   = text_pad[chosen]
    labels_sub = (np.asarray(labels)[chosen] if labels is not None else None)
    return imgs_sub, text_sub, labels_sub

def classes_for_label_key(label_key: str) -> int:
    return {"2_way_label":2, "3_way_label":3, "6_way_label":6}[label_key]

def get_label_arrays(label_key: str):
    if label_key == "6_way_label":
        return list(train_df["6_way_label"].values), list(valid_df["6_way_label"].values), list(test_df["6_way_label"].values)
    if label_key == "3_way_label":
        return list(train_df["3_way_label"].values), list(valid_df["3_way_label"].values), list(test_df["3_way_label"].values)
    if label_key == "2_way_label":
        return list(train_df["2_way_label"].values), list(valid_df["2_way_label"].values), list(test_df["2_way_label"].values)
    raise ValueError

def build_loaders(label_key: str,
                  max_train: Optional[int] = None, frac_train: Optional[float] = None,
                  max_valid: Optional[int] = None, frac_valid: Optional[float] = None,
                  max_test:  Optional[int] = None, frac_test:  Optional[float] = None,
                  batch_size: int = 32, num_workers: int = 0):
    tr_lbl, va_lbl, te_lbl = get_label_arrays(label_key)
    tr_imgs, tr_pad, tr_lbl = subset_arrays(train_imgs, train_pad, tr_lbl, max_n=max_train, frac=frac_train, seed=42, stratify=True)
    va_imgs, va_pad, va_lbl = subset_arrays(valid_imgs, valid_pad, va_lbl, max_n=max_valid, frac=frac_valid, seed=42, stratify=True)
    te_imgs, te_pad, te_lbl = subset_arrays(test_imgs,  test_pad,  te_lbl, max_n=max_test,  frac=frac_test,  seed=42, stratify=True)
    def _mk(imgs, pad, lbl, shuffle):
        return DataLoader(MultimodalFakedditDataset(imgs, pad, lbl, ids=None),
                          batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return _mk(tr_imgs,tr_pad,tr_lbl,True), _mk(va_imgs,va_pad,va_lbl,False), _mk(te_imgs,te_pad,te_lbl,False)

# =========================================
# Model
# =========================================
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class ImageTower(nn.Module):
    def __init__(self, in_ch=3, feat_dim=256):
        super().__init__()
        self.block1 = nn.Sequential(ConvBNReLU(in_ch, 32, k=5, s=2, p=2), ConvBNReLU(32, 32))
        self.block2 = nn.Sequential(ConvBNReLU(32, 64, s=2), ConvBNReLU(64, 64))
        self.block3 = nn.Sequential(ConvBNReLU(64, 128, s=2), ConvBNReLU(128, 128))
        self.block4 = nn.Sequential(ConvBNReLU(128, 256, s=2), ConvBNReLU(256, 256))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(256, feat_dim)
    def forward(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x); x = self.block4(x)
        return self.proj(self.gap(x).flatten(1))

class TextTower(nn.Module):
    def __init__(self, embedding: nn.Embedding, hidden=128, num_layers=1, bidirectional=True, dropout=0.1):
        super().__init__()
        self.embedding = embedding
        emb_dim = embedding.embedding_dim
        self.gru = nn.GRU(emb_dim, hidden, num_layers=num_layers, batch_first=True,
                          bidirectional=bidirectional, dropout=dropout if num_layers>1 else 0.0)
        self.out_dim = hidden * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, token_ids):
        x, _ = self.gru(self.embedding(token_ids))
        return self.dropout(x.mean(dim=1))

class MultiModalClassifier(nn.Module):
    def __init__(self, embedding_layer: nn.Embedding,
                 img_feat_dim=256, txt_hidden=128, txt_layers=1, txt_bidir=True,
                 fusion_hidden=256, num_classes=6, dropout=0.2):
        super().__init__()
        self.image_tower = ImageTower(3, img_feat_dim)
        self.text_tower  = TextTower(embedding_layer, hidden=txt_hidden,
                                     num_layers=txt_layers, bidirectional=txt_bidir, dropout=dropout)
        fusion_in = img_feat_dim + self.text_tower.out_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes),
        )
    def forward(self, image, text_ids):
        return self.fusion(torch.cat([self.image_tower(image), self.text_tower(text_ids)], dim=1))

def classes_for_label_key(label_key: str) -> int:
    return {"2_way_label":2, "3_way_label":3, "6_way_label":6}[label_key]

def build_model(label_key: str, embedding_layer: nn.Embedding,
                img_feat_dim=256, txt_hidden=128, txt_layers=1, txt_bidir=True,
                fusion_hidden=256, dropout=0.2, device: Optional[torch.device]=None):
    num_classes = classes_for_label_key(label_key)
    model = MultiModalClassifier(embedding_layer, img_feat_dim, txt_hidden, txt_layers, txt_bidir, fusion_hidden, num_classes, dropout)
    if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)

# =========================================
# Training & evaluation
# =========================================
def make_criterion(num_classes: int, class_weights: Optional[torch.Tensor] = None, device: Optional[torch.device] = None):
    if class_weights is not None and device is not None:
        class_weights = class_weights.to(device)
    return nn.CrossEntropyLoss(weight=class_weights)

@torch.no_grad()
def _metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor):
    preds = logits.argmax(dim=1)
    acc = (preds == labels).float().mean().item()
    return preds.cpu().numpy(), acc

@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, max_valid_batches: Optional[int] = None):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n_batches = 0.0, 0
    criterion = getattr(model, "_criterion", None)

    for b_idx, batch in enumerate(loader):
        if max_valid_batches is not None and b_idx >= max_valid_batches: break
        imgs = batch["image"].to(device, non_blocking=True)
        txt  = batch["text_ids"].to(device, non_blocking=True)
        labels = batch.get("label", None)
        logits = model(imgs, txt)
        if labels is None: continue
        labels = labels.to(device, non_blocking=True)
        if criterion is not None:
            loss = criterion(logits, labels)
            total_loss += loss.item()
        preds_np, _ = _metrics_from_logits(logits, labels)
        all_preds.append(preds_np)
        all_labels.append(labels.detach().cpu().numpy())
        n_batches += 1

    if len(all_labels) == 0:
        return {"loss": None, "acc": None, "f1_macro": None, "report": None, "confusion": None}

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)          # <-- safe
    rep = classification_report(y_true, y_pred, digits=4, zero_division=0)    # <-- safe
    cm  = confusion_matrix(y_true, y_pred)
    avg_loss = (total_loss / max(1, n_batches)) if criterion is not None else None
    return {"loss": avg_loss, "acc": acc, "f1_macro": f1m, "report": rep, "confusion": cm}

def train_one_epoch(model: nn.Module, loader, optimizer: optim.Optimizer,
                    device: torch.device, scaler: Optional[amp.GradScaler] = None,
                    max_train_batches: Optional[int] = None, grad_clip_norm: Optional[float] = 1.0):
    model.train()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    criterion = model._criterion
    for b_idx, batch in enumerate(loader):
        if max_train_batches is not None and b_idx >= max_train_batches: break
        imgs = batch["image"].to(device, non_blocking=True)
        txt  = batch["text_ids"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits = model(imgs, txt)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer); scaler.update()
        else:
            logits = model(imgs, txt); loss = criterion(logits, labels)
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        _, acc_b = _metrics_from_logits(logits, labels)
        total_loss += loss.item(); total_acc += acc_b; n_batches += 1
    return {"loss": total_loss / max(1, n_batches), "acc": total_acc / max(1, n_batches), "batches": n_batches}

def fit(model: nn.Module, train_loader, valid_loader, device: torch.device,
        epochs: int = 5, lr: float = 1e-3, weight_decay: float = 0.0,
        class_weights: Optional[torch.Tensor] = None, use_amp: bool = True,
        max_train_batches: Optional[int] = None, max_valid_batches: Optional[int] = None,
        ckpt_path: str = "best_multimodal.pt", scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        early_stop_patience: Optional[int] = 3):
    criterion = make_criterion(model.fusion[-1].out_features, class_weights=class_weights, device=device)
    model._criterion = criterion
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = amp.GradScaler('cuda', enabled=(use_amp and device.type == "cuda"))

    best_f1, best_epoch, epochs_no_improve = -1.0, -1, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(model, train_loader, optimizer, device, scaler=scaler, max_train_batches=max_train_batches)
        valid_stats = evaluate(model, valid_loader, device, max_valid_batches=max_valid_batches)
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                val_loss = valid_stats["loss"] if valid_stats["loss"] is not None else train_stats["loss"]
                scheduler.step(val_loss)
            else:
                scheduler.step()
        dt = time.time() - t0
        v_loss = valid_stats["loss"] if valid_stats["loss"] is not None else float('nan')
        v_acc  = valid_stats["acc"] if valid_stats["acc"] is not None else float('nan')
        v_f1   = valid_stats["f1_macro"] if valid_stats["f1_macro"] is not None else float('nan')
        print(f"[Epoch {ep:02d}] train loss {train_stats['loss']:.4f} acc {train_stats['acc']:.4f} | "
              f"valid loss {v_loss:.4f} acc {v_acc:.4f} f1 {v_f1:.4f} | {dt:.1f}s")
        cur_f1 = valid_stats["f1_macro"] if valid_stats["f1_macro"] is not None else -1.0
        if cur_f1 > best_f1:
            best_f1, best_epoch = cur_f1, ep
            torch.save({"model": model.state_dict(), "epoch": ep, "f1_macro": cur_f1}, ckpt_path)
            print(f"  ↳ Saved new best to {ckpt_path} (F1={cur_f1:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if (early_stop_patience is not None) and (epochs_no_improve >= early_stop_patience):
                print(f"  ↳ Early stopping at epoch {ep} (no improve {epochs_no_improve} epochs)."); break
    print(f"Best epoch {best_epoch} with F1={best_f1:.4f}")
    return {"best_epoch": best_epoch, "best_f1_macro": best_f1, "ckpt_path": ckpt_path}

# =========================================
# Runner 2_way
# =========================================
CONFIG = {
    "label_key": "2_way_label",
    "max_train": 15000, "max_valid": 1500, "max_test": 1500,
    "batch_size": 32, "num_workers": 0,
    "img_feat_dim": 256, "txt_hidden": 128, "txt_layers": 1, "txt_bidir": True,
    "fusion_hidden": 256, "dropout": 0.2,
    "epochs": 10, "lr": 1e-3, "weight_decay": 1e-4,
    "use_amp": True, "max_train_batches": None, "max_valid_batches": None,
    "class_weights": None, "ckpt_path": None,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if CONFIG["ckpt_path"] is None:
    CONFIG["ckpt_path"] = f"best_mm_{CONFIG['label_key'].replace('_way_label','')}way.pt"
print("Using device:", device)
print("Checkpoint:", CONFIG["ckpt_path"])

train_loader, valid_loader, test_loader = build_loaders(
    label_key=CONFIG["label_key"],
    max_train=CONFIG["max_train"], frac_train=None,
    max_valid=CONFIG["max_valid"], frac_valid=None,
    max_test=CONFIG["max_test"],   frac_test=None,
    batch_size=CONFIG["batch_size"], num_workers=CONFIG["num_workers"],
)
print("batches -> train:", len(train_loader), "| valid:", len(valid_loader), "| test:", len(test_loader))

model = build_model(
    label_key=CONFIG["label_key"], embedding_layer=embedding_layer,
    img_feat_dim=CONFIG["img_feat_dim"], txt_hidden=CONFIG["txt_hidden"],
    txt_layers=CONFIG["txt_layers"], txt_bidir=CONFIG["txt_bidir"],
    fusion_hidden=CONFIG["fusion_hidden"], dropout=CONFIG["dropout"], device=device,
)

# ---- Optional: compute class weights from the actual train batches ----
try:
    all_train_y = np.concatenate([b["label"].numpy() for b in train_loader if "label" in b])
    counts = np.bincount(all_train_y, minlength=classes_for_label_key(CONFIG["label_key"]))
    inv = 1.0 / np.maximum(counts, 1)
    CONFIG["class_weights"] = torch.tensor(inv / inv.sum() * len(counts), dtype=torch.float32)
    print("Class weights:", CONFIG["class_weights"])
except Exception as e:
    print("[WARN] Could not compute class weights from batches:", e)

train_summary = fit(
    model=model, train_loader=train_loader, valid_loader=valid_loader, device=device,
    epochs=CONFIG["epochs"], lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"],
    class_weights=CONFIG["class_weights"], use_amp=CONFIG["use_amp"],
    max_train_batches=CONFIG["max_train_batches"], max_valid_batches=CONFIG["max_valid_batches"],
    ckpt_path=CONFIG["ckpt_path"], scheduler=None, early_stop_patience=3,
)

# Load best & evaluate on TEST
if os.path.isfile(train_summary["ckpt_path"]):
    ckpt = torch.load(train_summary["ckpt_path"], map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded best checkpoint from epoch {ckpt.get('epoch')} (F1={ckpt.get('f1_macro')})")

test_stats = evaluate(model, test_loader, device)
print("\n=== TEST METRICS ===")
print("Acc:      ", test_stats["acc"])
print("F1-macro: ", test_stats["f1_macro"])
print("Confusion:\n", test_stats["confusion"])
print("\nClassification report:\n", test_stats["report"])

# =========================================
# Runner 3_way
# =========================================
CONFIG = {
    "label_key": "3_way_label",
    "max_train": 15000, "max_valid": 1500, "max_test": 1500,
    "batch_size": 32, "num_workers": 0,
    "img_feat_dim": 256, "txt_hidden": 128, "txt_layers": 1, "txt_bidir": True,
    "fusion_hidden": 256, "dropout": 0.2,
    "epochs": 10, "lr": 1e-3, "weight_decay": 1e-4,
    "use_amp": True, "max_train_batches": None, "max_valid_batches": None,
    "class_weights": None, "ckpt_path": None,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if CONFIG["ckpt_path"] is None:
    CONFIG["ckpt_path"] = f"best_mm_{CONFIG['label_key'].replace('_way_label','')}way.pt"
print("Using device:", device)
print("Checkpoint:", CONFIG["ckpt_path"])

train_loader, valid_loader, test_loader = build_loaders(
    label_key=CONFIG["label_key"],
    max_train=CONFIG["max_train"], frac_train=None,
    max_valid=CONFIG["max_valid"], frac_valid=None,
    max_test=CONFIG["max_test"],   frac_test=None,
    batch_size=CONFIG["batch_size"], num_workers=CONFIG["num_workers"],
)
print("batches -> train:", len(train_loader), "| valid:", len(valid_loader), "| test:", len(test_loader))

model = build_model(
    label_key=CONFIG["label_key"], embedding_layer=embedding_layer,
    img_feat_dim=CONFIG["img_feat_dim"], txt_hidden=CONFIG["txt_hidden"],
    txt_layers=CONFIG["txt_layers"], txt_bidir=CONFIG["txt_bidir"],
    fusion_hidden=CONFIG["fusion_hidden"], dropout=CONFIG["dropout"], device=device,
)

# ---- Optional: compute class weights from the actual train batches ----
try:
    all_train_y = np.concatenate([b["label"].numpy() for b in train_loader if "label" in b])
    counts = np.bincount(all_train_y, minlength=classes_for_label_key(CONFIG["label_key"]))
    inv = 1.0 / np.maximum(counts, 1)
    CONFIG["class_weights"] = torch.tensor(inv / inv.sum() * len(counts), dtype=torch.float32)
    print("Class weights:", CONFIG["class_weights"])
except Exception as e:
    print("[WARN] Could not compute class weights from batches:", e)

train_summary = fit(
    model=model, train_loader=train_loader, valid_loader=valid_loader, device=device,
    epochs=CONFIG["epochs"], lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"],
    class_weights=CONFIG["class_weights"], use_amp=CONFIG["use_amp"],
    max_train_batches=CONFIG["max_train_batches"], max_valid_batches=CONFIG["max_valid_batches"],
    ckpt_path=CONFIG["ckpt_path"], scheduler=None, early_stop_patience=3,
)

# Load best & evaluate on TEST
if os.path.isfile(train_summary["ckpt_path"]):
    ckpt = torch.load(train_summary["ckpt_path"], map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded best checkpoint from epoch {ckpt.get('epoch')} (F1={ckpt.get('f1_macro')})")

test_stats = evaluate(model, test_loader, device)
print("\n=== TEST METRICS ===")
print("Acc:      ", test_stats["acc"])
print("F1-macro: ", test_stats["f1_macro"])
print("Confusion:\n", test_stats["confusion"])
print("\nClassification report:\n", test_stats["report"])

# =========================================
# Runner 6_way
# =========================================
CONFIG = {
    "label_key": "6_way_label",
    "max_train": 15000, "max_valid": 1500, "max_test": 1500,
    "batch_size": 32, "num_workers": 0,
    "img_feat_dim": 256, "txt_hidden": 128, "txt_layers": 1, "txt_bidir": True,
    "fusion_hidden": 256, "dropout": 0.2,
    "epochs": 10, "lr": 1e-3, "weight_decay": 1e-4,
    "use_amp": True, "max_train_batches": None, "max_valid_batches": None,
    "class_weights": None, "ckpt_path": None,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if CONFIG["ckpt_path"] is None:
    CONFIG["ckpt_path"] = f"best_mm_{CONFIG['label_key'].replace('_way_label','')}way.pt"
print("Using device:", device)
print("Checkpoint:", CONFIG["ckpt_path"])

train_loader, valid_loader, test_loader = build_loaders(
    label_key=CONFIG["label_key"],
    max_train=CONFIG["max_train"], frac_train=None,
    max_valid=CONFIG["max_valid"], frac_valid=None,
    max_test=CONFIG["max_test"],   frac_test=None,
    batch_size=CONFIG["batch_size"], num_workers=CONFIG["num_workers"],
)
print("batches -> train:", len(train_loader), "| valid:", len(valid_loader), "| test:", len(test_loader))

model = build_model(
    label_key=CONFIG["label_key"], embedding_layer=embedding_layer,
    img_feat_dim=CONFIG["img_feat_dim"], txt_hidden=CONFIG["txt_hidden"],
    txt_layers=CONFIG["txt_layers"], txt_bidir=CONFIG["txt_bidir"],
    fusion_hidden=CONFIG["fusion_hidden"], dropout=CONFIG["dropout"], device=device,
)

# ---- Optional: compute class weights from the actual train batches ----
try:
    all_train_y = np.concatenate([b["label"].numpy() for b in train_loader if "label" in b])
    counts = np.bincount(all_train_y, minlength=classes_for_label_key(CONFIG["label_key"]))
    inv = 1.0 / np.maximum(counts, 1)
    CONFIG["class_weights"] = torch.tensor(inv / inv.sum() * len(counts), dtype=torch.float32)
    print("Class weights:", CONFIG["class_weights"])
except Exception as e:
    print("[WARN] Could not compute class weights from batches:", e)

train_summary = fit(
    model=model, train_loader=train_loader, valid_loader=valid_loader, device=device,
    epochs=CONFIG["epochs"], lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"],
    class_weights=CONFIG["class_weights"], use_amp=CONFIG["use_amp"],
    max_train_batches=CONFIG["max_train_batches"], max_valid_batches=CONFIG["max_valid_batches"],
    ckpt_path=CONFIG["ckpt_path"], scheduler=None, early_stop_patience=3,
)

# Load best & evaluate on TEST
if os.path.isfile(train_summary["ckpt_path"]):
    ckpt = torch.load(train_summary["ckpt_path"], map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded best checkpoint from epoch {ckpt.get('epoch')} (F1={ckpt.get('f1_macro')})")

test_stats = evaluate(model, test_loader, device)
print("\n=== TEST METRICS ===")
print("Acc:      ", test_stats["acc"])
print("F1-macro: ", test_stats["f1_macro"])
print("Confusion:\n", test_stats["confusion"])
print("\nClassification report:\n", test_stats["report"])

print("\n<====== Done =======>")






































# ####################################################################################
# #################################################################################### Multimodal CNN (text + images): imports
# ####################################################################################
# # Core
# import os, time, re, threading, multiprocessing as mp
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import math

# # Images
# import cv2

# # Text preprocessing
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# from typing import List, Optional, Union, Tuple


# from sklearn.feature_extraction.text import (
#     TfidfVectorizer, TfidfTransformer, CountVectorizer
# )
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
# from sklearn.model_selection import StratifiedShuffleSplit

# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# import torch
# from torch import nn, optim
# from torch.utils.data import Dataset, DataLoader

# # NLTK resources
# nltk.download("stopwords")
# nltk.download("punkt")
# nltk.download("wordnet")

# # Jupyter inline plotting (optional)
# # %matplotlib inline

# ####################################################################################
# #################################################################################### Load datasets + Set up paths
# ####################################################################################

# # === Paths (your local layout) ===
# BASE_ROOT = "/home/abdullahnaveed/blue_li_abdullah/pattern_project/project_dataset/fakeddit_data/multimodal_only_samples"

# TSV_TRAIN = os.path.join(BASE_ROOT, "multimodal_train.tsv")
# TSV_VALID = os.path.join(BASE_ROOT, "multimodal_validate.tsv")
# TSV_TEST  = os.path.join(BASE_ROOT, "multimodal_test_public.tsv")

# IMG_DIRS = {
#     "train": os.path.join(BASE_ROOT, "train_images"),
#     "valid": os.path.join(BASE_ROOT, "valid_images"),
#     "test":  os.path.join(BASE_ROOT, "test_images"),
# }

# # === Load TSVs ===
# use_cols = ["id", "title", "2_way_label", "3_way_label", "6_way_label"]
# train_all = pd.read_csv(TSV_TRAIN, sep="\t", usecols=lambda c: True)
# valid_all = pd.read_csv(TSV_VALID, sep="\t", usecols=lambda c: True)
# test_all  = pd.read_csv(TSV_TEST,  sep="\t", usecols=lambda c: True)

# # Keep only needed columns if present
# train_all = train_all[[c for c in use_cols if c in train_all.columns]]
# valid_all = valid_all[[c for c in use_cols if c in valid_all.columns]]
# test_all  = test_all [[c for c in use_cols if c in test_all.columns]]

# # === Basic cleaning: require non-null titles ===
# train_df = train_all.dropna(subset=["title"]).copy()
# valid_df = valid_all.dropna(subset=["title"]).copy()
# test_df  = test_all.dropna(subset=["title"]).copy()

# # Ensure integer labels (where available)
# for lab in ["2_way_label", "3_way_label", "6_way_label"]:
#     for df in (train_df, valid_df, test_df):
#         if lab in df.columns:
#             df[lab] = df[lab].astype(int)

# # === Image filenames & resolved paths ===
# for split, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
#     df["img_file"] = df["id"].astype(str) + ".jpg"
#     df["img_path"] = df["img_file"].apply(lambda f: os.path.join(IMG_DIRS[split], f))
#     # (optional) drop rows whose image is missing on disk
#     df["img_exists"] = df["img_path"].apply(os.path.exists)
#     before = len(df)
#     df.dropna(subset=["img_path"], inplace=True)
#     df = df[df["img_exists"]].copy()
#     if split == "train": train_df = df
#     if split == "valid": valid_df = df
#     if split == "test":  test_df  = df
#     print(f"[{split}] kept {len(df):,} / {before:,} rows with existing images")

# # === Quick peek ===
# print("train_df columns:", train_df.columns.tolist())
# print("valid_df shape:", valid_df.shape, "test_df shape:", test_df.shape)

# # === Convenience: lists for later (text, labels, image paths) ===
# def to_lists(df, label_col="6_way_label"):
#     texts  = df["title"].tolist()
#     labels = df[label_col].tolist() if label_col in df.columns else None
#     imgs   = df["img_path"].tolist()
#     return texts, labels, imgs

# # Example (we will switch label_col later as needed: "2_way_label", "3_way_label", "6_way_label")
# train_texts, train_labels6, train_imgs = to_lists(train_df, "6_way_label")
# valid_texts, valid_labels6, valid_imgs = to_lists(valid_df, "6_way_label")
# test_texts,  test_labels6,  test_imgs  = to_lists(test_df,  "6_way_label")


# ####################################################################################
# #################################################################################### Text preprocessing → Tokenization → Padding
# ####################################################################################

# # === Block 3: text preprocessing → tokenization → padding ===
# # ---- NLTK resource bootstrap (run once; harmless if already present) ----
# def _ensure(pkg, key):
#     try:
#         nltk.data.find(pkg)
#     except LookupError:
#         nltk.download(key)
# _ensure("tokenizers/punkt", "punkt")
# _ensure("tokenizers/punkt_tab", "punkt_tab")
# _ensure("corpora/stopwords", "stopwords")
# _ensure("corpora/wordnet", "wordnet")
# _ensure("corpora/omw-1.4", "omw-1.4")

# _stopwords = set(stopwords.words("english"))
# _lemmatizer = WordNetLemmatizer()

# # --- Optional lightweight tokenizer to avoid punkt dependency completely ---
# def word_tokenize_fast(s: str):
#     # split on non-letters (after lowercase) and drop empties
#     return [w for w in re.split(r"[^a-zA-Z]+", s.lower()) if w]

# # --- Basic cleaners (mirror reference flow) ---
# def preprocess_text_basic(s: str) -> str:
#     s = re.sub(r"[^a-zA-Z]", " ", s)      # keep only letters
#     s = re.sub(r"\s+", " ", s).strip()    # collapse spaces
#     return s.lower()

# def remove_stopwords_and_lemmatize(s: str) -> str:
#     # Choose ONE: use NLTK word_tokenize or the lightweight tokenizer
#     # from nltk.tokenize import word_tokenize
#     # toks = word_tokenize(s)
#     toks = word_tokenize_fast(s)           # robust, no downloads needed
#     toks = [w for w in toks if w not in _stopwords]
#     toks = [_lemmatizer.lemmatize(w) for w in toks]
#     return " ".join(toks)

# def clean_corpus(texts: List[str]) -> List[str]:
#     return [remove_stopwords_and_lemmatize(preprocess_text_basic(t)) for t in texts]

# # --- Clean each split (train_texts/valid_texts/test_texts were defined in Block 2) ---
# train_clean = clean_corpus(train_texts)
# valid_clean = clean_corpus(valid_texts)
# test_clean  = clean_corpus(test_texts)

# # --- Tokenizer (fit on ALL text as in the reference) ---
# MAX_VOCAB = 120_000
# tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token=None)
# tokenizer.fit_on_texts(train_clean + valid_clean + test_clean)
# print("Vocabulary length:", len(tokenizer.word_index))

# # --- Convert to sequences ---
# train_tok = tokenizer.texts_to_sequences(train_clean)
# valid_tok = tokenizer.texts_to_sequences(valid_clean)
# test_tok  = tokenizer.texts_to_sequences(test_clean)

# # --- Pad/truncate to fixed length (reference uses 15) ---
# SEQ_LEN = 15
# train_pad = pad_sequences(train_tok, maxlen=SEQ_LEN, truncating="post", padding="post")
# valid_pad = pad_sequences(valid_tok, maxlen=SEQ_LEN, truncating="post", padding="post")
# test_pad  = pad_sequences(test_tok,  maxlen=SEQ_LEN, truncating="post", padding="post")

# print("Shapes -> train:", train_pad.shape, "| valid:", valid_pad.shape, "| test:", test_pad.shape)

# ####################################################################################
# #################################################################################### Word Embeddings & Vocabulary
# ####################################################################################

# # ----- Config -----
# EMB_DIM   = 300
# MAX_VOCAB = 120_000        # must match Block 3
# FREEZE_EMB = True          # set False to fine-tune word embeddings

# # Change this path to your local GloVe file location
# EMB_PATH = os.path.expanduser("~/data/glove/glove.6B.300d.txt")

# # ----- Loader -----
# def load_glove_embeddings(glove_path: str, vocab_set: set) -> dict:
#     """
#     Load only the rows we need (words in vocab_set) from a GloVe .txt file.
#     Returns: dict[str -> np.ndarray(shape=(EMB_DIM,))]
#     """
#     embeddings = {}
#     if not os.path.isfile(glove_path):
#         print(f"[WARN] GloVe file not found at: {glove_path}")
#         print("       Embedding matrix will be randomly initialized.")
#         return embeddings

#     with open(glove_path, "r", encoding="utf-8") as f:
#         for line in f:
#             parts = line.rstrip().split(" ")
#             if len(parts) < EMB_DIM + 1:
#                 continue
#             word = parts[0]
#             if word in vocab_set:
#                 vec = np.asarray(parts[1:], dtype=np.float32)
#                 if vec.shape[0] == EMB_DIM:
#                     embeddings[word] = vec
#     print(f"[GloVe] Loaded {len(embeddings):,} vectors (matched in-vocab words).")
#     return embeddings

# # ----- Build embedding matrix for our tokenizer -----
# word_index = tokenizer.word_index  # from Block 3
# # limit to MAX_VOCAB most frequent (Keras assigns lower indices to more frequent)
# num_words = min(MAX_VOCAB, len(word_index) + 1)  # +1 for index padding at 0

# # Collect the subset of words we might load
# invocab_words = {w for w, idx in word_index.items() if idx < num_words}

# # Load glove vectors (subset)
# glove = load_glove_embeddings(EMB_PATH, invocab_words)

# # Initialize embedding matrix
# rng = np.random.default_rng(seed=1234)
# emb_matrix = rng.normal(loc=0.0, scale=0.05, size=(num_words, EMB_DIM)).astype(np.float32)

# # Ensure padding idx (0) is zeros
# emb_matrix[0] = np.zeros((EMB_DIM,), dtype=np.float32)

# # Fill known words
# hits = 0
# for w, idx in word_index.items():
#     if idx >= num_words:
#         continue
#     vec = glove.get(w)
#     if vec is not None:
#         emb_matrix[idx] = vec
#         hits += 1

# print(f"[EmbMatrix] num_words={num_words:,} | dim={EMB_DIM} | hits={hits:,} | OOV={num_words - 1 - hits:,}")

# # ----- Torch embedding layer -----
# embedding_layer = nn.Embedding(num_embeddings=num_words, embedding_dim=EMB_DIM, padding_idx=0)
# embedding_layer.weight.data = torch.from_numpy(emb_matrix)
# embedding_layer.weight.requires_grad = not FREEZE_EMB

# # Example sanity check (optional):
# with torch.no_grad():
#     sample_ids = torch.tensor([[1, 2, 3, 0, 5]])  # pretend token ids
#     sample_vecs = embedding_layer(sample_ids)
#     print("Embedding output shape:", sample_vecs.shape)  # (1, seq_len, EMB_DIM)

# ####################################################################################
# #################################################################################### Image Preprocessing & Utilities
# ####################################################################################

# # === Block 5: Image pipeline + Multimodal Dataset + DataLoaders (Python 3.8-safe) ===
# # ----- Image normalization (default: ImageNet stats; you can plug in your train-estimated stats) -----
# IM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
# IM_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# IMG_SIZE = 224   # final crop size
# RESIZE_TO = 256  # resize the shortest side to this before center-crop

# def resize_short_side(img: np.ndarray, short=RESIZE_TO) -> np.ndarray:
#     h, w = img.shape[:2]
#     if h == 0 or w == 0:
#         return img
#     if h < w:
#         new_h = short
#         new_w = int(round(w * (short / h)))
#     else:
#         new_w = short
#         new_h = int(round(h * (short / w)))
#     return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

# def center_crop(img: np.ndarray, size=IMG_SIZE) -> np.ndarray:
#     h, w = img.shape[:2]
#     y0 = max(0, (h - size) // 2)
#     x0 = max(0, (w - size) // 2)
#     return img[y0:y0+size, x0:x0+size]

# def img_to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
#     # HWC [0..255] -> CHW float32 normalized
#     img = img_rgb.astype(np.float32) / 255.0
#     img = (img - IM_MEAN) / IM_STD
#     img = np.transpose(img, (2, 0, 1))  # CHW
#     return torch.from_numpy(img)        # float32 tensor

# def load_and_process_image(path: str) -> torch.Tensor:
#     im = cv2.imread(path, cv2.IMREAD_COLOR)          # BGR
#     if im is None:
#         # fallback: create a gray image to avoid breaking the batch
#         im = np.full((IMG_SIZE, IMG_SIZE, 3), 128, dtype=np.uint8)
#     im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)         # RGB
#     im = resize_short_side(im, RESIZE_TO)
#     # guard for small images after resize
#     if im.shape[0] < IMG_SIZE or im.shape[1] < IMG_SIZE:
#         im = cv2.resize(im, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
#     else:
#         im = center_crop(im, IMG_SIZE)
#     return img_to_tensor(im)

# class MultimodalFakedditDataset(Dataset):
#     """
#     Returns a dict:
#       {
#         'image': Tensor [3, 224, 224],
#         'text_ids': Tensor [SEQ_LEN],
#         'label': LongTensor [] (class index),
#         'id': str,
#         'img_path': str
#       }
#     """
#     def __init__(self,
#                  img_paths: List[str],
#                  text_ids: np.ndarray,                           # shape [N, SEQ_LEN], from Block 3
#                  labels: Optional[Union[List[int], np.ndarray]],
#                  ids: Optional[List[str]] = None,
#                  label_key: str = "6_way_label"):
#         assert text_ids is not None and len(img_paths) == len(text_ids), "img_paths and text_ids must align"
#         if labels is not None:
#             assert len(labels) == len(img_paths), "labels and img_paths must align"
#         self.img_paths = img_paths
#         self.text_ids = np.asarray(text_ids, dtype=np.int64)
#         self.labels = np.asarray(labels, dtype=np.int64) if labels is not None else None
#         self.ids = ids if ids is not None else [None] * len(img_paths)
#         assert label_key in {"2_way_label", "3_way_label", "6_way_label"}, "label_key must be in {2,3,6}_way_label"
#         self.label_key = label_key

#     def __len__(self):
#         return len(self.img_paths)

#     def __getitem__(self, i):
#         img_t = load_and_process_image(self.img_paths[i])     # [3,224,224]
#         text_t = torch.from_numpy(self.text_ids[i])           # [SEQ_LEN], int64
#         item = {
#             "image": img_t,
#             "text_ids": text_t,
#             "id": self.ids[i],
#             "img_path": self.img_paths[i],
#         }
#         if self.labels is not None:
#             item["label"] = torch.tensor(int(self.labels[i]), dtype=torch.long)
#         return item

# def make_loader(imgs, text_pad, labels, ids=None, label_key="6_way_label",
#                 batch_size=32, shuffle=False, num_workers=0, pin_memory=True):
#     ds = MultimodalFakedditDataset(
#         img_paths=imgs,
#         text_ids=text_pad,
#         labels=labels,
#         ids=ids,
#         label_key=label_key
#     )
#     return DataLoader(
#         ds, batch_size=batch_size, shuffle=shuffle,
#         num_workers=num_workers, pin_memory=pin_memory
#     )

# # ------- Build loaders for 6-way as example (we can switch to 2/3 later) -------
# BATCH_SIZE = 32
# NUM_WORKERS = 0

# train_loader_6 = make_loader(train_imgs, train_pad, train_labels6, ids=None,
#                              label_key="6_way_label", batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
# valid_loader_6 = make_loader(valid_imgs, valid_pad, valid_labels6, ids=None,
#                              label_key="6_way_label", batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
# test_loader_6  = make_loader(test_imgs,  test_pad,  test_labels6,  ids=None,
#                              label_key="6_way_label", batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# print("train batches:", len(train_loader_6), "| valid:", len(valid_loader_6), "| test:", len(test_loader_6))

# ####################################################################################
# #################################################################################### Multimodal Model, Dataset & Loaders
# ####################################################################################
# # ---------- Collate-safe Dataset ----------
# class MultimodalFakedditDataset(Dataset):
#     """
#     Returns a dict with only collate-safe types:
#       'image': Tensor [3, 224, 224]
#       'text_ids': Tensor [SEQ_LEN] (int64)
#       'label': LongTensor [] (if labels provided and valid)
#       'id': str (never None; empty "" if unknown)
#       'img_path': str
#     """
#     def __init__(self,
#                  img_paths: List[str],
#                  text_ids: np.ndarray,                           # shape [N, SEQ_LEN]
#                  labels: Optional[Union[List[int], np.ndarray]],
#                  ids: Optional[List[Union[str,int]]] = None):
#         assert text_ids is not None and len(img_paths) == len(text_ids), "img_paths and text_ids must align"
#         if labels is not None:
#             assert len(labels) == len(img_paths), "labels and img_paths must align"

#         self.img_paths = img_paths
#         self.text_ids  = np.asarray(text_ids, dtype=np.int64)

#         if labels is not None:
#             lbl = np.asarray(labels)
#             # Replace NaN with -1; omit in __getitem__ if <0
#             lbl = np.where(np.isfinite(lbl.astype(float, copy=False)), lbl, -1)
#             self.labels = lbl.astype(np.int64, copy=False)
#         else:
#             self.labels = None

#         self.ids = [("" if (ids is None or ids[i] is None) else str(ids[i]))
#                     for i in range(len(img_paths))]

#     def __len__(self):
#         return len(self.img_paths)

#     def __getitem__(self, i):
#         img_t  = load_and_process_image(self.img_paths[i])      # from Block 5
#         text_t = torch.from_numpy(self.text_ids[i])             # [SEQ_LEN] int64

#         item = {
#             "image": img_t,
#             "text_ids": text_t,
#             "id": self.ids[i],                 # str
#             "img_path": self.img_paths[i],     # str
#         }
#         if self.labels is not None:
#             li = int(self.labels[i])
#             if li >= 0:
#                 item["label"] = torch.tensor(li, dtype=torch.long)
#         return item

# # ---------- Helper: pick classes by label key ----------
# def classes_for_label_key(label_key: str) -> int:
#     if label_key == "2_way_label":
#         return 2
#     if label_key == "3_way_label":
#         return 3
#     if label_key == "6_way_label":
#         return 6
#     raise ValueError("label_key must be one of {'2_way_label','3_way_label','6_way_label'}")

# # ---------- Helper: fetch labels as arrays given a key ----------
# def get_label_arrays(label_key: str):
#     if label_key == "6_way_label":
#         tr = list(train_df["6_way_label"].values)
#         va = list(valid_df["6_way_label"].values)
#         te = list(test_df["6_way_label"].values)
#     elif label_key == "3_way_label":
#         tr = list(train_df["3_way_label"].values)
#         va = list(valid_df["3_way_label"].values)
#         te = list(test_df["3_way_label"].values)
#     elif label_key == "2_way_label":
#         tr = list(train_df["2_way_label"].values)
#         va = list(valid_df["2_way_label"].values)
#         te = list(test_df["2_way_label"].values)
#     else:
#         raise ValueError("label_key must be one of {'2_way_label','3_way_label','6_way_label'}")
#     return tr, va, te

# # ---------- Subset helper ----------
# def subset_arrays(
#     imgs: List[str],
#     text_pad: np.ndarray,
#     labels: Optional[Union[List[int], np.ndarray]],
#     max_n: Optional[int] = None,         # e.g., 5000
#     frac: Optional[float] = None,        # e.g., 0.25 (ignored if max_n is set)
#     seed: int = 42,
#     stratify: bool = True
# ) -> Tuple[List[str], np.ndarray, Optional[np.ndarray]]:
#     """
#     Return (imgs_sub, text_sub, labels_sub) according to max_n or frac.
#     If both are None, returns inputs unchanged.
#     """
#     N = len(imgs)
#     if max_n is None and (frac is None or frac >= 1.0):
#         return imgs, text_pad, (np.asarray(labels) if labels is not None else None)

#     target = max_n if max_n is not None else int(np.ceil(N * float(frac)))
#     target = max(1, min(target, N))

#     idx_all = np.arange(N)
#     if stratify and labels is not None:
#         y = np.asarray(labels)
#         splitter = StratifiedShuffleSplit(n_splits=1, train_size=target, random_state=seed)
#         sub_idx, _ = next(splitter.split(idx_all.reshape(-1, 1), y))
#         chosen = sub_idx
#     else:
#         rng = np.random.default_rng(seed)
#         chosen = rng.choice(idx_all, size=target, replace=False)

#     chosen = np.sort(chosen)
#     imgs_sub   = [imgs[i] for i in chosen]
#     text_sub   = text_pad[chosen]
#     labels_sub = (np.asarray(labels)[chosen] if labels is not None else None)
#     return imgs_sub, text_sub, labels_sub

# # ---------- Loader builder ----------
# def build_loaders(label_key: str,
#                   max_train: Optional[int] = None, frac_train: Optional[float] = None,
#                   max_valid: Optional[int] = None, frac_valid: Optional[float] = None,
#                   max_test:  Optional[int] = None, frac_test:  Optional[float] = None,
#                   batch_size: int = 32, num_workers: int = 0):
#     tr_lbl, va_lbl, te_lbl = get_label_arrays(label_key)

#     tr_imgs, tr_pad, tr_lbl = subset_arrays(train_imgs, train_pad, tr_lbl, max_n=max_train, frac=frac_train, seed=42, stratify=True)
#     va_imgs, va_pad, va_lbl = subset_arrays(valid_imgs, valid_pad, va_lbl, max_n=max_valid, frac=frac_valid, seed=42, stratify=True)
#     te_imgs, te_pad, te_lbl = subset_arrays(test_imgs,  test_pad,  te_lbl, max_n=max_test,  frac=frac_test,  seed=42, stratify=True)

#     train_loader = DataLoader(MultimodalFakedditDataset(tr_imgs, tr_pad, tr_lbl, ids=None),
#                               batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
#     valid_loader = DataLoader(MultimodalFakedditDataset(va_imgs, va_pad, va_lbl, ids=None),
#                               batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
#     test_loader  = DataLoader(MultimodalFakedditDataset(te_imgs, te_pad, te_lbl, ids=None),
#                               batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

#     return train_loader, valid_loader, test_loader

# # ---------- Model components ----------
# class ConvBNReLU(nn.Module):
#     def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
#         super().__init__()
#         self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
#         self.bn   = nn.BatchNorm2d(out_ch)
#         self.act  = nn.ReLU(inplace=True)
#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))

# class ImageTower(nn.Module):
#     def __init__(self, in_ch=3, feat_dim=256):
#         super().__init__()
#         self.block1 = nn.Sequential(
#             ConvBNReLU(in_ch, 32, k=5, s=2, p=2),   # 112x112
#             ConvBNReLU(32, 32, k=3, s=1, p=1),
#         )
#         self.block2 = nn.Sequential(
#             ConvBNReLU(32, 64, k=3, s=2, p=1),      # 56x56
#             ConvBNReLU(64, 64, k=3, s=1, p=1),
#         )
#         self.block3 = nn.Sequential(
#             ConvBNReLU(64, 128, k=3, s=2, p=1),     # 28x28
#             ConvBNReLU(128, 128, k=3, s=1, p=1),
#         )
#         self.block4 = nn.Sequential(
#             ConvBNReLU(128, 256, k=3, s=2, p=1),    # 14x14
#             ConvBNReLU(256, 256, k=3, s=1, p=1),
#         )
#         self.gap  = nn.AdaptiveAvgPool2d(1)         # -> [B,256,1,1]
#         self.proj = nn.Linear(256, feat_dim)

#     def forward(self, x):
#         x = self.block1(x); x = self.block2(x); x = self.block3(x); x = self.block4(x)
#         x = self.gap(x).flatten(1)
#         x = self.proj(x)
#         return x

# class TextTower(nn.Module):
#     def __init__(self, embedding: nn.Embedding, hidden=128, num_layers=1, bidirectional=True, dropout=0.1):
#         super().__init__()
#         self.embedding = embedding
#         emb_dim = embedding.embedding_dim
#         self.gru = nn.GRU(
#             input_size=emb_dim, hidden_size=hidden, num_layers=num_layers,
#             batch_first=True, bidirectional=bidirectional,
#             dropout=dropout if num_layers > 1 else 0.0,
#         )
#         self.out_dim = hidden * (2 if bidirectional else 1)
#         self.dropout = nn.Dropout(p=dropout)

#     def forward(self, token_ids):
#         x, _ = self.gru(self.embedding(token_ids))   # [B,SEQ_LEN,2H]
#         x = x.mean(dim=1)                            # mean-pool
#         x = self.dropout(x)
#         return x

# class MultiModalClassifier(nn.Module):
#     def __init__(self, embedding_layer: nn.Embedding,
#                  img_feat_dim=256, txt_hidden=128, txt_layers=1, txt_bidir=True,
#                  fusion_hidden=256, num_classes=6, dropout=0.2):
#         super().__init__()
#         self.image_tower = ImageTower(in_ch=3, feat_dim=img_feat_dim)
#         self.text_tower  = TextTower(embedding_layer, hidden=txt_hidden,
#                                      num_layers=txt_layers, bidirectional=txt_bidir, dropout=dropout)
#         fusion_in = img_feat_dim + self.text_tower.out_dim
#         self.fusion = nn.Sequential(
#             nn.Linear(fusion_in, fusion_hidden),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(fusion_hidden, num_classes),
#         )

#     def forward(self, image, text_ids):
#         fi = self.image_tower(image)
#         ft = self.text_tower(text_ids)
#         fused = torch.cat([fi, ft], dim=1)
#         logits = self.fusion(fused)
#         return logits

# # ---------- Model builder ----------
# def build_model(label_key: str,
#                 embedding_layer: nn.Embedding,
#                 img_feat_dim: int = 256,
#                 txt_hidden: int = 128,
#                 txt_layers: int = 1,
#                 txt_bidir: bool = True,
#                 fusion_hidden: int = 256,
#                 dropout: float = 0.2,
#                 device: Optional[torch.device] = None) -> nn.Module:
#     num_classes = classes_for_label_key(label_key)
#     model = MultiModalClassifier(
#         embedding_layer=embedding_layer,
#         img_feat_dim=img_feat_dim,
#         txt_hidden=txt_hidden,
#         txt_layers=txt_layers,
#         txt_bidir=txt_bidir,
#         fusion_hidden=fusion_hidden,
#         num_classes=num_classes,
#         dropout=dropout,
#     )
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     return model.to(device)


# ####################################################################################
# #################################################################################### Training & Evaluation Loop
# ####################################################################################

# def make_criterion(num_classes: int,
#                    class_weights: Optional[torch.Tensor] = None,
#                    device: Optional[torch.device] = None):
#     if class_weights is not None and device is not None:
#         class_weights = class_weights.to(device)
#     return nn.CrossEntropyLoss(weight=class_weights)

# @torch.no_grad()
# def _metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor):
#     preds = logits.argmax(dim=1)
#     acc = (preds == labels).float().mean().item()
#     return preds.cpu().numpy(), acc

# @torch.no_grad()
# def evaluate(model: nn.Module,
#              loader,
#              device: torch.device,
#              max_valid_batches: Optional[int] = None):
#     model.eval()
#     all_preds, all_labels = [], []
#     total_loss, n_batches = 0.0, 0
#     criterion = getattr(model, "_criterion", None)

#     for b_idx, batch in enumerate(loader):
#         if (max_valid_batches is not None) and (b_idx >= max_valid_batches):
#             break

#         imgs = batch["image"].to(device, non_blocking=True)
#         txt  = batch["text_ids"].to(device, non_blocking=True)
#         labels = batch.get("label", None)

#         logits = model(imgs, txt)

#         if labels is None:
#             continue

#         labels = labels.to(device, non_blocking=True)
#         if criterion is not None:
#             loss = criterion(logits, labels)
#             total_loss += loss.item()

#         preds_np, _ = _metrics_from_logits(logits, labels)
#         all_preds.append(preds_np)
#         all_labels.append(labels.detach().cpu().numpy())
#         n_batches += 1

#     if len(all_labels) == 0:
#         return {"loss": None, "acc": None, "f1_macro": None, "report": None, "confusion": None}

#     import numpy as np
#     y_true = np.concatenate(all_labels)
#     y_pred = np.concatenate(all_preds)

#     acc = accuracy_score(y_true, y_pred)
#     f1m = f1_score(y_true, y_pred, average="macro")
#     rep = classification_report(y_true, y_pred, digits=4)
#     cm  = confusion_matrix(y_true, y_pred)

#     avg_loss = (total_loss / max(1, n_batches)) if criterion is not None else None
#     return {"loss": avg_loss, "acc": acc, "f1_macro": f1m, "report": rep, "confusion": cm}

# def train_one_epoch(model: nn.Module,
#                     loader,
#                     optimizer: optim.Optimizer,
#                     device: torch.device,
#                     scaler: Optional[torch.cuda.amp.GradScaler] = None,
#                     max_train_batches: Optional[int] = None,
#                     grad_clip_norm: Optional[float] = 1.0):
#     model.train()
#     total_loss, total_acc, n_batches = 0.0, 0.0, 0
#     criterion = model._criterion

#     for b_idx, batch in enumerate(loader):
#         if (max_train_batches is not None) and (b_idx >= max_train_batches):
#             break

#         imgs = batch["image"].to(device, non_blocking=True)
#         txt  = batch["text_ids"].to(device, non_blocking=True)
#         labels = batch["label"].to(device, non_blocking=True)

#         optimizer.zero_grad(set_to_none=True)

#         if scaler is not None:
#             with torch.cuda.amp.autocast():
#                 logits = model(imgs, txt)
#                 loss = criterion(logits, labels)
#             scaler.scale(loss).backward()
#             if grad_clip_norm is not None:
#                 scaler.unscale_(optimizer)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
#             scaler.step(optimizer)
#             scaler.update()
#         else:
#             logits = model(imgs, txt)
#             loss = criterion(logits, labels)
#             loss.backward()
#             if grad_clip_norm is not None:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
#             optimizer.step()

#         _, acc_b = _metrics_from_logits(logits, labels)
#         total_loss += loss.item()
#         total_acc  += acc_b
#         n_batches  += 1

#     return {"loss": total_loss / max(1, n_batches),
#             "acc":  total_acc / max(1, n_batches),
#             "batches": n_batches}

# def fit(model: nn.Module,
#         train_loader,
#         valid_loader,
#         device: torch.device,
#         epochs: int = 5,
#         lr: float = 1e-3,
#         weight_decay: float = 0.0,
#         class_weights: Optional[torch.Tensor] = None,
#         use_amp: bool = True,
#         max_train_batches: Optional[int] = None,
#         max_valid_batches: Optional[int] = None,
#         ckpt_path: str = "best_multimodal.pt",
#         scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
#         early_stop_patience: Optional[int] = 3):
#     """
#     Train the model and save the best (by valid F1-macro) to ckpt_path.
#     Returns dict with best epoch/F1 and checkpoint path.
#     """
#     num_classes = model.fusion[-1].out_features
#     criterion = make_criterion(num_classes, class_weights=class_weights, device=device)
#     model._criterion = criterion

#     optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#     scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

#     best_f1 = -1.0
#     best_epoch = -1
#     epochs_no_improve = 0

#     for ep in range(1, epochs + 1):
#         t0 = time.time()
#         train_stats = train_one_epoch(model, train_loader, optimizer, device,
#                                       scaler=scaler, max_train_batches=max_train_batches)
#         valid_stats = evaluate(model, valid_loader, device, max_valid_batches=max_valid_batches)

#         if scheduler is not None:
#             if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
#                 val_loss = valid_stats["loss"] if valid_stats["loss"] is not None else train_stats["loss"]
#                 scheduler.step(val_loss)
#             else:
#                 scheduler.step()

#         dt = time.time() - t0
#         v_loss = valid_stats["loss"] if valid_stats["loss"] is not None else float('nan')
#         v_acc  = valid_stats["acc"] if valid_stats["acc"] is not None else float('nan')
#         v_f1   = valid_stats["f1_macro"] if valid_stats["f1_macro"] is not None else float('nan')

#         print(f"[Epoch {ep:02d}] "
#               f"train loss {train_stats['loss']:.4f} acc {train_stats['acc']:.4f} | "
#               f"valid loss {v_loss:.4f} acc {v_acc:.4f} f1 {v_f1:.4f} | "
#               f"{dt:.1f}s")

#         cur_f1 = valid_stats["f1_macro"] if valid_stats["f1_macro"] is not None else -1.0
#         if cur_f1 > best_f1:
#             best_f1 = cur_f1
#             best_epoch = ep
#             torch.save({"model": model.state_dict(),
#                         "epoch": ep,
#                         "f1_macro": cur_f1}, ckpt_path)
#             print(f"  ↳ Saved new best to {ckpt_path} (F1={cur_f1:.4f})")
#             epochs_no_improve = 0
#         else:
#             epochs_no_improve += 1
#             if (early_stop_patience is not None) and (epochs_no_improve >= early_stop_patience):
#                 print(f"  ↳ Early stopping at epoch {ep} (no improve {epochs_no_improve} epochs).")
#                 break

#     print(f"Best epoch {best_epoch} with F1={best_f1:.4f}")
#     return {"best_epoch": best_epoch, "best_f1_macro": best_f1, "ckpt_path": ckpt_path}


# ####################################################################################
# #################################################################################### Runner: Configure, Train & Save Model
# ####################################################################################

# # -------- CONFIG: set what you want here --------
# CONFIG = {
#     # Label granularity: one of {"2_way_label","3_way_label","6_way_label"}
#     "label_key": "2_way_label",

#     # Subset sizes (use None to take full split). You can instead use frac_* in build_loaders call if desired.
#     "max_train": 1000,
#     "max_valid": 100,
#     "max_test":  100,

#     # Data loader
#     "batch_size": 32,
#     "num_workers": 0,

#     # Model hyperparams
#     "img_feat_dim": 256,
#     "txt_hidden": 128,
#     "txt_layers": 1,
#     "txt_bidir": True,
#     "fusion_hidden": 256,
#     "dropout": 0.2,

#     # Training hyperparams
#     "epochs": 3,
#     "lr": 1e-3,
#     "weight_decay": 1e-4,
#     "use_amp": True,
#     "max_train_batches": None,   # e.g., 100 to cap per epoch
#     "max_valid_batches": None,   # e.g., 20

#     # Optional class weights (set to None to disable). Example shows how to compute below.
#     "class_weights": None,

#     # Checkpoint path (per label granularity)
#     "ckpt_path": None,  # if None, will auto-name below
# }

# # ---- derive device & checkpoint path ----
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if CONFIG["ckpt_path"] is None:
#     CONFIG["ckpt_path"] = f"best_mm_{CONFIG['label_key'].replace('_way_label','')}way.pt"

# print("Using device:", device)
# print("Checkpoint:", CONFIG["ckpt_path"])

# # ---- Build loaders for chosen label granularity ----
# train_loader, valid_loader, test_loader = build_loaders(
#     label_key=CONFIG["label_key"],
#     max_train=CONFIG["max_train"], frac_train=None,
#     max_valid=CONFIG["max_valid"], frac_valid=None,
#     max_test=CONFIG["max_test"],   frac_test=None,
#     batch_size=CONFIG["batch_size"],
#     num_workers=CONFIG["num_workers"],
# )
# print("batches -> train:", len(train_loader), "| valid:", len(valid_loader), "| test:", len(test_loader))

# # ---- Build model for chosen label granularity ----
# model = build_model(
#     label_key=CONFIG["label_key"],
#     embedding_layer=embedding_layer,           # from Block 4
#     img_feat_dim=CONFIG["img_feat_dim"],
#     txt_hidden=CONFIG["txt_hidden"],
#     txt_layers=CONFIG["txt_layers"],
#     txt_bidir=CONFIG["txt_bidir"],
#     fusion_hidden=CONFIG["fusion_hidden"],
#     dropout=CONFIG["dropout"],
#     device=device,
# )

# # ---- Optional: compute class weights automatically (commented by default) ----
# # import numpy as np
# # tr_labels_list, _, _ = get_label_arrays(CONFIG["label_key"])
# # # match the exact subset used in train_loader:
# # import itertools
# # train_labels_used = list(itertools.chain.from_iterable([[b["label"].numpy()] for b in train_loader]))  # labels per item
# # # If the above produces arrays, flatten:
# # train_labels_used = np.array(train_labels_used).reshape(-1)
# # counts = np.bincount(train_labels_used.astype(int))
# # inv = 1.0 / np.maximum(counts, 1)
# # CONFIG["class_weights"] = torch.tensor(inv / inv.sum() * len(counts), dtype=torch.float32)
# # print("Class weights:", CONFIG["class_weights"])

# # ---- Train ----
# train_summary = fit(
#     model=model,
#     train_loader=train_loader,
#     valid_loader=valid_loader,
#     device=device,
#     epochs=CONFIG["epochs"],
#     lr=CONFIG["lr"],
#     weight_decay=CONFIG["weight_decay"],
#     class_weights=CONFIG["class_weights"],
#     use_amp=CONFIG["use_amp"],
#     max_train_batches=CONFIG["max_train_batches"],
#     max_valid_batches=CONFIG["max_valid_batches"],
#     ckpt_path=CONFIG["ckpt_path"],
#     scheduler=None,
#     early_stop_patience=3,
# )

# # ---- Load BEST checkpoint & keep the trained model variable ----
# if os.path.isfile(train_summary["ckpt_path"]):
#     ckpt = torch.load(train_summary["ckpt_path"], map_location=device)
#     model.load_state_dict(ckpt["model"])
#     print(f"Loaded best checkpoint from epoch {ckpt.get('epoch')} (F1={ckpt.get('f1_macro')})")

# trained_model = model  # <- keep this for later inference/evaluation

# # ---- Evaluate on TEST now (optional) ----
# test_stats = evaluate(trained_model, test_loader, device)
# print("\n=== TEST METRICS ===")
# print("Acc:      ", test_stats["acc"])
# print("F1-macro: ", test_stats["f1_macro"])
# print("Confusion:\n", test_stats["confusion"])
# print("\nClassification report:\n", test_stats["report"])

# # ---- Convenience: keep handy objects for later use ----
# RUNTIME = {
#     "device": device,
#     "config": CONFIG,
#     "train_summary": train_summary,
#     "loaders": {
#         "train": train_loader,
#         "valid": valid_loader,
#         "test":  test_loader,
#     },
#     "ckpt_path": CONFIG["ckpt_path"],
#     "model": trained_model,
# }

# print("\nSaved references:")
# print(" - trained_model (nn.Module)")
# print(" - RUNTIME['loaders']['train'|'valid'|'test']")
# print(" - RUNTIME['ckpt_path'] =", RUNTIME['ckpt_path'])

# ####################################################################################
# #################################################################################### 
# ####################################################################################


# print("<====== Done =======>")
