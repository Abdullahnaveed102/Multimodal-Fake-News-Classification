#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ####################################################################################
# #################################################################################### Multimodal CNN (text + images): imports
# ####################################################################################
import os, re, time, math
from typing import List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# ####################################################################################
# #################################################################################### NLTK bootstrap (quiet, one-time)
# ####################################################################################
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

# ####################################################################################
# #################################################################################### Load datasets + Set up paths
# ####################################################################################
BASE_ROOT = "/home/abdullahnaveed/blue_li_abdullah/pattern_project/project_dataset/fakeddit_data/multimodal_only_samples"

TSV_TRAIN = os.path.join(BASE_ROOT, "multimodal_train.tsv")
TSV_VALID = os.path.join(BASE_ROOT, "multimodal_validate.tsv")
TSV_TEST  = os.path.join(BASE_ROOT, "multimodal_test_public.tsv")

IMG_DIRS = {
    "train": os.path.join(BASE_ROOT, "train_images"),
    "valid": os.path.join(BASE_ROOT, "valid_images"),
    "test":  os.path.join(BASE_ROOT, "test_images"),
}

# keep only what we need (add author/domain/time + labels)
use_cols = [
    "id", "title",
    "author", "domain", "created_utc",
    "upvote_ratio", "num_comments", "score",
    "2_way_label", "3_way_label", "6_way_label",
]

train_all = pd.read_csv(TSV_TRAIN, sep="\t", usecols=lambda c: True)
valid_all = pd.read_csv(TSV_VALID, sep="\t", usecols=lambda c: True)
test_all  = pd.read_csv(TSV_TEST,  sep="\t", usecols=lambda c: True)

train_all = train_all[[c for c in use_cols if c in train_all.columns]]
valid_all = valid_all[[c for c in use_cols if c in valid_all.columns]]
test_all  = test_all [[c for c in use_cols if c in test_all.columns]]

# require title (text)
train_df = train_all.dropna(subset=["title"]).copy()
valid_df = valid_all.dropna(subset=["title"]).copy()
test_df  = test_all.dropna(subset=["title"]).copy()

# labels to int
for lab in ["2_way_label", "3_way_label", "6_way_label"]:
    for df in (train_df, valid_df, test_df):
        if lab in df.columns:
            df[lab] = df[lab].astype(int)

# build image paths and keep only rows whose image exists
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

# ####################################################################################
# #################################################################################### Text preprocessing → Tokenization → Padding
# ####################################################################################
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

# ####################################################################################
# #################################################################################### Word Embeddings & Vocabulary (GloVe unchanged)
# ####################################################################################
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

# ####################################################################################
# #################################################################################### Image Preprocessing & Utilities
# ####################################################################################
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

# ####################################################################################
# #################################################################################### Tabular feature engineering (no hasImage)
# ####################################################################################
from dataclasses import dataclass

@dataclass
class TabularConfig:
    numeric_cols: tuple = ("upvote_ratio", "num_comments", "score")  # counts will be log1p
    cat_domain_topk: int = 200
    cat_author_topk: int = 500

class TabularEncoder:
    def __init__(self, cfg: TabularConfig = TabularConfig()):
        self.cfg = cfg
        self.num_means = None
        self.num_stds  = None
        self.dom2id = None
        self.auth2id = None

    @staticmethod
    def _to_dt(series):
        if series is None:
            return pd.Series([], dtype="datetime64[ns]")
        if np.issubdtype(series.dtype, np.number):
            return pd.to_datetime(series, unit="s", errors="coerce")
        return pd.to_datetime(series, errors="coerce")

    def fit(self, df: pd.DataFrame):
        # --- categorical vocabularies (topK + UNK=0) ---
        def topk_map(s: pd.Series, k: int):
            vc = s.fillna("__NA__").astype(str).value_counts()
            vocab = ["__UNK__"] + vc.index[:k].tolist()
            return {w: i for i, w in enumerate(vocab)}
        self.dom2id  = topk_map(df.get("domain",  pd.Series([])), self.cfg.cat_domain_topk)
        self.auth2id = topk_map(df.get("author",  pd.Series([])), self.cfg.cat_author_topk)

        X = self._numeric_matrix(df)
        self.num_means = X.mean(axis=0)
        self.num_stds  = X.std(axis=0) + 1e-8
        return self

    def _numeric_matrix(self, df: pd.DataFrame) -> np.ndarray:
        d = df.copy()

        # upvote_ratio (0..1)
        d["upvote_ratio"] = pd.to_numeric(d.get("upvote_ratio", 0.0), errors="coerce").fillna(0.0)

        # counts -> log1p
        for c in ("num_comments", "score"):
            if c in d.columns:
                d[c] = np.log1p(pd.to_numeric(d[c], errors="coerce").fillna(0.0))
            else:
                d[c] = 0.0

        # time features from created_utc (hour, day-of-week cyclical)
        dt = self._to_dt(d.get("created_utc", pd.Series([np.nan]*len(d))))
        hour = dt.dt.hour.fillna(0).astype(int)
        dow  = dt.dt.dayofweek.fillna(0).astype(int)
        d["hour_sin"] = np.sin(2*np.pi*hour/24); d["hour_cos"] = np.cos(2*np.pi*hour/24)
        d["dow_sin"]  = np.sin(2*np.pi*dow/7);   d["dow_cos"]  = np.cos(2*np.pi*dow/7)

        num_feats = ["upvote_ratio", "num_comments", "score", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
        return d[num_feats].astype(float).to_numpy(dtype=np.float32)

    def _cat_ids(self, s: pd.Series, mp: dict) -> np.ndarray:
        if s is None:
            return np.zeros((0,), dtype=np.int64)
        x = s.fillna("__NA__").astype(str).map(mp).fillna(0).astype(int)
        return x.to_numpy(dtype=np.int64)

    def transform(self, df: pd.DataFrame):
        Xnum = self._numeric_matrix(df)
        Xnum = (Xnum - self.num_means) / self.num_stds
        x_dom  = self._cat_ids(df.get("domain"), self.dom2id)
        x_auth = self._cat_ids(df.get("author"), self.auth2id)
        return Xnum.astype(np.float32), x_dom, x_auth

    @property
    def n_num(self): return 7  # 3 base + 4 cyclical time
    @property
    def n_dom(self): return len(self.dom2id or {})
    @property
    def n_auth(self): return len(self.auth2id or {})

# fit on train, transform all
tab_enc = TabularEncoder().fit(train_df)
train_num, train_dom, train_auth = tab_enc.transform(train_df)
valid_num, valid_dom, valid_auth = tab_enc.transform(valid_df)
test_num,  test_dom,  test_auth  = tab_enc.transform(test_df)

# ####################################################################################
# #################################################################################### Dataset & DataLoaders (collate-safe; with tabular)
# ####################################################################################
class MultimodalFakedditDataset(Dataset):
    """
    Returns:
      image:   FloatTensor [3,224,224]
      text_ids:LongTensor  [SEQ_LEN]
      x_num:   FloatTensor [F_num]
      x_dom:   LongTensor  []
      x_auth:  LongTensor  []
      label:   LongTensor  [] (optional)
      id, img_path: str (for reference)
    """
    def __init__(self, img_paths: List[str], text_ids: np.ndarray, labels: Optional[Union[List[int], np.ndarray]],
                 x_num: np.ndarray, x_dom: np.ndarray, x_auth: np.ndarray, ids=None):
        N = len(img_paths)
        assert N == len(text_ids) == len(x_num) == len(x_dom) == len(x_auth)
        self.img_paths = img_paths
        self.text_ids  = np.asarray(text_ids, dtype=np.int64)
        self.labels    = (np.asarray(labels, dtype=np.int64) if labels is not None else None)
        self.x_num     = np.asarray(x_num, dtype=np.float32)
        self.x_dom     = np.asarray(x_dom, dtype=np.int64)
        self.x_auth    = np.asarray(x_auth, dtype=np.int64)
        self.ids       = [("" if ids is None or ids[i] is None else str(ids[i])) for i in range(N)]

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, i):
        item = {
            "image":    load_and_process_image(self.img_paths[i]),
            "text_ids": torch.from_numpy(self.text_ids[i]),
            "x_num":    torch.from_numpy(self.x_num[i]),
            "x_dom":    torch.tensor(int(self.x_dom[i]), dtype=torch.long),
            "x_auth":   torch.tensor(int(self.x_auth[i]), dtype=torch.long),
            "id":       self.ids[i],
            "img_path": self.img_paths[i],
        }
        if self.labels is not None:
            item["label"] = torch.tensor(int(self.labels[i]), dtype=torch.long)
        return item

def subset_with_indices(imgs, text_pad, labels, xnum, xdom, xauth,
                        max_n=None, frac=None, seed=42, stratify=True):
    N = len(imgs)
    if max_n is None and (frac is None or float(frac) >= 1.0):
        idx = np.arange(N)
    else:
        target = max_n if max_n is not None else int(np.ceil(N * float(frac)))
        target = max(1, min(target, N))
        if stratify and labels is not None:
            y = np.asarray(labels)
            splitter = StratifiedShuffleSplit(n_splits=1, train_size=target, random_state=seed)
            idx, _ = next(splitter.split(np.arange(N).reshape(-1,1), y))
        else:
            rng = np.random.default_rng(seed); idx = np.sort(rng.choice(np.arange(N), size=target, replace=False))
    idx = np.asarray(idx)
    imgs_sub = [imgs[i] for i in idx]
    return (imgs_sub, text_pad[idx],
            (np.asarray(labels)[idx] if labels is not None else None),
            xnum[idx], xdom[idx], xauth[idx])

def get_label_arrays(label_key: str):
    if label_key == "6_way_label":
        return list(train_df["6_way_label"].values), list(valid_df["6_way_label"].values), list(test_df["6_way_label"].values)
    if label_key == "3_way_label":
        return list(train_df["3_way_label"].values), list(valid_df["3_way_label"].values), list(test_df["3_way_label"].values)
    if label_key == "2_way_label":
        return list(train_df["2_way_label"].values), list(valid_df["2_way_label"].values), list(test_df["2_way_label"].values)
    raise ValueError("label_key must be one of {'2_way_label','3_way_label','6_way_label'}")

def classes_for_label_key(label_key: str) -> int:
    return {"2_way_label":2, "3_way_label":3, "6_way_label":6}[label_key]

def build_loaders(label_key: str,
                  max_train: Optional[int] = None, frac_train: Optional[float] = None,
                  max_valid: Optional[int] = None, frac_valid: Optional[float] = None,
                  max_test:  Optional[int] = None, frac_test:  Optional[float] = None,
                  batch_size: int = 32, num_workers: int = 0):

    tr_lbl, va_lbl, te_lbl = get_label_arrays(label_key)

    tr = subset_with_indices(train_imgs, train_pad, tr_lbl, train_num, train_dom, train_auth,
                             max_n=max_train, frac=frac_train, seed=42, stratify=True)
    va = subset_with_indices(valid_imgs, valid_pad, va_lbl, valid_num, valid_dom, valid_auth,
                             max_n=max_valid, frac=frac_valid, seed=42, stratify=True)
    te = subset_with_indices(test_imgs,  test_pad,  te_lbl, test_num,  test_dom,  test_auth,
                             max_n=max_test,  frac=frac_test,  seed=42, stratify=True)

    def mk_loader(pack, shuffle):
        imgs, pad, lbl, xnum, xdom, xauth = pack
        ds = MultimodalFakedditDataset(imgs, pad, lbl, xnum, xdom, xauth, ids=None)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    return mk_loader(tr, True), mk_loader(va, False), mk_loader(te, False)

# ####################################################################################
# #################################################################################### Model
# ####################################################################################
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
        self.block2 = nn.Sequential(ConvBNReLU(32, 64, s=2),               ConvBNReLU(64, 64))
        self.block3 = nn.Sequential(ConvBNReLU(64, 128, s=2),              ConvBNReLU(128, 128))
        self.block4 = nn.Sequential(ConvBNReLU(128, 256, s=2),             ConvBNReLU(256, 256))
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
                 fusion_hidden=256, num_classes=6, dropout=0.2,
                 n_num: int = 7, n_dom: int = 1, n_auth: int = 1,
                 dom_emb_dim: int = 16, auth_emb_dim: int = 32, num_mlp_dim: int = 32):
        super().__init__()
        self.image_tower = ImageTower(3, img_feat_dim)
        self.text_tower  = TextTower(embedding_layer, hidden=txt_hidden,
                                     num_layers=txt_layers, bidirectional=txt_bidir, dropout=dropout)

        # ----- tabular pieces -----
        self.num_mlp = nn.Sequential(
            nn.Linear(n_num, num_mlp_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        ) if n_num > 0 else None

        self.dom_emb  = nn.Embedding(max(n_dom,1),  dom_emb_dim)  if n_dom  > 0 else None
        self.auth_emb = nn.Embedding(max(n_auth,1), auth_emb_dim) if n_auth > 0 else None

        fusion_in = img_feat_dim + self.text_tower.out_dim
        if self.num_mlp is not None: fusion_in += num_mlp_dim
        if self.dom_emb  is not None: fusion_in += dom_emb_dim
        if self.auth_emb is not None: fusion_in += auth_emb_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes),
        )

    def forward(self, image, text_ids, x_num=None, x_dom=None, x_auth=None):
        parts = [self.image_tower(image), self.text_tower(text_ids)]
        if self.num_mlp is not None and x_num is not None:  parts.append(self.num_mlp(x_num))
        if self.dom_emb  is not None and x_dom  is not None: parts.append(self.dom_emb(x_dom))
        if self.auth_emb is not None and x_auth is not None: parts.append(self.auth_emb(x_auth))
        return self.fusion(torch.cat(parts, dim=1))

def build_model(label_key: str, embedding_layer: nn.Embedding,
                img_feat_dim=256, txt_hidden=128, txt_layers=1, txt_bidir=True,
                fusion_hidden=256, dropout=0.2, device: Optional[torch.device]=None):
    num_classes = classes_for_label_key(label_key)
    model = MultiModalClassifier(
        embedding_layer=embedding_layer,
        img_feat_dim=img_feat_dim, txt_hidden=txt_hidden, txt_layers=txt_layers, txt_bidir=txt_bidir,
        fusion_hidden=fusion_hidden, num_classes=num_classes, dropout=dropout,
        n_num=tab_enc.n_num, n_dom=tab_enc.n_dom, n_auth=tab_enc.n_auth,
    )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)

# ####################################################################################
# #################################################################################### Training & Evaluation Loop
# ####################################################################################
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
        xnum = batch["x_num"].to(device, non_blocking=True)
        xdom = batch["x_dom"].to(device, non_blocking=True)
        xaut = batch["x_auth"].to(device, non_blocking=True)
        labels = batch.get("label", None)

        logits = model(imgs, txt, xnum, xdom, xaut)

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
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)          # safe
    rep = classification_report(y_true, y_pred, digits=4, zero_division=0)    # safe
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
        xnum = batch["x_num"].to(device, non_blocking=True)
        xdom = batch["x_dom"].to(device, non_blocking=True)
        xaut = batch["x_auth"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits = model(imgs, txt, xnum, xdom, xaut)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer); scaler.update()
        else:
            logits = model(imgs, txt, xnum, xdom, xaut)
            loss = criterion(logits, labels)
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        _, acc_b = _metrics_from_logits(logits, labels)
        total_loss += loss.item(); total_acc += acc_b; n_batches += 1

    return {"loss": total_loss / max(1, n_batches),
            "acc":  total_acc  / max(1, n_batches),
            "batches": n_batches}

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
        train_stats = train_one_epoch(model, train_loader, optimizer, device, scaler=scaler,
                                      max_train_batches=max_train_batches)
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
                print(f"  ↳ Early stopping at epoch {ep} (no improve {epochs_no_improve} epochs).")
                break

    print(f"Best epoch {best_epoch} with F1={best_f1:.4f}")
    return {"best_epoch": best_epoch, "best_f1_macro": best_f1, "ckpt_path": ckpt_path}

# ####################################################################################
# #################################################################################### Runner: Configure, Train & Save Model
# ####################################################################################
CONFIG = {
    "label_key": "2_way_label",
    "max_train": 1000, "max_valid": 100, "max_test": 100,
    "batch_size": 32, "num_workers": 0,
    "img_feat_dim": 256,
    "txt_hidden": 128, "txt_layers": 1, "txt_bidir": True,
    "fusion_hidden": 256, "dropout": 0.2,
    "epochs": 3, "lr": 1e-3, "weight_decay": 1e-4,
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

# Optional: compute class weights from the actual train batches
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
