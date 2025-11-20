#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#########################################
########################## Text preprocessing (tokenizer-ready)
#########################################

import os
import json
import re
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

# Try NLTK stopwords + lemmatizer, but fall back gracefully if data isn't present
USE_NLTK = True
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    _ = stopwords.words("english")  # will raise LookupError if not downloaded
    _ = WordNetLemmatizer()
except Exception:
    USE_NLTK = False

# Keras Tokenizer (lightweight)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#########################################
########################## Config (edit here)
#########################################

DATA_ROOT    = "../project_dataset/fakeddit_data/multimodal_only_samples"
PREPARED_DIR = "prepared_data"
INPUT_TRAIN  = "train_final.csv"
INPUT_VALID  = "valid_final.csv"
INPUT_TEST   = "test_final.csv"

# Label column & task size
LABEL_COL    = "6_way_label"   # "2_way_label" | "3_way_label" | "6_way_label" <========= CHANGE THIS

# Row caps (-1 = use all)
TRAIN_CAP    = -1
VALID_CAP    = -1
TEST_CAP     = -1

# Tokenizer / sequence settings
VOCAB_CAP    = 120_000
SEQ_LEN      = 15
LOWERCASE    = True

# Output dir for preprocessed text items                                       <========= CHANGE THIS
OUT_SUBDIR   = "text_proc_6way"

#########################################
########################## Utilities
#########################################

def cap_df(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    if cap is None or cap < 0 or cap >= len(df):
        return df
    return df.iloc[:cap].copy()

def basic_clean(s: Optional[str]) -> str:
    """Lowercase, drop URLs, keep letters only, collapse spaces."""
    if not isinstance(s, str):
        return ""
    x = s.strip()
    if LOWERCASE:
        x = x.lower()
    x = re.sub(r"https?://\S+|www\.\S+", " ", x)          # remove urls
    x = re.sub(r"[^a-zA-Z\s]", " ", x)                    # keep letters only
    x = re.sub(r"\s+", " ", x).strip()                   # collapse spaces
    return x

def add_nltk_clean(tokens: List[str]) -> List[str]:
    """Optional: stopword removal + lemmatization (if NLTK available)."""
    if not USE_NLTK:
        return tokens
    sw = set(stopwords.words("english"))
    lem = WordNetLemmatizer()
    keep: List[str] = []
    for t in tokens:
        if not t or t in sw:
            continue
        keep.append(lem.lemmatize(t))
    return keep

def to_tokens(s: str) -> List[str]:
    # split on non-letters (text already letters-only)
    toks = re.split(r"\s+", s)
    toks = [t for t in toks if t]
    toks = add_nltk_clean(toks)
    return toks

def clean_concat_row(row: pd.Series) -> str:
    """Use clean_title + title as in your previous work."""
    ct = row.get("clean_title", "")
    tt = row.get("title", "")
    s = f"{ct} {tt}"
    return basic_clean(s)

def load_csvs(base: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(base / PREPARED_DIR / INPUT_TRAIN, dtype=str)
    valid = pd.read_csv(base / PREPARED_DIR / INPUT_VALID, dtype=str)
    test  = pd.read_csv(base / PREPARED_DIR / INPUT_TEST, dtype=str)
    return train, valid, test

#########################################
########################## Main
#########################################

def main():
    base = Path(DATA_ROOT)
    out_dir = base / PREPARED_DIR / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load CSVs and cap
    df_tr, df_va, df_te = load_csvs(base)
    df_tr = cap_df(df_tr, TRAIN_CAP)
    df_va = cap_df(df_va, VALID_CAP)
    df_te = cap_df(df_te, TEST_CAP)

    # Drop rows missing both titles
    def has_any_title(df: pd.DataFrame) -> pd.DataFrame:
        mask = (df.get("clean_title", "").fillna("") != "") | (df.get("title", "").fillna("") != "")
        return df[mask].copy()

    df_tr = has_any_title(df_tr)
    df_va = has_any_title(df_va)
    df_te = has_any_title(df_te)

    # Clean text
    tr_text = df_tr.apply(clean_concat_row, axis=1).tolist()
    va_text = df_va.apply(clean_concat_row, axis=1).tolist()
    te_text = df_te.apply(clean_concat_row, axis=1).tolist()

    # Tokenize via regex + (optional) stopword+lemma BEFORE fitting tokenizer,
    # so the tokenizer sees your final tokens distribution.
    tr_tokens = [" ".join(to_tokens(s)) for s in tr_text]
    va_tokens = [" ".join(to_tokens(s)) for s in va_text]
    te_tokens = [" ".join(to_tokens(s)) for s in te_text]

    # Fit a single tokenizer on all splits (as you did previously)
    tok = Tokenizer(num_words=VOCAB_CAP, oov_token=None, lower=False, split=" ")
    tok.fit_on_texts(tr_tokens + va_tokens + te_tokens)

    # Convert to sequences (+ pad/trunc)
    tr_seq = tok.texts_to_sequences(tr_tokens)
    va_seq = tok.texts_to_sequences(va_tokens)
    te_seq = tok.texts_to_sequences(te_tokens)

    tr_seq = pad_sequences(tr_seq, maxlen=SEQ_LEN, padding="post", truncating="post")
    va_seq = pad_sequences(va_seq, maxlen=SEQ_LEN, padding="post", truncating="post")
    te_seq = pad_sequences(te_seq, maxlen=SEQ_LEN, padding="post", truncating="post")

    # Save aligned CSV “used” slices for the trainer (ensures row-alignment with sequences)
    df_tr.to_csv(out_dir / "train_used.csv", index=False)
    df_va.to_csv(out_dir / "valid_used.csv", index=False)
    df_te.to_csv(out_dir / "test_used.csv",  index=False)

    # Save sequences
    np.save(out_dir / "train_seq.npy", tr_seq)
    np.save(out_dir / "valid_seq.npy", va_seq)
    np.save(out_dir / "test_seq.npy",  te_seq)

    # Save tokenizer (JSON)
    tok_json = tok.to_json()
    with open(out_dir / "tokenizer.json", "w") as f:
        f.write(tok_json)

    # Meta
    meta = {
        "vocab_cap": VOCAB_CAP,
        "actual_vocab_size": min(VOCAB_CAP, len(tok.word_index)),
        "seq_len": SEQ_LEN,
        "label_col": LABEL_COL,
        "n_train": int(len(df_tr)),
        "n_valid": int(len(df_va)),
        "n_test":  int(len(df_te)),
        "use_nltk": USE_NLTK,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[ok] saved sequences and tokenizer to {out_dir}")

if __name__ == "__main__":
    main()