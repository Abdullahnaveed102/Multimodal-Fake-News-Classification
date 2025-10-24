#!/usr/bin/env python3
# ============================================================
# Preprocessing (relative paths, no image preprocessing)
# - Prefer all paired (image+text) rows first, then fill with text-only up to cap
# - Adds: hasImage ("TRUE"/"FALSE"), image_path (relative or "")
# - Saves train_final/valid_final/test_final under DATA_ROOT/OUT_DIR_NAME
# ============================================================

import os
import json
import random
from pathlib import Path
from typing import Optional, List

import pandas as pd

##############################################
################################## User config
##############################################
# Point this to your dataset directory (relative or absolute)
DATA_ROOT = "../../project_dataset/fakeddit_data/multimodal_only_samples"

# Original big TSVs (relative to DATA_ROOT)
TRAIN_TSV = "multimodal_train.tsv"
VALID_TSV = "multimodal_validate.tsv"
TEST_TSV  = "multimodal_test_public.tsv"

# Clean sub-CSV (paired subset with downloaded images)
TRAIN_SUB_CLEAN = "train_subset_clean.csv"
VALID_SUB_CLEAN = "valid_subset_clean.csv"
TEST_SUB_CLEAN  = "test_subset_clean.csv"

# Image folders (relative to DATA_ROOT); image files are id.jpg
TRAIN_IMG_DIR = "train_images"
VALID_IMG_DIR = "valid_images"
TEST_IMG_DIR  = "test_images"

# Output folder (relative to DATA_ROOT)
OUT_DIR_NAME = "prepared_data"

# Caps: process first N rows from big TSVs (per split)
TEXT_CAP_TRAIN = 100000
TEXT_CAP_VALID = 10000
TEXT_CAP_TEST  = 10000

# Reproducibility
SEED = 42

# Keep columns in the SAME order as clean CSV
KEEP_COLS = [
    "author", "clean_title", "created_utc", "domain", "hasImage", "id", "image_url",
    "linked_submission_id", "num_comments", "score", "subreddit", "title", "upvote_ratio",
    "2_way_label", "3_way_label", "6_way_label"
]

##############################################
################################## Helpers
##############################################
def _read_tsv(root: Path, fname: str, cap: Optional[int]) -> pd.DataFrame:
    p = root / fname
    if not p.exists():
        raise FileNotFoundError(f"Missing TSV: {p}")
    df = pd.read_csv(p, sep="\t", dtype=str)
    if cap is not None:
        df = df.iloc[:cap].copy()
    return df

def _read_csv(root: Path, fname: str) -> pd.DataFrame:
    p = root / fname
    if not p.exists():
        raise FileNotFoundError(f"Missing CSV: {p}")
    return pd.read_csv(p, dtype=str)

def _attach_flags_and_paths(df_all: pd.DataFrame, sub_clean: pd.DataFrame, img_dir_rel: str, root: Path) -> pd.DataFrame:
    """
    Adds hasImage (TRUE/FALSE) and image_path (relative) for rows whose id.jpg exists AND id is in sub_clean.
    Preserves original row order from df_all.
    """
    df = df_all.copy()
    df["id"] = df["id"].astype(str)

    sub_ids = set(map(str, sub_clean["id"].astype(str).tolist()))
    img_dir = root / img_dir_rel

    has_list: List[str] = []
    path_list: List[str] = []
    for _id in df["id"].tolist():
        rel_path = f"{img_dir_rel}/{_id}.jpg"
        if (_id in sub_ids) and (img_dir / f"{_id}.jpg").exists():
            has_list.append("TRUE")
            path_list.append(rel_path)
        else:
            has_list.append("FALSE")
            path_list.append("")
    df["hasImage"] = has_list
    df["image_path"] = path_list
    return df

def _prefer_paired_then_fill(df_with_flags: pd.DataFrame, cap: int) -> pd.DataFrame:
    """
    Return up to 'cap' rows, preferring hasImage==TRUE first, then filling with text-only.
    Order within each group (paired/textonly) follows original df order.
    """
    paired = df_with_flags[df_with_flags["hasImage"] == "TRUE"]
    textonly = df_with_flags[df_with_flags["hasImage"] == "FALSE"]

    n_paired = min(len(paired), cap)
    paired_sel = paired.iloc[:n_paired]
    remaining = cap - n_paired

    if remaining > 0:
        filler = textonly.iloc[:remaining]
        out = pd.concat([paired_sel, filler], ignore_index=True)
    else:
        out = paired_sel

    return out

def _finalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Start with schema order, then add extra 'image_path' & 'split' at the end.
    Deduplicate while preserving order.
    """
    cols = [c for c in KEEP_COLS if c in df.columns]
    if "image_path" in df.columns:
        cols.append("image_path")
    if "split" in df.columns:
        cols.append("split")

    seen, out = set(), []
    for c in cols:
        if c not in seen:
            out.append(c); seen.add(c)
    return df[out]

##############################################
################################## Main
##############################################
def main():
    random.seed(SEED)
    root = Path(DATA_ROOT)
    out_dir = root / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------ Load sources with caps ------------------
    df_train_all = _read_tsv(root, TRAIN_TSV, TEXT_CAP_TRAIN)
    df_valid_all = _read_tsv(root, VALID_TSV, TEXT_CAP_VALID)
    df_test_all  = _read_tsv(root, TEST_TSV,  TEXT_CAP_TEST)

    df_train_sub = _read_csv(root, TRAIN_SUB_CLEAN)
    df_valid_sub = _read_csv(root, VALID_SUB_CLEAN)
    df_test_sub  = _read_csv(root, TEST_SUB_CLEAN)

    # --------- Add hasImage + relative image_path to each split ---------
    train_flags = _attach_flags_and_paths(df_train_all, df_train_sub, TRAIN_IMG_DIR, root)
    valid_flags = _attach_flags_and_paths(df_valid_all, df_valid_sub, VALID_IMG_DIR, root)
    test_flags  = _attach_flags_and_paths(df_test_all,  df_test_sub,  TEST_IMG_DIR,  root)

    # --------------- Prefer paired then fill to the requested cap ---------------
    # Train uses TEXT_CAP_TRAIN rows max, preferring all images first
    train_final = _prefer_paired_then_fill(train_flags, cap=TEXT_CAP_TRAIN)

    # Valid uses TEXT_CAP_VALID rows max, preferring all images first
    valid_final = _prefer_paired_then_fill(valid_flags, cap=TEXT_CAP_VALID)

    # Test uses TEXT_CAP_TEST rows max, preferring all images first
    test_final  = _prefer_paired_then_fill(test_flags,  cap=TEXT_CAP_TEST)

    # ------------------ Mark split and finalize columns ------------------
    train_final["split"] = "train"
    valid_final["split"] = "valid"
    test_final["split"]  = "test"

    train_out = _finalize_columns(train_final)
    valid_out = _finalize_columns(valid_final)
    test_out  = _finalize_columns(test_final)

    # ------------------ Save CSVs and stats ------------------
    paths = {
        "train_csv": out_dir / "train_final.csv",
        "valid_csv": out_dir / "valid_final.csv",
        "test_csv":  out_dir / "test_final.csv",
        "stats":     out_dir / "stats.json",
    }
    train_out.to_csv(paths["train_csv"], index=False)
    valid_out.to_csv(paths["valid_csv"], index=False)
    test_out.to_csv(paths["test_csv"], index=False)

    stats = {
        "caps": {
            "TEXT_CAP_TRAIN": TEXT_CAP_TRAIN,
            "TEXT_CAP_VALID": TEXT_CAP_VALID,
            "TEXT_CAP_TEST": TEXT_CAP_TEST
        },
        "counts": {
            "train_total": int(len(train_out)),
            "train_paired": int((train_out["hasImage"] == "TRUE").sum()),
            "train_textonly": int((train_out["hasImage"] == "FALSE").sum()),
            "valid_total": int(len(valid_out)),
            "valid_paired": int((valid_out["hasImage"] == "TRUE").sum()),
            "valid_textonly": int((valid_out["hasImage"] == "FALSE").sum()),
            "test_total": int(len(test_out)),
            "test_paired": int((test_out["hasImage"] == "TRUE").sum()),
            "test_textonly": int((test_out["hasImage"] == "FALSE").sum()),
        },
        "notes": "image_path is relative (e.g., 'train_images/abc123.jpg'). No image preprocessing performed."
    }
    with open(paths["stats"], "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))
    print(f"Saved:\n- {paths['train_csv']}\n- {paths['valid_csv']}\n- {paths['test_csv']}\n- {paths['stats']}")

if __name__ == "__main__":
    main()