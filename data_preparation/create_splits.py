#!/usr/bin/env python3
"""
Script: create_splits.py
Description: 
    Generates patient-level stratified folds for ROP classification.
    MATCHES RESEARCH PIPELINE LOGIC:
    1. "Test Set Hunter": Iterates seeds to guarantee Test Set contains Pathology (Severe/Plus).
    2. "Optimizer": Finds best 5-Fold split based on 'Badness Score'.
    3. "Safety Net": Explicitly moves patients to Validation if any class (0-3) is missing.

Usage:
    python data_preparation/create_splits.py --data_dir ./data/images --out_dir ./data/splits
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# --- Configuration ---
N_CV_SPLITS = 5
TEST_SIZE_RATIO = 0.10
MIN_FILENAME_PARTS = 6
# Seeds evaluated for optimal balance (Same as Notebook)
SEARCH_SEEDS = [42, 56, 2023, 99, 7, 123, 2024, 888]
IMAGE_EXTENSIONS = ("**/*.jpg", "**/*.jpeg", "**/*.png")

# Diagnostic Mapping (ICROP Guidelines)
DIAG_MAP = {0: 0, 1: 1, 2: 1, 9: 1, 3: 2, 4: 2, 8: 2, 10: 3, 11: 3, 12: 3, 13: 3}

def infer_device_from_path(path):
    p = path.lower()
    if "_d1_" in p or "retcam" in p: return "RetCam"
    if "_d2_" in p or "natus" in p: return "Natus"
    if "_d3_" in p or "icon" in p or "phoenix" in p: return "Phoenix"
    return "Unknown"

def parse_path(p):
    fname = os.path.basename(p)
    parts = fname.split('_')
    if len(parts) < MIN_FILENAME_PARTS: return None
    try:
        return {
            "patient_id": parts[0],
            "diagnosis_code": int(parts[5].replace("DG", "")),
            "plus_form": int(parts[6].replace("PF", "")),
            "gestational_age": float(parts[2].replace("GA", "")),
            "birth_weight": float(parts[3].replace("BW", "")),
            "postconceptual_age": float(parts[4].replace("PA", "")),
            "device": infer_device_from_path(p),
            "image_path": p
        }
    except Exception: return None

def calculate_imbalance_score(train_df, val_df):
    """Lower is better. Matches notebook logic exactly."""
    t_plus = train_df["binary_plus_form"].mean()
    v_plus = val_df["binary_plus_form"].mean()
    t_sev = (train_df["broad_diagnosis"]==2).mean()
    v_sev = (val_df["broad_diagnosis"]==2).mean()
    
    dev_diff = 0
    for d in train_df["device"].unique():
        t_d = (train_df["device"]==d).mean()
        v_d = (val_df["device"]==d).mean()
        dev_diff += abs(t_d - v_d)
        
    score = (abs(t_plus - v_plus) * 20.0) + (abs(t_sev - v_sev) * 5.0) + (dev_diff * 2.0)
    
    if train_df["binary_plus_form"].sum() == 0: score += 1000
    if val_df["binary_plus_form"].sum() == 0:   score += 1000
    return score

def ensure_class_presence(train_df, val_df, target_col, class_val):
    """
    If a specific class is missing in Validation, move the smallest patient from Train.
    This fixes the 'Zero Mild' or 'Zero Plus' issues.
    """
    if (val_df[target_col] == class_val).sum() > 0:
        return train_df, val_df

    # Find candidates in Train
    candidates = train_df[train_df[target_col] == class_val]
    if len(candidates) == 0: return train_df, val_df

    # Find smallest patient (fewest images)
    patient_counts = candidates.groupby("patient_id").size().sort_values()
    pid_to_move = patient_counts.index[0]
    
    # Move
    rows = train_df[train_df["patient_id"] == pid_to_move]
    train_df = train_df[train_df["patient_id"] != pid_to_move]
    val_df = pd.concat([val_df, rows])
    
    print(f"[INFO] Fixing missing class '{class_val}': Moved Patient {pid_to_move} to Validation.")
    return train_df, val_df

def main(args):
    print(f"[INFO] Parsing dataset from: {args.data_dir}")
    all_paths = []
    for ext in IMAGE_EXTENSIONS:
        all_paths.extend(glob.glob(os.path.join(args.data_dir, ext), recursive=True))
    
    if not all_paths:
        raise FileNotFoundError(f"No images found in {args.data_dir}")

    rows = [parse_path(p) for p in all_paths]
    df = pd.DataFrame([r for r in rows if r is not None])
    
    # Target Engineering
    df["broad_diagnosis"] = df["diagnosis_code"].map(DIAG_MAP).fillna(3).astype(int)
    df["binary_plus_form"] = df["plus_form"].apply(lambda x: 1 if int(x) == 2 else 0)
    
    dev_map = {d: i for i, d in enumerate(df["device"].unique())}
    df["device_code"] = df["device"].map(dev_map)
    df["strat_label"] = (df["binary_plus_form"] * 100) + (df["device_code"] * 10) + df["broad_diagnosis"]
    
    print(f"[INFO] Total Images: {len(df)} | Patients: {df['patient_id'].nunique()}")
    
    # --- Step 1: Extract 10% Test Set (The Hunter Logic) ---
    print("\n[INFO] Searching for balanced Test Set (10%)...")
    
    test_seed = 0
    found_valid_test = False
    
    # Try up to 100 seeds to guarantee Test Set has pathology
    for test_seed in range(100):
        splitter = StratifiedGroupKFold(n_splits=int(1/TEST_SIZE_RATIO), shuffle=True, random_state=test_seed)
        dev_idx, test_idx = next(splitter.split(df, df["strat_label"], df["patient_id"]))
        
        df_test_cand = df.iloc[test_idx]
        
        # Criteria: Must have Severe, Plus, and Mild cases
        has_severe = (df_test_cand["broad_diagnosis"] == 2).sum() > 0
        has_plus = (df_test_cand["binary_plus_form"] == 1).sum() > 0
        has_mild = (df_test_cand["broad_diagnosis"] == 1).sum() > 0
        
        if has_severe and has_plus and has_mild:
            print(f"[INFO] Found Valid Test Set with Seed {test_seed}")
            df_test = df_test_cand.reset_index(drop=True)
            df_dev = df.iloc[dev_idx].reset_index(drop=True)
            found_valid_test = True
            break
            
    if not found_valid_test:
        print("[WARNING] Could not find perfect test split. Using last attempt.")
    
    os.makedirs(args.out_dir, exist_ok=True)
    df_test.to_csv(os.path.join(args.out_dir, "test.csv"), index=False)
    print(f"[INFO] Test Set saved: {len(df_test)} images.")
    
    # --- Step 2: Optimize 5-Fold CV ---
    print(f"\n[INFO] Optimizing 5-Fold CV on Dev Set...")
    
    best_seed = None
    best_score = float('inf')
    best_splits = None
    
    for seed in SEARCH_SEEDS:
        sgkf = StratifiedGroupKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=seed)
        splits = list(sgkf.split(df_dev, df_dev["strat_label"], df_dev["patient_id"]))
        
        fold_scores = []
        for t_idx, v_idx in splits:
            t = df_dev.iloc[t_idx]
            v = df_dev.iloc[v_idx]
            fold_scores.append(calculate_imbalance_score(t, v))
        
        final_score = np.mean(fold_scores)
        if final_score < best_score:
            best_score = final_score
            best_seed = seed
            best_splits = splits
            
    print(f"[INFO] Optimal Seed: {best_seed} (Score: {best_score:.2f})")
    
    # --- Step 3: Save Folds & Fix Missing Classes ---
    meta = {'test_seed': test_seed, 'cv_seed': best_seed}
    
    for k, (t_idx, v_idx) in enumerate(best_splits):
        train_df = df_dev.iloc[t_idx].reset_index(drop=True)
        val_df = df_dev.iloc[v_idx].reset_index(drop=True)
        
        # 1. Ensure Plus Disease
        train_df, val_df = ensure_class_presence(train_df, val_df, "binary_plus_form", 1)
        
        # 2. Ensure All Diagnoses (0, 1, 2, 3)
        for cls in [0, 1, 2, 3]:
            train_df, val_df = ensure_class_presence(train_df, val_df, "broad_diagnosis", cls)
            
        fold_dir = os.path.join(args.out_dir, f"fold_{k}")
        os.makedirs(fold_dir, exist_ok=True)
        train_df.to_csv(os.path.join(fold_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(fold_dir, "val.csv"), index=False)
        
        print(f"[INFO] Fold {k} Saved. Val N={len(val_df)}")
        
    with open(os.path.join(args.out_dir, "split_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n[SUCCESS] All splits generated in {args.out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to images")
    parser.add_argument("--out_dir", type=str, default="./splits_balanced")
    args = parser.parse_args()
    main(args)