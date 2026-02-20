#!/usr/bin/env python3
"""
Script: make_patches.py
Description: 
    Generates the vascular topology inputs for the VascuMIL texture stream.
    
    Process:
    1. Loads the patient splits (Train/Val).
    2. Resizes images to high-resolution (768x768).
    3. Computes Vascular Topology Maps (VMAP) using Frangi Vesselness filters.
    4. Extracts patch pairs (RGB + VMAP) based on vessel density scoring.
    5. Saves patches to disk and generates a new metadata CSV.

Usage:
    Run from the project root directory:
    python data_preparation/make_patches.py \
        --splits_dir ./data/splits \
        --out_dir ./data/mil_dataset/metadata \
        --patch_dir ./data/mil_dataset/images \
        --fold 0 \
        --save_patches
"""

import os
import sys
import cv2
import argparse
import random
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path
import math
import hashlib

# --- Dynamic Import Setup ---
# Add project root to sys.path to allow importing from 'utils'
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

try:
    from utils import preprocessing as preproc
except ImportError as e:
    raise ImportError(f"Could not import 'utils.preprocessing'. Ensure you are running from the project root. Error: {e}")

# Suppress Frangi warnings
warnings.filterwarnings("ignore")

try:
    from skimage.filters import frangi
    _HAS_FRANGI = True
except ImportError:
    _HAS_FRANGI = False

# --- Defaults ---
DEFAULT_PATCH_SIZE = 224
DEFAULT_PREPROC_SIZE = 768
POS_PATCHES_PER_IMAGE = 24
NEG_PATCHES_PER_IMAGE = 8
MAX_POS_PER_IMAGE = 64
TOP_K_VESSEL_PIXELS = 200
LOW_VESSEL_THRESHOLD = 0.12
RANDOM_SEED = 42

def detect_fundus_circle_from_rgb(rgb):
    """Fallback fundus detection to generate mask if preprocessing mask is insufficient."""
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (7, 7), 0)

    try:
        circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, dp=1.1, minDist=min(h, w) // 2,
                                   param1=60, param2=30, minRadius=min(h, w)//6, maxRadius=min(h, w)//2)
    except Exception:
        circles = None

    mask = np.zeros((h, w), dtype=np.uint8)
    if circles is not None and len(circles) > 0:
        x, y, r = circles[0][0].astype(int)
        cv2.circle(mask, (x, y), max(4, int(r * 0.98)), 255, -1)
        return mask

    _, th = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    if num <= 1:
        cv2.circle(mask, (w//2, h//2), int(min(w, h)*0.48), 255, -1)
        return mask

    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == largest_idx).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    return mask

def compute_vesselness_map(green_uint8):
    """Return normalized vesselness float32 [0..1] same shape as input."""
    if green_uint8 is None: return np.zeros((0,), dtype=np.float32)
    grayf = green_uint8.astype(np.float32) / 255.0

    if _HAS_FRANGI:
        try:
            v = frangi(grayf, sigmas=range(1, 4))
            v = np.nan_to_num(v)
        except Exception:
            gx = cv2.Sobel(green_uint8, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(green_uint8, cv2.CV_32F, 0, 1)
            v = np.sqrt(gx*gx + gy*gy)
    else:
        gx = cv2.Sobel(green_uint8, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(green_uint8, cv2.CV_32F, 0, 1, ksize=3)
        v = np.sqrt(gx*gx + gy*gy)

    v_min, v_max = v.min(), v.max()
    if v_max > v_min:
        v = (v - v_min) / (v_max - v_min + 1e-12)
    return v.astype(np.float32)

def inpaint_vmap_over_background(vmap_float, mask_uint8):
    """Inpaint vmap over background to remove mask boundary artifacts."""
    v8 = (np.clip(vmap_float, 0.0, 1.0) * 255.0).astype(np.uint8)
    if mask_uint8 is None: return v8
    inp_mask = (mask_uint8 == 0).astype(np.uint8) * 255
    if inp_mask.max() == 0: return v8
    try:
        out = cv2.inpaint(v8, inp_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    except Exception:
        out = v8.copy()
    return cv2.GaussianBlur(out, (3, 3), 0)

def suppress_border_vmap(vmap_uint8, mask_uint8):
    """Attenuate vmap values near the mask boundary."""
    if vmap_uint8 is None or mask_uint8 is None: return vmap_uint8
    inside = (mask_uint8 > 0).astype(np.uint8)
    dist = cv2.distanceTransform(inside, cv2.DIST_L2, 5).astype(np.float32)
    att = np.clip((dist - 10.0) / 10.0, 0.0, 1.0)
    return (vmap_uint8.astype(np.float32) * att).astype(np.uint8)

def patch_id_from_row(row, patch_size):
    """Generates unique deterministic ID for a patch."""
    parent = Path(row["parent_image_path"])
    stem = parent.stem
    x, y = int(row["x"]), int(row["y"])
    key = f"{row['parent_image_path']}|{x}|{y}|{patch_size}|{patch_size}"
    hsh = hashlib.md5(key.encode("utf-8")).hexdigest()[:10]
    return f"{stem}__{x}_{y}_{patch_size}_{patch_size}__{hsh}"

def save_vmap_and_patch(crop_rgb, vmap_patch, out_dir, pid, save_rgb=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    v_path = out_dir / f"{pid}.png"
    r_path = None
    cv2.imwrite(str(v_path), vmap_patch)
    if save_rgb and crop_rgb is not None:
        r_path = out_dir / f"{pid}.jpg"
        cv2.imwrite(str(r_path), cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return str(v_path), str(r_path) if r_path else None

def sample_patches_for_image(row, args):
    """Worker function to process a single fundus image."""
    img_path = row.get("image_path")
    if args.data_dir:
        local_path = os.path.join(args.data_dir, os.path.basename(img_path))
        img_path = local_path if os.path.exists(local_path) else img_path
        
    if not os.path.exists(img_path): return []

    label = int(row.get("binary_plus_form", row.get("plus", 0)))
    pid_patient = row.get("patient_id", "")

    try:
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except: return []

    # Shared Preprocessing (768x768 for Texture stream)
    final_rgb, mask, green_masked = preproc.preprocess_full_image(
        rgb, target_square=args.preproc_target_square, apply_mask_to_img=True
    )
    
    H, W = final_rgb.shape[:2]
    if (mask > 0).sum() < 100: mask = detect_fundus_circle_from_rgb(final_rgb)

    vmap = compute_vesselness_map(green_masked)
    if vmap.size == 0: return []
    
    vmap_ip = inpaint_vmap_over_background(vmap, mask)
    vmap_supp = suppress_border_vmap(vmap_ip, mask)

    flat_idx = np.argsort(vmap_supp.ravel())[::-1]
    chosen_centers = []
    pad = args.patch_size // 2 + 1
    
    for idx in flat_idx:
        if len(chosen_centers) >= MAX_POS_PER_IMAGE: break
        y, x = int(idx // W), int(idx % W)
        if not (pad <= x < W - pad and pad <= y < H - pad): continue
        if mask[y, x] == 0: continue
        # Enforce spatial separation
        if any(math.hypot(y - cy, x - cx) < (args.patch_size * 0.3) for (cy, cx) in chosen_centers): continue
        chosen_centers.append((y, x))

    patches = []
    def add_patch(cy, cx, lab):
        tlx, tly = int(cx - args.patch_size // 2), int(cy - args.patch_size // 2)
        c_rgb, c_mask, c_green = preproc.crop_from_preprocessed(final_rgb, mask, tlx, tly, args.patch_size, args.patch_size)
        
        if c_rgb is None or np.std(c_green) < 3.0: return None
        if c_mask is not None and (np.sum(c_mask > 0) / c_mask.size) < 0.5: return None

        v_crop, _, _ = preproc.crop_from_preprocessed(np.dstack([vmap_supp]*3), mask, tlx, tly, args.patch_size, args.patch_size)
        vmap_patch = v_crop[:,:,0]

        # Calculate vessel density score
        f_vals = vmap_patch.ravel()
        n_top = min(len(f_vals), TOP_K_VESSEL_PIXELS)
        score = np.mean(np.partition(f_vals, -n_top)[-n_top:]) if n_top > 0 else 0.0

        pid = patch_id_from_row({"parent_image_path": img_path, "x": tlx, "y": tly}, args.patch_size)
        s_dir = Path(args.patch_dir) / f"fold_{row.get('fold', 0)}" / row.get("split", "train")
        vp, rp = save_vmap_and_patch(c_rgb, vmap_patch, s_dir, pid, save_rgb=args.save_patches)

        return {"parent_image_path": img_path, "patient_id": pid_patient, "x": tlx, "y": tly, "label": lab, 
                "vessel_score": score, "vmap_path": vp, "rgb_patch_path": rp, 
                "fold": row.get('fold', 0), "split": row.get('split', 'train')}

    for cy, cx in chosen_centers[:args.top_k]:
        p = add_patch(cy, cx, 1 if label==1 else 0)
        if p: patches.append(p)

    return patches

def main(args):
    print(f"[INFO] Initializing Patch Extraction for Fold {args.fold}")
    os.makedirs(args.out_dir, exist_ok=True)
    
    for split_name in ["train", "val"]:
        csv_path = Path(args.splits_dir) / f"fold_{args.fold}" / f"{split_name}.csv"
        if not csv_path.exists(): 
            print(f"[WARNING] {split_name} split not found for fold {args.fold}. Skipping.")
            continue
        
        print(f"[INFO] Processing {split_name} set...")
        df = pd.read_csv(csv_path)
        df['fold'] = args.fold
        df['split'] = split_name
        
        results = Parallel(n_jobs=args.num_workers)(
            delayed(sample_patches_for_image)(row.to_dict(), args) for _, row in tqdm(df.iterrows(), total=len(df))
        )
        
        flat_results = [p for sub in results for item in sub for p in (item if isinstance(item, list) else [item])]
        out_csv = Path(args.out_dir) / f"patches_{split_name}.csv"
        pd.DataFrame(flat_results).to_csv(out_csv, index=False)
        print(f"[SUCCESS] Wrote {len(flat_results)} patches to {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vascular Topology Patch Extractor")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--splits_dir", type=str, required=True, help="Path to create_splits.py output")
    parser.add_argument("--out_dir", type=str, required=True, help="Path to save metadata CSVs")
    parser.add_argument("--patch_dir", type=str, required=True, help="Path to save .jpg/.png assets")
    parser.add_argument("--data_dir", type=str, default=None, help="Optional image root override")
    parser.add_argument("--patch_size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--preproc_target_square", type=int, default=DEFAULT_PREPROC_SIZE)
    parser.add_argument("--top_k", type=int, default=POS_PATCHES_PER_IMAGE)
    parser.add_argument("--neg_k", type=int, default=NEG_PATCHES_PER_IMAGE)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_patches", action="store_true", help="Save RGB JPGs alongside VMAP PNGs")
    
    main(parser.parse_args())