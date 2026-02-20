#!/usr/bin/env python3
"""
Script: viz_combined_explainability.py
Description: 
    Generates multi-modal "Glass Box" explainability panels for the ROP ensemble.
    For each selected patient, it produces a 3-panel figure:
    1. Original Fundus Image.
    2. MS-AQNet Global Structural Attention (Grad-CAM++).
    3. VascuMIL Local Vascular Threat Map (Reprojected Attention).

Usage:
    python visualization/viz_combined_explainability.py \
        --ms_path ./output/ms_models/best_ms_clinical.pth \
        --mil_path ./output/mil_models/best_mil_clinical.pth \
        --data_dir ./data/images \
        --val_csv ./data/splits/fold_0/val.csv \
        --patches_csv ./data/mil_dataset/metadata/patches_val.csv \
        --out_dir ./output/visualizations
"""

import os
import sys
import cv2
import torch
import argparse
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import torch.nn.functional as F

# --- Dynamic Import Setup ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

try:
    from models.model_ms_aqnet import MS_AQNet
    from models.model_vascumil import VascuMIL
    from utils.loader_structure import build_val_transform, TAB_COLS
    from utils.loader_texture import _apply_aligned_transforms
except ImportError as e:
    raise ImportError(f"Required modules not found. Ensure models/ and utils/ are populated. Error: {e}")

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Global Constants matching trained pipeline
MS_SIZE = 384
MIL_PATCH_SIZE = 224
MIL_PREPROC_SIZE = 768
MIL_CHUNK_SIZE = 32
TAB_MEANS = {'gestational_age': 30.0, 'birth_weight': 1500.0, 'postconceptual_age': 36.0}
TAB_STDS = {'gestational_age': 4.0, 'birth_weight': 500.0, 'postconceptual_age': 4.0}

# --- 1. Native Grad-CAM Engine (Structure) ---
class NativeGradCAM:
    """Lightweight Grad-CAM implementation for custom multi-modal architectures."""
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output): self.activations = output
    def save_gradient(self, module, grad_input, grad_output): self.gradients = grad_output[0]

    def __call__(self, img, meta):
        self.model.zero_grad()
        out = self.model(img, meta)
        logits = out['stage_logits']
        target = torch.argmax(logits, dim=1).item()
        logits[0, target].backward()
        
        grads = self.gradients.cpu().data.numpy()[0]
        acts = self.activations.cpu().data.numpy()[0]
        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights): 
            cam += w * acts[i]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (MS_SIZE, MS_SIZE))
        if cam.max() > 0: cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam, target

# --- 2. Visualization Generators ---

def get_ms_visualization(row, model, cam_engine, trans, device):
    """Generates structural attention heatmap."""
    try:
        img_path = row['image_path']
        pil_img = Image.open(img_path).convert("RGB")
        img_t = trans(pil_img).unsqueeze(0).to(device)
        img_t.requires_grad = True
        
        vec = [(row.get(c, TAB_MEANS[c]) - TAB_MEANS[c]) / (TAB_STDS[c]+1e-8) for c in TAB_COLS]
        tab_t = torch.tensor([vec], dtype=torch.float32).to(device)
        
        cam_map, pred_cls = cam_engine(img_t, tab_t)
        
        orig_np = np.array(pil_img.resize((MS_SIZE, MS_SIZE)))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        overlay = cv2.addWeighted(orig_np, 0.6, heatmap, 0.4, 0)
        return orig_np, overlay, pred_cls
    except Exception as e:
        logging.warning(f"MS-AQNet viz failed for {row['patient_id']}: {e}")
        return None, None, None

def get_mil_visualization(row, df_patches, model, cache_root, device):
    """Generates local vessel threat map."""
    pid = row['patient_id']
    fname = os.path.basename(row['image_path'])
    subset = df_patches[df_patches['patient_id'] == pid]
    # Match specific image in multi-image series
    subset = subset[subset['parent_image_path'].str.contains(Path(fname).stem, regex=False)]
    
    if len(subset) == 0: return None
    
    tensors, vmaps, coords = [], [], []
    for _, p_row in subset.iterrows():
        p_name = Path(p_row['rgb_patch_path']).name
        # Resolve paths via cache root
        p_rgb = Path(cache_root) / f"fold_{p_row.get('fold', 0)}" / p_row.get('split', 'val') / p_name
        p_vmap = p_rgb.with_suffix('.png')
        
        if not p_rgb.exists():
            # Fallback search
            found = list(Path(cache_root).glob(f"**/{p_name}"))
            if found: 
                p_rgb = found[0]
                p_vmap = p_rgb.with_suffix('.png')
        
        try:
            rt, vt = _apply_aligned_transforms(Image.open(p_rgb), Image.open(p_vmap), MIL_PATCH_SIZE, False)
            tensors.append(torch.cat([rt, vt], dim=0))
            vmaps.append(np.array(Image.open(p_vmap).convert("L")))
            coords.append((p_row['x'], p_row['y']))
        except: continue
        
    if not tensors: return None

    # Inference
    all_emb = []
    with torch.no_grad():
        for i in range(0, len(tensors), MIL_CHUNK_SIZE):
            chunk = torch.stack(tensors[i:i+MIL_CHUNK_SIZE]).to(device)
            emb = model.encoder(chunk)
            all_emb.append(emb)
            
        H_inst = torch.cat(all_emb, dim=0).unsqueeze(0)
        _, attn = model.att(H_inst)
        attn = attn.squeeze().cpu().numpy()

    # High-Contrast Stitching
    heatmap = np.zeros((MIL_PREPROC_SIZE, MIL_PREPROC_SIZE), dtype=np.float32)
    # 1. Square for contrast & Normalize Locally
    attn = attn ** 2
    if attn.max() > attn.min(): attn = (attn - attn.min()) / (attn.max() - attn.min())
        
    for i, (x, y) in enumerate(coords):
        score = attn[i]
        weighted_patch = (vmaps[i].astype(np.float32) / 255.0) * score
        h_end = min(y + MIL_PATCH_SIZE, MIL_PREPROC_SIZE)
        w_end = min(x + MIL_PATCH_SIZE, MIL_PREPROC_SIZE)
        ph, pw = h_end - y, w_end - x
        heatmap[y:h_end, x:w_end] = np.maximum(heatmap[y:h_end, x:w_end], weighted_patch[:ph, :pw])
        
    # 2. Rendering
    try:
        orig = cv2.cvtColor(cv2.imread(row['image_path']), cv2.COLOR_BGR2RGB)
        orig = cv2.resize(orig, (MIL_PREPROC_SIZE, MIL_PREPROC_SIZE))
    except: return None
    
    # Apply Gamma (0.6) for Neon visibility
    if heatmap.max() > 0:
        heatmap_norm = heatmap / heatmap.max()
        heatmap_norm = np.power(heatmap_norm, 0.6)
    else: heatmap_norm = heatmap
        
    heatmap_color = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * heatmap_norm), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    
    # Masking vessels (threshold 0.1)
    mask = np.dstack([(heatmap_norm > 0.1).astype(np.float32)]*3) 
    # Dim background to 40% intensity
    dimmed_bg = (orig.astype(np.float32) * 0.4).astype(np.uint8)
    final_viz = dimmed_bg * (1 - mask) + heatmap_color * mask
    
    return cv2.resize(final_viz.astype(np.uint8), (MS_SIZE, MS_SIZE))

# --- 3. Main Logic ---

def run_visualization(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load Models
    ms_model = MS_AQNet(pretrained=False).to(device)
    ms_model.build(device=device)
    ms_ckpt = torch.load(args.ms_path, map_location=device)
    ms_model.load_state_dict(ms_ckpt['model_state'] if 'model_state' in ms_ckpt else ms_ckpt)
    ms_model.eval()
    
    mil_model = VascuMIL(pretrained=False).to(device)
    mil_model.build(device=device)
    mil_model.load_state_dict(torch.load(args.mil_path, map_location=device))
    mil_model.eval()
    
    cam_engine = NativeGradCAM(ms_model, ms_model.backbone.blocks[-1])
    ms_trans = build_val_transform(MS_SIZE)
    
    # Load Data
    df_val = pd.read_csv(args.val_csv)
    df_patches = pd.read_csv(args.patches_csv)
    labels = {0: "Normal", 1: "Mild", 2: "Severe", 3: "Other"}

    for cls_idx in [2, 1, 0]:
        print(f"\n[INFO] Generating explainability for Class: {labels[cls_idx]}")
        candidates = df_val[df_val['broad_diagnosis'] == cls_idx]
        if cls_idx == 2: # Prefer Plus Disease for Severe
            p_candidates = candidates[candidates['binary_plus_form'] == 1]
            if not p_candidates.empty: candidates = p_candidates
            
        # RANDOMIZATION: Pull fresh samples per run
        candidates = candidates.sample(frac=1).reset_index(drop=True)
        
        count = 0
        for _, row in candidates.iterrows():
            if count >= args.num_samples: break
            
            orig, ms_map, ms_pred = get_ms_visualization(row, ms_model, cam_engine, ms_trans, device)
            mil_map = get_mil_visualization(row, df_patches, mil_model, args.cache_root, device)
            
            if orig is None or mil_map is None: continue
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
            axes[0].imshow(orig)
            axes[0].set_title(f"Input: {labels[cls_idx]}", fontsize=14, weight='bold')
            axes[1].imshow(ms_map)
            axes[1].set_title(f"Structural (MS-AQNet)\nPred: {labels[ms_pred]}", fontsize=14)
            axes[2].imshow(mil_map)
            axes[2].set_title(f"Vascular (VascuMIL)\nThreat Map", fontsize=14)
            
            for ax in axes: ax.axis('off')
            plt.tight_layout()
            
            save_name = f"viz_{labels[cls_idx]}_{row['patient_id']}.png"
            plt.savefig(os.path.join(args.out_dir, save_name), bbox_inches='tight')
            plt.show()
            count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble Explainability Suite")
    parser.add_argument("--ms_path", type=str, required=True)
    parser.add_argument("--mil_path", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--patches_csv", type=str, required=True)
    parser.add_argument("--cache_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./output/viz")
    parser.add_argument("--num_samples", type=int, default=2, help="Random samples per class")
    run_visualization(parser.parse_args())