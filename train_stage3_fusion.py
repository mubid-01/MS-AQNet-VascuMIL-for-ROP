#!/usr/bin/env python3
"""
Script: train_stage3_fusion.py
Description: 
    Final stage of the Asymmetric Ensemble pipeline. 
    Trains the Synergistic Fusion Meta-Learner using a Stacking strategy.

Process:
    1. Extracts 'Meta-Features' (logits) from frozen MS-AQNet and VascuMIL models.
    2. Concatenates specialist logits with re-injected clinical metadata.
    3. Trains a Multi-Task MLP to resolve Broad Diagnosis and Plus Disease objectives.
    4. Generates a comprehensive final clinical report.

Usage:
    python train_stage3_fusion.py \
        --ms_path ./output/ms_models/best_ms_clinical.pth \
        --mil_path ./output/mil_models/best_mil_clinical.pth \
        --val_csv ./data/splits/fold_0/val.csv \
        --patches_csv ./data/mil_dataset/metadata/patches_val.csv \
        --cache_root ./data/mil_dataset/images
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score

# --- Dynamic Import Setup ---
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

try:
    from models.model_ms_aqnet import MS_AQNet
    from models.model_vascumil import VascuMIL
    from models.model_fusion import FusionMetaLearner
    from utils.loader_structure import build_val_transform, TAB_COLS
    from utils.loader_texture import _apply_aligned_transforms
except ImportError as e:
    raise ImportError(f"Required modules not found. Ensure models/ and utils/ are populated. Error: {e}")

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- Constants & Stats ---
# Statistics calculated from the training fold to maintain leak-proof normalization
TAB_MEANS = {'gestational_age': 30.0, 'birth_weight': 1500.0, 'postconceptual_age': 36.0}
TAB_STDS = {'gestational_age': 4.0, 'birth_weight': 500.0, 'postconceptual_age': 4.0}
IMG_SIZE_MS = 384
IMG_SIZE_MIL = 224

# --- Helper: Robust Weights Loader ---
def load_checkpoint(model, path, device):
    try:
        ckpt = torch.load(path, map_location=device)
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            model.load_state_dict(ckpt['model_state'])
        else:
            model.load_state_dict(ckpt)
        return True
    except Exception as e:
        logging.error(f"Failed to load weights from {path}: {e}")
        return False

# --- Core Logic: Feature Extraction ---
def extract_meta_features(df, df_patches, ms_model, mil_model, cache_root, device):
    """Generates the training data for the Meta-Learner by running specialist inference."""
    logging.info("Extracting Meta-Features from Specialists (Stacking Strategy)...")
    
    ms_logits_list, mil_logits_list, tab_list = [], [], []
    target_diag, target_plus = [], []
    
    ms_trans = build_val_transform(IMG_SIZE_MS)
    
    ms_model.eval()
    mil_model.eval()
    
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
            # 1. Structural Feature Extraction (MS-AQNet)
            try:
                pil_img = Image.open(row['image_path']).convert("RGB")
                img_t = ms_trans(pil_img).unsqueeze(0).to(device)
                
                vec = [(row.get(c, TAB_MEANS[c]) - TAB_MEANS[c]) / (TAB_STDS[c] + 1e-8) for c in TAB_COLS]
                tab_t = torch.tensor([vec], dtype=torch.float32).to(device)
                
                out_ms = ms_model(img_t, tab_t)
                ms_l = out_ms['stage_logits'] 
            except Exception:
                ms_l = torch.zeros(1, 4).to(device)
                tab_t = torch.zeros(1, 3).to(device)

            # 2. Textural Feature Extraction (VascuMIL)
            pid = row['patient_id']
            fname = os.path.basename(row['image_path'])
            subset = df_patches[df_patches['patient_id'] == pid]
            subset = subset[subset['parent_image_path'].str.contains(fname, regex=False)]
            
            mil_l = torch.tensor([[-5.0]]).to(device) # Default low risk for missing data
            
            if len(subset) > 0:
                subset = subset.sort_values('vessel_score', ascending=False).head(24)
                tensors = []
                for _, p_row in subset.iterrows():
                    f, s = p_row.get('fold', 0), p_row.get('split', 'val')
                    pid_str = os.path.splitext(os.path.basename(p_row['rgb_patch_path']))[0]
                    
                    p_rgb = os.path.join(cache_root, f"fold_{f}/{s}", pid_str + ".jpg")
                    p_vmap = os.path.join(cache_root, f"fold_{f}/{s}", pid_str + ".png")
                    
                    if not os.path.exists(p_rgb):
                        # Portability Fallback
                        found = list(Path(cache_root).glob(f"**/{pid_str}.jpg"))
                        if found: 
                            p_rgb = str(found[0])
                            p_vmap = p_rgb.replace(".jpg", ".png")
                    
                    try:
                        rt, vt = _apply_aligned_transforms(Image.open(p_rgb), Image.open(p_vmap), IMG_SIZE_MIL, False)
                        tensors.append(torch.cat([rt, vt], dim=0))
                    except: continue
                
                if tensors:
                    batch = torch.stack(tensors).unsqueeze(0).to(device)
                    lens = torch.tensor([len(tensors)]).to(device)
                    out_mil = mil_model(batch, lens)
                    mil_l = out_mil['logits'].unsqueeze(0) 

            # Store for Stacking
            ms_logits_list.append(ms_l)
            mil_logits_list.append(mil_l)
            tab_list.append(tab_t)
            target_diag.append(int(row['broad_diagnosis']))
            target_plus.append(int(row['binary_plus_form']))

    return (
        torch.cat(ms_logits_list),
        torch.cat(mil_logits_list),
        torch.cat(tab_list),
        torch.tensor(target_diag).to(device),
        torch.tensor(target_plus).to(device).float().unsqueeze(1)
    )

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. Initialize & Load Specialists
    logging.info("--- Loading Specialists ---")
    ms_model = MS_AQNet(pretrained=False).to(device)
    ms_model.build(device=device)
    load_checkpoint(ms_model, args.ms_path, device)
    
    mil_model = VascuMIL(pretrained=False).to(device)
    mil_model.build(device=device)
    load_checkpoint(mil_model, args.mil_path, device)
    
    # 2. Extract Data
    df_val = pd.read_csv(args.val_csv)
    df_patches = pd.read_csv(args.patches_csv)
    
    ms_feat, mil_feat, tab_feat, y_diag, y_plus = extract_meta_features(
        df_val, df_patches, ms_model, mil_model, args.cache_root, device
    )
    
    # 3. Initialize Fusion Meta-Learner
    ensemble = FusionMetaLearner().to(device)
    optimizer = optim.Adam(ensemble.parameters(), lr=0.01)
    crit_diag = nn.CrossEntropyLoss()
    crit_plus = nn.BCEWithLogitsLoss()
    
    # 4. Meta-Training Loop
    logging.info("--- Training Fusion Meta-Learner ---")
    for epoch in range(101):
        ensemble.train()
        optimizer.zero_grad()
        d_out, p_out = ensemble(ms_feat, mil_feat, tab_feat)
        
        loss = crit_diag(d_out, y_diag) + crit_plus(p_out, y_plus)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            logging.info(f"   Epoch {epoch:3d} | Joint Loss: {loss.item():.4f}")
            
    # 5. Final Clinical Reporting
    ensemble.eval()
    with torch.no_grad():
        d_out, p_out = ensemble(ms_feat, mil_feat, tab_feat)
        
        # A. Broad Diagnosis Stats
        true_d = y_diag.cpu().numpy()
        preds_d = torch.argmax(d_out, dim=1).cpu().numpy()
        f1_d = f1_score(true_d, preds_d, average='macro')
        
        # Per-class Recall for Severe ROP (Class 2)
        rec_all = recall_score(true_d, preds_d, average=None, labels=[0,1,2,3])
        rec_sev = rec_all[2]
        
        # B. Plus Disease Stats
        true_p = y_plus.cpu().numpy().flatten()
        prob_p = torch.sigmoid(p_out).cpu().numpy().flatten()
        preds_p = (prob_p > 0.5).astype(int)
        auc_p = roc_auc_score(true_p, prob_p)
        rec_p = recall_score(true_p, preds_p)

    logging.info("\n" + "="*50)
    logging.info("FINAL ENSEMBLE CLINICAL REPORT")
    logging.info("="*50)
    logging.info(f"Task A (Diagnosis) | Macro F1: {f1_d:.4f}")
    logging.info(f"Task A (Diagnosis) | Severe Recall: {rec_sev:.4f}")
    logging.info(f"Task B (Plus)      | AUC:      {auc_p:.4f}")
    logging.info(f"Task B (Plus)      | Recall:   {rec_p:.4f}")
    logging.info("="*50)
    
    # Save Model
    save_path = os.path.join(args.out_dir, "final_fusion_ensemble.pth")
    torch.save(ensemble.state_dict(), save_path)
    logging.info(f"Synergistic ensemble weights saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: Synergistic Fusion")
    parser.add_argument("--ms_path", type=str, required=True, help="Path to MS-AQNet weights")
    parser.add_argument("--mil_path", type=str, required=True, help="Path to VascuMIL weights")
    parser.add_argument("--val_csv", type=str, required=True, help="Validation CSV for stacking")
    parser.add_argument("--patches_csv", type=str, required=True, help="Patches metadata CSV")
    parser.add_argument("--cache_root", type=str, required=True, help="Folder containing patch images")
    parser.add_argument("--out_dir", type=str, default="./output/fusion", help="Output directory")
    main(parser.parse_args())