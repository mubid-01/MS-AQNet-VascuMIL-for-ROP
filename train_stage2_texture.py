#!/usr/bin/env python3
"""
Script: train_stage2_texture.py
Description: 
    Production training engine for VascuMIL (Texture Specialist).
    Focuses on micro-vascular Plus Disease detection via Gated Attention MIL.
    
Features:
    - 4-Channel (RGB + VMAP) instance processing.
    - Gated Attention pooling with variable bag lengths.
    - Bag-level Weighted Random Sampling for class balancing.
    - Balanced BCE loss with static/dynamic positive weighting.
    - Dual-save strategy: Best F1 and Best Clinical (Sensitivity-weighted).
"""

import os
import sys
import argparse
import logging
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

# --- Dynamic Import Setup ---
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

try:
    from utils.loader_texture import PatchMILDataset, mil_collate
    from models.model_vascumil import VascuMIL, count_parameters
except ImportError as e:
    raise ImportError(
        f"Required modules not found. Ensure models/model_vascumil.py "
        f"and utils/loader_texture.py exist. Error: {e}"
    )

# Setup Logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(y_true, y_pred, y_prob):
    """Calculates a comprehensive suite of clinical binary metrics."""
    res = {}
    res["acc"] = float(accuracy_score(y_true, y_pred))
    res["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    res["prec"] = float(precision_score(y_true, y_pred, zero_division=0))
    res["rec"] = float(recall_score(y_true, y_pred, zero_division=0))
    
    if len(set(y_true)) > 1:
        try:
            res["auc"] = float(roc_auc_score(y_true, y_prob))
        except:
            res["auc"] = 0.5
    else:
        res["auc"] = 0.5
    return res

def train_one_epoch(model, loader, optimizer, scaler, criterion, device):
    model.train()
    total_loss = 0.0
    preds, trues, probs = [], [], []
    use_amp = (device == "cuda")
    
    for batch in tqdm(loader, desc="[TRAIN]", leave=False):
        images = batch["images"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device).float()

        optimizer.zero_grad()
        
        with autocast(enabled=use_amp):
            outputs = model(images, lengths)
            logits = outputs["logits"]
            loss = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        
        with torch.no_grad():
            p = torch.sigmoid(logits).cpu().numpy()
            probs.extend(p.flatten().tolist())
            preds.extend((p > 0.5).astype(int).flatten().tolist())
            trues.extend(labels.cpu().numpy().flatten().astype(int).tolist())

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(trues, preds, probs)
    return avg_loss, metrics

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, trues, probs = [], [], []
    use_amp = (device == "cuda")

    with torch.no_grad():
        for batch in tqdm(loader, desc="[VAL  ]", leave=False):
            images = batch["images"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device).float()

            with autocast(enabled=use_amp):
                outputs = model(images, lengths)
                logits = outputs["logits"]
                loss = criterion(logits, labels)

            total_loss += loss.item()
            
            p = torch.sigmoid(logits).cpu().numpy()
            probs.extend(p.flatten().tolist())
            preds.extend((p > 0.5).astype(int).flatten().tolist())
            trues.extend(labels.cpu().numpy().flatten().astype(int).tolist())

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(trues, preds, probs)
    return avg_loss, metrics

def run_training(args):
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    
    logging.info(f" Starting VascuMIL Stage 2 Training (Texture Specialist)")
    logging.info(f"   Batch Size: {args.batch_size} | Bag Size: {args.bag_size}")

    # 1. Dataset & Balanced Sampler
    train_ds = PatchMILDataset(args.train_csv, args.cache_root, split="train", 
                               instances_per_bag=args.bag_size, is_train=True, sample_topk=True)
    val_ds = PatchMILDataset(args.val_csv, args.cache_root, split="val", 
                             instances_per_bag=args.bag_size, is_train=False, sample_topk=True)
    
    logging.info(f"   Dataset Loaded: {len(train_ds)} Train Bags | {len(val_ds)} Val Bags")

    # Bag-Level Label Extraction for Sampler
    train_labels = []
    for p in train_ds.parents:
        rows = [r for _, r in train_ds.grouped[p]]
        r = rows[0]
        lbl = float(r.get("binary_plus_form", r.get("plus", 0)))
        train_labels.append(lbl)
    
    counts = Counter(train_labels)
    n_neg, n_pos = counts[0.0], counts[1.0]
    
    # Weighted Sampler (Inverse Frequency)
    w_pos = 1.0 / n_pos if n_pos > 0 else 1.0
    w_neg = 1.0 / n_neg if n_neg > 0 else 1.0
    sample_weights = [w_pos if l == 1.0 else w_neg for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, collate_fn=mil_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=mil_collate)

    # 2. Model Initialization
    model = VascuMIL(pretrained=True, proj_dim=256, freeze_backbone=True).to(device)
    model.build(sample_input=(1, 4, 224, 224), device=device)
    logging.info(f" Trainable parameters: {count_parameters(model):,}")

    # 3. Loss Configuration (Fixes Calibration Shock)
    if args.static_pos_weight is not None:
        final_weight = float(args.static_pos_weight)
    else:
        # Dynamic calculation based on dataset imbalance
        final_weight = (n_neg / max(1, n_pos)) * args.pos_weight_mul
        final_weight = min(final_weight, 10.0) 
    
    logging.info(f" BCE pos_weight: {final_weight:.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([final_weight]).to(device))

    # 4. Optimizer & Schedulers
    # Start with Head-only training
    head_params = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    optimizer = optim.AdamW(head_params, lr=args.lr_head, weight_decay=1e-3)
    scaler = GradScaler() if device == "cuda" else None
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_f1 = 0.0
    best_clinical = 0.0

    # 5. Training Loop
    logging.info("-" * 80)
    logging.info(f"{'Ep':<3} | {'TLoss':<7} | {'VLoss':<7} | {'AUC':<6} | {'F1':<6} | {'Rec':<6} | {'Prec':<6} | {'ClinSc':<6}")
    logging.info("-" * 80)
    
    for ep in range(1, args.epochs + 1):
        # Differential Unfreeze Logic
        if ep == args.freeze_epochs + 1:
            logging.info(f">>> Unfreezing backbone at epoch {ep}")
            for p in model.encoder.backbone.parameters():
                p.requires_grad = True
            # Rebuild optimizer for all params
            optimizer = optim.AdamW([
                {'params': [p for n, p in model.named_parameters() if "backbone" not in n], 'lr': args.lr_head},
                {'params': [p for n, p in model.named_parameters() if "backbone" in n], 'lr': args.lr_backbone}
            ], weight_decay=1e-3)
            
        t_loss, _ = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device)
        v_loss, m = validate(model, val_loader, criterion, device)
        
        # Clinical Score = Average of F1 and Recall (Sensitivity)
        clinical = (m['f1'] + m['rec']) / 2.0
        scheduler.step(clinical)
        
        logging.info(f"{ep:02d} | {t_loss:.4f} | {v_loss:.4f} | {m['auc']:.4f} | {m['f1']:.4f} | {m['rec']:.4f} | {m['prec']:.4f} | {clinical:.4f}")
        
        # Checkpointing
        save_obj = {"epoch": ep, "model_state": model.state_dict(), "optim": optimizer.state_dict(), "metrics": m}
        
        if m['f1'] > best_f1:
            best_f1 = m['f1']
            torch.save(save_obj, os.path.join(args.out_dir, "best_mil_f1.pth"))
            logging.info(" Saved Best F1 Model")
            
        if clinical > best_clinical:
            best_clinical = clinical
            torch.save(save_obj, os.path.join(args.out_dir, "best_mil_clinical.pth"))
            logging.info(" Saved Best Clinical Model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 Training: Texture Stream")
    # Path Arguments
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--cache_root", type=str, required=True, help="Folder containing patch images")
    parser.add_argument("--out_dir", type=str, default="./output/mil_models")
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--bag_size", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--freeze_epochs", type=int, default=5)
    # Imbalance Control
    parser.add_argument("--pos_weight_mul", type=float, default=3.0)
    parser.add_argument("--static_pos_weight", type=float, default=None, help="Set to 1.0 if using sampler")
    # Environment
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    
    run_training(parser.parse_args())