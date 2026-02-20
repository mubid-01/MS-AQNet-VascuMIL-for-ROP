#!/usr/bin/env python3
"""
Script: train_stage1_structure.py
Description: 
    Production training engine for MS-AQNet (Structure Specialist).
    Implements the 'Bulldog' strategy for high-sensitivity ROP staging.
    
Features:
    - Multi-scale Active Query and FiLM fusion.
    - Class-weighted Clinical Focal Loss (Bulldog Weights).
    - Frozen Batch Normalization for small-batch stability.
    - Automated backbone unfreezing schedule.
"""

import os
import sys
import argparse
import logging
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

# --- Dynamic Import Setup ---
# Ensures the script can find 'models' and 'utils' folders from the root
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

try:
    from utils.loader_structure import make_dataloaders
    from models.model_ms_aqnet import MS_AQNet, count_parameters
except ImportError as e:
    raise ImportError(
        f"Required modules not found. Ensure models/model_ms_aqnet.py "
        f"and utils/loader_structure.py exist. Error: {e}"
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

class ClinicalFocalLoss(nn.Module):
    """Multiclass Focal Loss with Clinical Prioritization Weights."""
    def __init__(self, class_weights: torch.Tensor = None, gamma: float = 2.0, device="cpu"):
        super().__init__()
        self.gamma = float(gamma)
        self.device = device
        # Default: Bulldog Weights [Normal, Mild, Severe, Other]
        if class_weights is None:
            self.class_weights = torch.tensor([0.5, 1.0, 5.0, 1.0]).to(device)
        else:
            self.class_weights = class_weights.to(device)
            
        self.ce = nn.CrossEntropyLoss(weight=self.class_weights, reduction="none")

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        return loss.mean()

def train_one_epoch(model, loader, optimizer, scaler, criterion, device, clip_norm=1.0, aux_weight=0.2):
    model.train()
    total_loss = 0.0
    n = 0
    use_amp = (device == "cuda")

    for batch in tqdm(loader, desc="Train", leave=False):
        imgs = batch["image"].to(device)
        tabs = batch["tabular"].to(device)
        targets = batch["diagnosis"].to(device).long()

        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            out = model(imgs, tabs)
            logits = out["stage_logits"]
            loss = criterion(logits, targets)
            
            # Deep Supervision Loss
            if out.get("aux1") is not None:
                loss += aux_weight * criterion(out["aux1"], targets)
            if out.get("aux2") is not None:
                loss += aux_weight * criterion(out["aux2"], targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

        total_loss += float(loss.item())
        n += 1

    return total_loss / max(1, n)

def validate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    preds, trues, probs = [], [], []
    n = 0
    use_amp = (device == "cuda")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            imgs = batch["image"].to(device)
            tabs = batch["tabular"].to(device)
            targets = batch["diagnosis"].to(device).long()

            with autocast(enabled=use_amp):
                out = model(imgs, tabs)
                logits = out["stage_logits"]
                loss = criterion(logits, targets)
            
            total_loss += float(loss.item())
            n += 1

            soft = torch.softmax(logits, dim=1).cpu().numpy()
            pred = np.argmax(soft, axis=1)
            preds.extend(pred.tolist())
            trues.extend(targets.cpu().numpy().tolist())
            probs.extend(soft.tolist())

    # Metric Calculation
    f1 = f1_score(trues, preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(np.array(trues), np.array(probs), multi_class='ovr', average='macro')
    except:
        auc = 0.5
    
    rec = recall_score(trues, preds, average=None, labels=list(range(num_classes)), zero_division=0)
    prec = precision_score(trues, preds, average=None, labels=list(range(num_classes)), zero_division=0)

    return {"loss": total_loss / max(1, n), "f1": float(f1), "auc": float(auc), "rec": rec, "prec": prec}

def run_training(args):
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    
    logging.info(f" Starting MS-AQNet Stage 1 Training | Fold: {args.fold}")

    # 1. Dataloaders
    train_loader, val_loader, _, _ = make_dataloaders(
        splits_dir=args.folds_dir, 
        fold=args.fold, 
        batch_size=args.batch_size,
        sampler=(not args.no_sampler), 
        image_size=384, 
        num_workers=args.num_workers
    )

    # 2. Weights Configuration
    if args.hard_weights:
        cw = torch.tensor([float(x) for x in args.hard_weights.split(",")]).float()
    else:
        cw = torch.tensor([0.5, 1.0, 5.0, 1.0]).float() # Bulldog Defaults
    logging.info(f"⚖️ Applied Loss Weights: {cw.tolist()}")

    # 3. Model Initialization
    model = MS_AQNet(pretrained=True, freeze_backbone=True, device=device)
    model.build(sample_input=(1, 3, 384, 384), device=device, num_classes=args.num_classes)
    logging.info(f"📊 Trainable parameters: {count_parameters(model):,}")
    
    criterion = ClinicalFocalLoss(class_weights=cw, gamma=2.0, device=device)

    # 4. Optimizer & Scheduler
    # Stage 1: Heads only
    head_params = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    optimizer = optim.AdamW([{'params': head_params, 'lr': args.lr_head, 'weight_decay': args.wd_head}], betas=(0.9, 0.999))
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_f1 = 0.0
    best_clinical = 0.0

    # 5. Training Loop
    logging.info("-" * 85)
    logging.info(f"{'Ep':<3} | {'TLoss':<7} | {'VLoss':<7} | {'F1':<6} | {'AUC':<6} | {'Rec[2]':<6} | {'ClinSc':<6}")
    logging.info("-" * 85)

    for ep in range(1, args.epochs + 1):
        # Backbone Unfreezing
        if ep == (args.freeze_backbone_epochs + 1):
            logging.info(f">>> Unfreezing backbone at epoch {ep}")
            for p in model.backbone.parameters():
                p.requires_grad = True
            
            head_params = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
            back_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
            
            optimizer = optim.AdamW([
                {"params": head_params, "lr": args.lr_head, "weight_decay": args.wd_head},
                {"params": back_params, "lr": args.lr_backbone, "weight_decay": args.wd_backbone}
            ], betas=(0.9, 0.999))
            scaler = GradScaler()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device, args.clamp_grad_norm, args.aux_weight)
        m = validate(model, val_loader, criterion, device, args.num_classes)
        
        # Clinical Score calculation
        rec_sev = float(m["rec"][2]) if len(m["rec"]) > 2 else 0.0
        clinical = (m["f1"] + rec_sev) / 2.0

        logging.info(f"{ep:02d} | {train_loss:.4f} | {m['loss']:.4f} | {m['f1']:.4f} | {m['auc']:.4f} | {rec_sev:.4f} | {clinical:.4f}")

        scheduler.step(m["f1"])

        # Checkpoints
        save_obj = {
            "epoch": ep,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "f1": m["f1"],
            "clinical": clinical
        }

        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            torch.save(save_obj, os.path.join(args.out_dir, "best_ms_f1.pth"))
            logging.info(" Saved Best F1")

        if clinical > best_clinical:
            best_clinical = clinical
            torch.save(save_obj, os.path.join(args.out_dir, "best_ms_clinical.pth"))
            logging.info(" Saved Best Clinical")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 1 Training: Structure Stream")
    # Path Arguments
    p.add_argument("--folds_dir", type=str, required=True, help="Path to directory with train/val CSVs")
    p.add_argument("--out_dir", type=str, default="./output/ms_models", help="Output directory")
    # Training Config
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--lr_backbone", type=float, default=1e-6)
    p.add_argument("--freeze_backbone_epochs", type=int, default=5)
    p.add_argument("--wd_head", type=float, default=1e-3)
    p.add_argument("--wd_backbone", type=float, default=1e-5)
    # Loss Config
    p.add_argument("--aux_weight", type=float, default=0.2)
    p.add_argument("--clamp_grad_norm", type=float, default=1.0)
    p.add_argument("--no_sampler", action="store_true", help="Disable weighted sampler")
    p.add_argument("--hard_weights", type=str, default=None, help="Comma-separated class weights")
    # Environment
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=56)
    p.add_argument("--num_workers", type=int, default=4)
    
    run_training(p.parse_args())