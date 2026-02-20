#!/usr/bin/env python3
"""
Module: loader_structure.py
Description: 
    PyTorch Dataset and DataLoader implementations for the MS-AQNet (Structure Stream).
    
    Key Features:
    - Imports core preprocessing logic from `utils.preprocessing`.
    - Implements aggressive geometric augmentation (180-deg rotation, elastic transform).
    - Applies leak-free Z-score normalization for clinical metadata.
    - Includes optional 'Effective Number of Samples' Class-Balanced Weighted Sampler.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# --- Dynamic Import Setup ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

try:
    from utils import preprocessing as preproc
except ImportError:
    try:
        import preprocessing as preproc
    except ImportError:
        print("[WARNING] Could not import 'utils.preprocessing'. Ensure paths are correct.")

# --- Constants ---
TAB_COLS = ["gestational_age", "birth_weight", "postconceptual_age"]
DEFAULT_IMAGE_SIZE = 384
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class PreprocessThenPIL:
    """Wrapper to apply preproc_shared logic in a Torchvision pipeline."""
    def __init__(self, target_square=DEFAULT_IMAGE_SIZE):
        self.target_square = target_square

    def __call__(self, pil_img):
        img_np = np.array(pil_img.convert("RGB"))
        final_rgb, _, _ = preproc.preprocess_full_image(
            img_np, 
            target_square=self.target_square,
            apply_mask_to_img=True
        )
        return Image.fromarray(final_rgb.astype("uint8"))

def build_train_transform(image_size=DEFAULT_IMAGE_SIZE):
    """Aggressive augmentation for small medical datasets."""
    return T.Compose([
        PreprocessThenPIL(target_square=image_size),
        T.RandomAffine(degrees=180, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.RandomApply([T.ElasticTransform(alpha=50.0, sigma=5.0)], p=0.25),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def build_val_transform(image_size=DEFAULT_IMAGE_SIZE):
    """Deterministic pipeline for validation and inference."""
    return T.Compose([
        PreprocessThenPIL(target_square=image_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

class StructureDataset(Dataset):
    def __init__(self, csv_path, tab_means, tab_stds, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.tab_means = tab_means
        self.tab_stds = tab_stds
        
        for c in TAB_COLS:
            if c not in self.df.columns:
                self.df[c] = self.tab_means.get(c, 0.0)
            self.df[c] = pd.to_numeric(self.df[c], errors="coerce").fillna(self.tab_means.get(c, 0.0))

    def __len__(self): 
        return len(self.df)

    def _safe_load(self, path: str):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return Image.new("RGB", (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self._safe_load(str(row["image_path"]))
        
        if self.transform is not None:
            img_t = self.transform(img)
        else:
            img_t = TF.to_tensor(img)
            img_t = TF.normalize(img_t, IMAGENET_MEAN, IMAGENET_STD)

        # Corrected Tabular Normalization Loop
        tab_array = np.array([
            (row[c] - self.tab_means.get(c, 0.0)) / (self.tab_stds.get(c, 1.0) + 1e-8) 
            for c in TAB_COLS
        ], dtype=np.float32)
        
        tab_t = torch.tensor(tab_array, dtype=torch.float32)
        diagnosis = int(row.get("broad_diagnosis", 0))
        
        return {
            "image": img_t,
            "tabular": tab_t,
            "diagnosis": torch.tensor(diagnosis, dtype=torch.long),
            "patient_id": str(row.get("patient_id", ""))
        }

def make_dataloaders(splits_dir, fold=0, batch_size=16, sampler=False, image_size=DEFAULT_IMAGE_SIZE, num_workers=4):
    fold_dir = Path(splits_dir) / f"fold_{fold}"
    train_csv, val_csv = fold_dir / "train.csv", fold_dir / "val.csv"
    
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing split CSVs in {fold_dir}")

    train_df = pd.read_csv(train_csv)
    # Calculate stats only on numeric tabular columns
    tab_means = train_df[TAB_COLS].mean().to_dict()
    tab_stds = train_df[TAB_COLS].std().replace(0, 1.0).to_dict()

    train_ds = StructureDataset(str(train_csv), tab_means, tab_stds, transform=build_train_transform(image_size))
    val_ds   = StructureDataset(str(val_csv), tab_means, tab_stds, transform=build_val_transform(image_size))

    sampler_obj = None
    if sampler:
        counts_series = train_df["broad_diagnosis"].value_counts().sort_index()
        counts = counts_series.values
        beta = 0.999
        eff_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / (eff_num + 1e-8)
        weights = weights / np.sum(weights) * len(counts)
        
        mapping = {int(k): float(weights[i]) for i, k in enumerate(counts_series.index)}
        sample_weights = train_df["broad_diagnosis"].map(lambda x: mapping.get(int(x), 1.0)).values
        sampler_obj = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler_obj, 
                              shuffle=(sampler_obj is None), num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, sampler_obj, {"tab_means": tab_means, "tab_stds": tab_stds}

if __name__ == "__main__":
    print("[INFO] Testing Structure DataLoader...")
    # Update these paths to match your GitHub structure for a local test
    TEST_SPLITS = "./data/splits"
    if os.path.exists(TEST_SPLITS):
        try:
            t_loader, v_loader, _, stats = make_dataloaders(splits_dir=TEST_SPLITS, fold=0, batch_size=2)
            batch = next(iter(t_loader))
            print("✅ Dataloader initialized.")
            print(f"   Image shape:   {batch['image'].shape}")
            print(f"   Tabular shape: {batch['tabular'].shape}")
        except Exception as e:
            print(f"❌ Test failed: {e}")
    else:
        print(f"[SKIP] Test skipped. Directory {TEST_SPLITS} not found.")