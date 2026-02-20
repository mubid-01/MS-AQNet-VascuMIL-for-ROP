#!/usr/bin/env python3
"""
Module: loader_texture.py
Description: 
    PyTorch Dataset and DataLoader implementations for VascuMIL (Texture Stream).
    
    Key Features:
    - 4-Channel Construction: Loads RGB patches and corresponding Vascular Topology Maps (VMAP),
      concatenating them depth-wise.
    - Synchronized Augmentation: Ensures geometric transforms (flips, rotations) are 
      applied identically to both RGB and VMAP to maintain spatial alignment.
    - Bag Formulation: Groups patches by parent image (patient) to create 
      variable-length 'bags' for Multiple Instance Learning.
"""

import os
import sys
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision.transforms.functional as TF
import random

# --- Dynamic Import Setup ---
# Add project root to sys.path so we can import from 'utils'
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

try:
    from utils import preprocessing as preproc
except ImportError:
    # Fallback for localized execution
    try:
        import preprocessing as preproc
    except ImportError:
        print("[WARNING] Could not import 'utils.preprocessing'. Ensure paths are correct.")

# --- Constants ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
TAB_COLS = ["gestational_age", "birth_weight", "postconceptual_age"]

def patch_id_from_row(row: Dict[str, Any]) -> str:
    """Generates unique deterministic ID for a patch based on location and parent."""
    parent = Path(row.get("parent_image_path", ""))
    stem = parent.stem
    x = int(row.get("x", 0))
    y = int(row.get("y", 0))
    w = int(row.get("w", 224))
    h = int(row.get("h", 224))
    
    key = f"{row.get('parent_image_path', '')}|{x}|{y}|{w}|{h}"
    hsh = hashlib.md5(key.encode("utf-8")).hexdigest()[:10]
    return f"{stem}__{x}_{y}_{w}_{h}__{hsh}.jpg"

def _apply_aligned_transforms(pil_rgb: Image.Image, pil_vmap: Optional[Image.Image],
                              patch_size: int, is_train: bool):
    """
    Applies identical spatial transformations to both RGB and VMAP.
    Returns: (rgb_tensor, vmap_tensor)
    """
    if pil_rgb.mode != "RGB":
        pil_rgb = pil_rgb.convert("RGB")
    if pil_vmap is not None and pil_vmap.mode != "L":
        pil_vmap = pil_vmap.convert("L")

    # Base Resize
    pil_rgb = pil_rgb.resize((patch_size, patch_size), resample=Image.BILINEAR)
    if pil_vmap is not None:
        pil_vmap = pil_vmap.resize((patch_size, patch_size), resample=Image.BILINEAR)

    if is_train:
        # Spatial: Flips
        if random.random() < 0.5:
            pil_rgb = ImageOps.mirror(pil_rgb)
            if pil_vmap is not None:
                pil_vmap = ImageOps.mirror(pil_vmap)
        if random.random() < 0.5:
            pil_rgb = ImageOps.flip(pil_rgb)
            if pil_vmap is not None:
                pil_vmap = ImageOps.flip(pil_vmap)
        
        # Spatial: Rotation
        angle = random.uniform(-30.0, 30.0)
        pil_rgb = pil_rgb.rotate(angle, resample=Image.BILINEAR, fillcolor=(0,0,0))
        if pil_vmap is not None: 
            pil_vmap = pil_vmap.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
        
        # Spatial: Affine Translation
        if random.random() < 0.25:
            max_t = int(patch_size * 0.06)
            tx = random.randint(-max_t, max_t)
            ty = random.randint(-max_t, max_t)
            
            pil_rgb = TF.affine(pil_rgb, angle=0, translate=(tx, ty), scale=1.0, shear=0, 
                                interpolation=T.InterpolationMode.BILINEAR, fill=(0,0,0))
            if pil_vmap is not None:
                pil_vmap = TF.affine(pil_vmap, angle=0, translate=(tx, ty), scale=1.0, shear=0, 
                                     interpolation=T.InterpolationMode.BILINEAR, fill=0)

        # Photometric: Color Jitter (Applied ONLY to RGB texture)
        cj = T.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05)
        pil_rgb = cj(pil_rgb)

    # Convert to Tensors
    rgb_t = TF.to_tensor(pil_rgb)
    rgb_t = TF.normalize(rgb_t, IMAGENET_MEAN, IMAGENET_STD)

    if pil_vmap is None:
        vmap_t = torch.zeros((1, patch_size, patch_size), dtype=torch.float32)
    else:
        vmap_np = np.array(pil_vmap).astype(np.float32) / 255.0
        vmap_t = torch.from_numpy(vmap_np).unsqueeze(0).float()

    return rgb_t, vmap_t

class PatchMILDataset(Dataset):
    def __init__(
        self,
        patches_csv: str,
        cache_root: Optional[str] = None, 
        fold: int = 0,
        split: str = "train",
        instances_per_bag: int = 24,
        is_train: bool = True,
        sample_topk: bool = True,
        patch_size: int = 224,
    ):
        super().__init__()
        self.patches_csv = Path(patches_csv)
        if not self.patches_csv.exists():
            raise FileNotFoundError(f"Patches CSV not found: {self.patches_csv}")
        
        self.df = pd.read_csv(self.patches_csv)
        self.fold = int(fold)
        self.split = split
        self.cache_root = Path(cache_root) if cache_root is not None else None
        self.instances_per_bag = int(instances_per_bag)
        self.is_train = bool(is_train)
        self.sample_topk = bool(sample_topk)
        self.patch_size = int(patch_size)

        self.grouped = {}
        for idx, row in self.df.iterrows():
            p = str(row["parent_image_path"])
            self.grouped.setdefault(p, []).append((idx, row.to_dict()))

        self.parents = sorted(list(self.grouped.keys()))

    def __len__(self): 
        return len(self.parents)

    def _resolve_path(self, row: Dict[str, Any], col_name: str, ext: str):
        """Dynamically locates the patch file using absolute path or cache root."""
        if col_name in row and isinstance(row[col_name], str):
            p = Path(row[col_name])
            if p.exists(): return p
            if self.cache_root:
                p_root = self.cache_root / p.name
                if p_root.exists(): return p_root
                p_fold = self.cache_root / f"fold_{self.fold}" / self.split / p.name
                if p_fold.exists(): return p_fold
        return None

    def _sample_indices_for_bag(self, rows: List[Dict[str, Any]]) -> List[int]:
        """Selects K patches per bag, prioritizing high vesselness scores."""
        n = len(rows)
        k = self.instances_per_bag
        if n == 0: return []
        
        if self.sample_topk and "vessel_score" in rows[0]:
            sorted_idx = sorted(range(n), key=lambda i: float(rows[i].get("vessel_score", 0.0)), reverse=True)
            if k <= n: 
                return sorted_idx[:k]
            else:
                chosen = sorted_idx[:]
                extra = np.random.choice(sorted_idx, size=(k - n), replace=True).tolist()
                chosen.extend(extra)
                return chosen
        else:
            if n >= k: 
                return np.random.choice(range(n), size=k, replace=False).tolist()
            else: 
                return np.random.choice(range(n), size=k, replace=True).tolist()

    def __getitem__(self, idx: int):
        parent = self.parents[idx]
        rows = [r for _, r in self.grouped[parent]]
        
        row0 = rows[0]
        if "binary_plus_form" in row0: lab = float(row0["binary_plus_form"])
        elif "plus" in row0: lab = float(row0["plus"])
        else: lab = 0.0

        selected_indices = self._sample_indices_for_bag(rows)
        insts = []
        
        for i in selected_indices:
            row = rows[i]
            p_rgb = self._resolve_path(row, "rgb_patch_path", ".jpg")
            p_vmap = self._resolve_path(row, "vmap_path", ".png")
            
            try:
                pil_rgb = Image.open(p_rgb).convert("RGB") if p_rgb else None
            except: 
                pil_rgb = None
            
            try:
                pil_vmap = Image.open(p_vmap).convert("L") if p_vmap else None
            except: 
                pil_vmap = None
            
            if pil_rgb is None: 
                pil_rgb = Image.new("RGB", (self.patch_size, self.patch_size), (0,0,0))
                
            rgb_t, vmap_t = _apply_aligned_transforms(pil_rgb, pil_vmap, self.patch_size, self.is_train)
            inst = torch.cat([rgb_t, vmap_t], dim=0)
            insts.append(inst)

        return {
            "images": torch.stack(insts, dim=0),
            "label": torch.tensor(lab, dtype=torch.float32),
            "parent": parent
        }

def mil_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Handles variable length bags by returning padded tensors and lengths."""
    B = len(batch)
    lengths = [item["images"].shape[0] for item in batch]
    M_max = max(lengths)
    C, H, W = batch[0]["images"].shape[1:]
    
    images = torch.zeros((B, M_max, C, H, W), dtype=torch.float32)
    labels = torch.zeros((B,), dtype=torch.float32)
    parents = []
    
    for i, item in enumerate(batch):
        m = lengths[i]
        images[i, :m] = item["images"]
        labels[i] = item["label"]
        parents.append(item.get("parent", ""))
        
    return {
        "images": images,
        "labels": labels,
        "parents": parents,
        "lengths": torch.tensor(lengths, dtype=torch.int64)
    }

if __name__ == "__main__":
    print("[INFO] Testing loader_texture.py module...")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to patches CSV")
    parser.add_argument("--root", type=str, required=True, help="Path to patch images root")
    args = parser.parse_args()
    
    try:
        ds = PatchMILDataset(patches_csv=args.csv, cache_root=args.root, instances_per_bag=4)
        loader = DataLoader(ds, batch_size=2, collate_fn=mil_collate)
        batch = next(iter(loader))
        print(" Dataloader initialized successfully.")
        print(f"   Batch Shape: {batch['images'].shape}")
        print(f"   Labels:      {batch['labels']}")
    except Exception as e:
        print(f" Test failed: {e}")