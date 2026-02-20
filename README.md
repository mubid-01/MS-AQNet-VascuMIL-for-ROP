# Context-Aware Asymmetric Ensembling for ROP Screening

**Official PyTorch Implementation**

> **Paper Title:** *Context-Aware Asymmetric Ensembling: Integrating Active Query Mechanisms and Vascular Multiple Instance Learning for Interpretable Retinopathy of Prematurity Screening*

---

## 1. Introduction & Motivation

Retinopathy of Prematurity (ROP) is a leading cause of childhood blindness. Automated screening is complicated by the dual nature of the disease:
1.  **Global Structural Anomalies:** Ridges, demarcation lines, and detachments (Stage-based disease).
2.  **Local Micro-Vascular Irregularities:** Arteriolar tortuosity and venous dilation (Plus disease).

Standard Deep Learning approaches often fail on ROP datasets due to **extreme class imbalance**, **limited sample sizes**, and **resolution trade-offs** (downsizing destroys vascular details). Furthermore, most models are "Black Boxes" that ignore critical clinical priors like Gestational Age (GA) and Birth Weight (BW).

### Our Solution: The Asymmetric Ensemble
We propose a novel framework that decouples the diagnostic task into two specialized streams using a **"Resolution Bifurcation"** strategy:

*   **Structure Stream (MS-AQNet):** Operates on global images ($384 \times 384$). Features a novel **Active Query Mechanism** where clinical metadata explicitly "queries" visual feature maps to localize risk-relevant structural pathology.
*   **Texture Stream (VascuMIL):** Operates on high-resolution ($768 \times 768$) patch bags. Utilizes **Vascular Topology Maps (VMAP)** and **Gated Attention MIL** to detect micro-vascular Plus Disease independent of clinical history.

**Key Results (N=188):**
*   **Macro F1 (Diagnosis):** 0.93
*   **ROC AUC (Plus Disease):** 0.999
*   **Severe ROP Sensitivity:** 96.3%

---

## 2. Methodology Overview

### Phase A: Intelligent Data Engineering
We employ a rigorous preprocessing pipeline to standardize data from heterogeneous devices (Clarity, Natus, Phoenix):
*   **Morphological Erosion:** Removes circular aperture artifacts.
*   **Vascular Topology Extraction:** Generates VMAPs using Frangi filters on the Green channel.
*   **4-Channel Stacking:** RGB patches are concatenated depth-wise with VMAPs to enforce geometric awareness in the CNN.

### Phase B: The Structure Specialist (MS-AQNet)
*   **Backbone:** EfficientNet-B0 with **Frozen Batch Normalization** (to stabilize small-batch training).
*   **Active Query:** Clinical metadata is projected into a latent vector that computes spatial attention maps via dot-product similarity.
*   **FiLM Modulation:** Global features are scaled and shifted based on patient risk profiles.

### Phase C: The Texture Specialist (VascuMIL)
*   **Architecture:** Multiple Instance Learning (MIL) with Gated Attention (Ilse et al.).
*   **Input:** Bags of 24 high-resolution patches ($224 \times 224$).
*   **Mechanism:** Assigns attention scores to "sick" patches (tortuosity) while suppressing background noise.

---

## 3. Dataset

This project utilizes the public dataset: **"Retinal Image Dataset of Infants and Retinopathy of Prematurity"** (*Timkovič et al., Scientific Data 2024*).

### Setup
1.  Download the dataset.
2.  Ensure your directory structure looks like this:

### Repository Hierarchy
```text
CAA-Ensemble-ROP/
│
├── data_preparation/          # Data Engineering scripts
│   ├── create_splits.py       # Patient-isolated stratified K-fold splitter
│   └── make_patches.py        # VMAP generation and patch extraction
│
├── models/                    # Architecture definitions
│   ├── model_ms_aqnet.py      # MS-AQNet (Structure)
│   ├── model_vascumil.py      # VascuMIL (Texture)
│   └── model_fusion.py        # Fusion Meta-Learner
│
├── utils/                     # Shared loaders & utilities
│   ├── preprocessing.py       # Single source of truth for image processing
│   ├── loader_structure.py    # Dataloader for MS-AQNet
│   └── loader_texture.py      # Dataloader for VascuMIL
│
├── visualization/             # Explainability ("Glass Box") tools
│   └── viz_combined_explainability.py
│
├── train_stage1_structure.py  # Engine for Structure Stream
├── train_stage2_texture.py    # Engine for Texture Stream
├── train_stage3_fusion.py     # Engine for Synergistic Ensemble
├── requirements.txt           
└── README.md
```
## 4. Execution Pipeline

### Step 1: Data Splitting
Generate the patient-grouped stratified folds. This isolates a 10% held-out test set and creates 5-fold cross-validation splits.

```bash
python data_preparation/create_splits.py --data_dir ./data/images --out_dir ./data/splits
```
### Step 2: Patch & VMAP Extraction
Generate the high-resolution input for the Texture Stream.

```bash
python data_preparation/make_patches.py \
    --splits_dir ./data/splits \
    --out_dir ./data/mil_dataset/metadata \
    --patch_dir ./data/mil_dataset/images \
    --fold 0 \
    --save_patches \
    --preproc_target_square 768
```
### Step 3: Train MS-AQNet (Structure Stream)

```bash
python train_stage1_structure.py \
    --folds_dir ./data/splits \
    --out_dir ./output/ms_models \
    --fold 0 \
    --epochs 30 \
    --batch_size 16
```

### Step 4: Train VascuMIL (Texture Stream)

```bash
python train_stage2_texture.py \
    --train_csv ./data/mil_dataset/metadata/patches_train.csv \
    --val_csv ./data/mil_dataset/metadata/patches_val.csv \
    --cache_root ./data/mil_dataset/images \
    --out_dir ./output/mil_models \
    --bag_size 24 \
    --batch_size 4 \
    --static_pos_weight 1.0
```

### Step 5: Train Synergistic Ensemble

```bash
python train_stage3_fusion.py \
    --ms_path ./output/ms_models/best_ms_clinical.pth \
    --mil_path ./output/mil_models/best_mil_clinical.pth \
    --val_csv ./data/splits/fold_0/val.csv \
    --patches_csv ./data/mil_dataset/metadata/patches_val.csv \
    --cache_root ./data/mil_dataset/images \
    --out_dir ./output/final_ensemble```
```

## 5. Clinical Explainability ("Glass Box")

This framework includes a suite of visualization tools to validate model decision-making against clinical landmarks.

To generate the main explainability figure:

```bash
python visualization/viz_combined_explainability.py \
    --ms_path ./output/ms_models/best_ms_clinical.pth \
    --mil_path ./output/mil_models/best_mil_clinical.pth \
    --cache_root ./data/mil_dataset/images \
    --val_csv ./data/splits/fold_0/val.csv \
    --patches_csv ./data/mil_dataset/metadata/patches_val.csv```
```
## Acknowledgments
We thank the authors of the Scientific Data ROP dataset (Timkovič et al.) for making their data publicly available.
```

