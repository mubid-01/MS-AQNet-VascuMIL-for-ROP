#!/usr/bin/env python3
"""
Module: model_fusion.py
Description: 
    Implementation of the Synergistic Fusion Meta-Learner.
    
    This model serves as the final integration stage of the Asymmetric Ensemble.
    It performs cross-modal calibration by processing the joint distribution of:
    1. Structural Logits (from MS-AQNet)
    2. Textural Logits (from VascuMIL)
    3. Clinical Metadata (GA, BW, PA)
    
    The network is designed as a Multi-Task MLP that simultaneously resolves 
    Broad ROP Diagnosis (4-class) and Plus Disease (Binary).
"""

import torch
import torch.nn as nn

def count_parameters(model):
    """Calculates total trainable parameters in the fusion layer."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class FusionMetaLearner(nn.Module):
    """
    A multi-task MLP for synergistic feature fusion.
    Processes concatenated logits and clinical priors to resolve 
    diagnostic discordance.
    """
    def __init__(self, ms_dim: int = 4, mil_dim: int = 1, tab_dim: int = 3):
        super(FusionMetaLearner, self).__init__()
        
        # Total Input: 4 (Structure) + 1 (Texture) + 3 (Clinical) = 8
        self.input_dim = ms_dim + mil_dim + tab_dim
        
        # Non-linear Calibration Backbone
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True)
        )
        
        # Head 1: Broad Diagnosis (4-class classification)
        self.diagnosis_head = nn.Linear(16, 4)
        
        # Head 2: Plus Disease (Binary regression/classification)
        self.plus_head = nn.Linear(16, 1)
        
    def forward(self, ms_logits: torch.Tensor, mil_logit: torch.Tensor, tabular: torch.Tensor):
        """
        Args:
            ms_logits: Tensor [B, 4] from MS-AQNet.
            mil_logit: Tensor [B, 1] from VascuMIL.
            tabular:   Tensor [B, 3] Normalized clinical metadata.
        Returns:
            diag_logits: [B, 4]
            plus_logit:  [B, 1]
        """
        # Concatenate signals into a unified fusion vector
        combined = torch.cat([ms_logits, mil_logit, tabular], dim=1)
        
        # Process through hidden layers
        features = self.fusion_layer(combined)
        
        # Resolve multi-task objectives
        diag_logits = self.diagnosis_head(features)
        plus_logit = self.plus_head(features)
        
        return diag_logits, plus_logit

if __name__ == "__main__":
    # --- Structural Unit Test ---
    print("[INFO] Initializing Fusion Meta-Learner...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Init dimensions
    MS_DIM, MIL_DIM, TAB_DIM = 4, 1, 3
    model = FusionMetaLearner(MS_DIM, MIL_DIM, TAB_DIM).to(device)
    
    # Dummy Logits
    dummy_ms = torch.randn(4, MS_DIM).to(device)
    dummy_mil = torch.randn(4, MIL_DIM).to(device)
    dummy_tab = torch.randn(4, TAB_DIM).to(device)
    
    print(f"   Input Dimension: {model.input_dim}")
    print(f"   Trainable Params: {count_parameters(model):,}")
    
    # Forward Pass
    try:
        diag, plus = model(dummy_ms, dummy_mil, dummy_tab)
        print(" Fusion logic verified.")
        print(f"   Diagnosis Output Shape: {diag.shape}")
        print(f"   Plus Disease Output Shape: {plus.shape}")
    except Exception as e:
        print(f" Forward pass failed: {e}")