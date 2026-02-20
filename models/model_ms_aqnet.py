#!/usr/bin/env python3
"""
Module: model_ms_aqnet.py
Description: 
    PyTorch implementation of the Multi-Scale Active Query Network (MS-AQNet).
    
    This model serves as the 'Structure Specialist' for ROP screening. 
    It incorporates three hierarchical scales of visual features and 
    contextualizes them using patient metadata through an Active Query 
    Mechanism and FiLM modulation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

def count_parameters(model):
    """Returns the total count of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def spatial_softmax(x, eps=1e-8):
    """
    Computes Softmax across the spatial dimension (HW) for attention maps.
    Uses max-subtraction for numerical stability in mixed-precision training.
    """
    # x shape expected: [B, HW]
    mx = x.max(-1, keepdim=True)[0]
    ex = torch.exp(x - mx)
    return ex / (ex.sum(-1, keepdim=True) + eps)

class GN_Block(nn.Module):
    """
    Linear -> GroupNorm -> ReLU -> Dropout block.
    GroupNormalization ensures stable training statistics for small batch sizes.
    """
    def __init__(self, in_dim, out_dim, groups=8, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GroupNorm(groups, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class QueryMLP(nn.Module):
    """Projects 1x3 clinical metadata into a latent query vector for attention."""
    def __init__(self, in_dim=3, hidden=64, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class MS_AQNet(nn.Module):
    def __init__(
        self,
        backbone_name="efficientnet_b0",
        pretrained=True,
        query_dim=64,
        kv_dim=64,
        freeze_backbone=True,
        drop_path_rate=0.2,
        device="cpu"
    ):
        super().__init__()
        self.device_name = device
        self.query_dim = query_dim
        self.kv_dim = kv_dim

        # 1. Feature Extractor (Backbone)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(2, 3, 4), # Fine, Mid, Coarse
            drop_path_rate=drop_path_rate
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # 2. Metadata Projectors (Active Query)
        self.q_fine   = QueryMLP(3, query_dim, query_dim)
        self.q_mid    = QueryMLP(3, query_dim, query_dim)
        self.q_coarse = QueryMLP(3, query_dim, query_dim)

        # Placeholders for dynamic layers (initialized in build())
        self.key_proj = None
        self.val_proj = None
        self.pool_reduce = None
        
        # Learnable gating alpha (Initial -2.0 biases model towards visual input)
        self.alpha = nn.Parameter(torch.tensor([-2.0, -2.0, -2.0]))

        self.cls_head = None
        self.aux1 = None
        self.aux2 = None
        self.film = None 

        self.built = False

    def build(self, sample_input=(1, 3, 384, 384), device=None, num_classes=4):
        """Initializes layers that depend on the backbone's specific channel output."""
        device = device or self.device_name
        self.to(device)
        with torch.no_grad():
            dummy = torch.randn(*sample_input).to(device)
            feats = self.backbone(dummy)
        
        channels = [f.shape[1] for f in feats]

        # Key/Value Standardizers
        self.key_proj = nn.ModuleList([nn.Conv2d(c, self.query_dim, 1).to(device) for c in channels])
        self.val_proj = nn.ModuleList([nn.Conv2d(c, self.kv_dim, 1).to(device) for c in channels])
        self.pool_reduce = nn.ModuleList([nn.Linear(c, self.kv_dim).to(device) for c in channels])

        pooled_dim = self.kv_dim * 3
        hidden = 256

        # FiLM Layer (Global Fusion)
        self.film = nn.Sequential(
            nn.Linear(3, 32), 
            nn.ReLU(), 
            nn.Linear(32, pooled_dim * 2)
        ).to(device)

        # Classification Heads
        self.cls_head = nn.Sequential(
            GN_Block(pooled_dim, hidden),
            nn.Linear(hidden, num_classes)
        ).to(device)

        self.aux1 = nn.Sequential(GN_Block(self.kv_dim, 64), nn.Linear(64, num_classes)).to(device)
        self.aux2 = nn.Sequential(GN_Block(self.kv_dim, 64), nn.Linear(64, num_classes)).to(device)

        self.built = True
        self.to(device)
    
    def train(self, mode=True):
        """
        Force Batch Normalization layers in the backbone to stay in evaluation mode.
        This prevents statistic corruption in small-batch training.
        """
        super().train(mode)
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def forward(self, img, meta):
        """
        img: [B, 3, 384, 384]
        meta: [B, 3] Clinical Context
        """
        if not self.built:
            raise RuntimeError("Model must be built using model.build() before forward pass.")

        feats = self.backbone(img)
        pooled = []
        att_maps = []

        queries = [self.q_fine(meta), self.q_mid(meta), self.q_coarse(meta)]

        # --- Multiscale Active Query Mechanism ---
        for i in range(3):
            Vi = feats[i]
            K = self.key_proj[i](Vi)
            V = self.val_proj[i](Vi)

            B, d, H, W = K.shape
            Kf = K.view(B, d, -1)
            Q = queries[i]

            # Matrix similarity via einsum: [B, d] x [B, d, HW] -> [B, HW]
            logits = torch.einsum("bd,bdn->bn", Q, Kf) / math.sqrt(d)
            A = spatial_softmax(logits)
            Amap = A.view(B, 1, H, W)
            att_maps.append(Amap)

            # Residual Gating Logic
            gate = torch.sigmoid(self.alpha[i])
            Vi_mod = (1 - gate) * Vi + gate * (Amap * Vi)

            # Global Average Pooling
            p = F.adaptive_avg_pool2d(Vi_mod, 1).view(B, -1)
            if p.shape[1] != self.kv_dim:
                p = self.pool_reduce[i](p)
            pooled.append(p)

        # --- Global Aggregation & FiLM ---
        emb = torch.cat(pooled, dim=1) # [B, 192]
        
        gamma_beta = self.film(meta)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        emb = gamma * emb + beta

        # --- Prediction ---
        logits = self.cls_head(emb)
        aux1 = self.aux1(pooled[0])
        aux2 = self.aux2(pooled[1])

        return {
            "stage_logits": logits, 
            "aux1": aux1, 
            "aux2": aux2, 
            "att_maps": att_maps
        }

if __name__ == "__main__":
    # Unit Test
    print("[INFO] Initializing MS-AQNet...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MS_AQNet(pretrained=False, freeze_backbone=True, device=device)
    model.build(sample_input=(1, 3, 384, 384), device=device, num_classes=4)
    
    print(f"   Trainable Params: {count_parameters(model):,}")
    
    dummy_img = torch.randn(2, 3, 384, 384).to(device)
    dummy_meta = torch.randn(2, 3).to(device)
    
    print("[INFO] Testing forward pass...")
    out = model(dummy_img, dummy_meta)
    print("✅ Success.")
    print(f"   Stage Logits Shape: {out['stage_logits'].shape}")
    print(f"   Attention Maps:     {len(out['att_maps'])} scales generated.")