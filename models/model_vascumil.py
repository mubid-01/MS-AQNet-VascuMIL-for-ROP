#!/usr/bin/env python3
"""
Module: model_vascumil.py
Description: 
    Implementation of the Vascular-Aware Multiple Instance Learning (VascuMIL) Network.
    
    Architecture Details:
    - Backbone: EfficientNet-B0 with a modified 4-channel input stem.
    - Instance Encoding: High-resolution patch processing within a bag structure.
    - Gated Attention: Ilse et al. mechanism using tanh and sigmoid branches.
    - Projection: Latent manifold alignment using LayerNormalization and GELU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

def count_parameters(model):
    """Calculates the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def convert_to_4channel(model):
    """
    Modifies the input stem of a CNN to accept 4 channels (RGB + VMAP).
    Preserves ImageNet weights for RGB channels and initializes the 
    vascular channel via Kaiming Normal initialization.
    """
    old_conv = model.conv_stem
    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None)
    )

    with torch.no_grad():
        # Copy original RGB weights
        new_conv.weight.data[:, :3, :, :] = old_conv.weight.data.clone()
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data.clone()
        # Initialize 4th channel (Vessel Map)
        nn.init.kaiming_normal_(new_conv.weight.data[:, 3:, :, :], mode='fan_out', nonlinearity='relu')

    model.conv_stem = new_conv

class InstanceEncoder(nn.Module):
    """Encodes localized patches into a latent feature space."""
    def __init__(self, pretrained=True, proj_dim=256, freeze_backbone=False):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            features_only=True,
            out_indices=(3,)
        )
        convert_to_4channel(self.backbone)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.proj_dim = int(proj_dim)
        self.built = False
        self.proj = None

    def build(self, sample_input=(1, 4, 224, 224), device="cpu"):
        self.to(device)
        with torch.no_grad():
            x = torch.randn(*sample_input, device=device)
            feat = self.backbone(x)[0]
        C = feat.shape[1]

        self.proj = nn.Sequential(
            nn.Linear(C, self.proj_dim),
            nn.LayerNorm(self.proj_dim),
            nn.GELU()
        ).to(device)
        self.built = True

    def forward(self, x):
        if not self.built:
            raise RuntimeError("InstanceEncoder must be built before forward pass.")
        feats = self.backbone(x)[0]   
        pooled = F.adaptive_avg_pool2d(feats, 1).view(feats.shape[0], -1)
        return self.proj(pooled)

class GatedAttention(nn.Module):
    """Gated Attention pooling as proposed by Ilse et al."""
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, H, mask=None):
        Vh = torch.tanh(self.V(H))      
        Uh = torch.sigmoid(self.U(H))   
        a = self.w(Vh * Uh).squeeze(-1) 

        if mask is not None:
            # Use -10000.0 for numerical stability in FP16/AMP
            a = a.masked_fill(~mask.bool(), -10000.0)

        A = torch.softmax(a, dim=1)     
        bag_emb = torch.sum(A.unsqueeze(-1) * H, dim=1)  
        return bag_emb, A

class VascuMIL(nn.Module):
    def __init__(
        self,
        pretrained=True,
        proj_dim=256,
        att_hidden=128,
        mlp_hidden=128,
        dropout=0.25,
        freeze_backbone=False,
        device="cpu"
    ):
        super().__init__()
        self.device_name = device
        self.encoder = InstanceEncoder(pretrained, proj_dim, freeze_backbone)
        self.att = GatedAttention(proj_dim, att_hidden)
        self.head = nn.Sequential(
            nn.Linear(proj_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1)
        )
        self.built = False

    def build(self, sample_input=(1, 4, 224, 224), device=None):
        device = device or self.device_name
        self.encoder.build(sample_input, device=device)
        self.to(device)
        self.built = True

    def forward(self, images, lengths):
        """
        Args:
            images: [B, M, 4, H, W] tensor.
            lengths: [B] tensor of valid instances.
        """
        if not self.built:
            raise RuntimeError("Model must be built before execution.")

        B, M, C, H, W = images.shape
        device = images.device
        
        # Parallel processing of all instances in the bag
        flat = images.view(B * M, C, H, W)
        emb = self.encoder(flat)   
        
        H_inst = emb.view(B, M, -1)    

        # Masking for variable bag lengths
        idxs = torch.arange(M, device=device).unsqueeze(0)   
        mask = idxs < lengths.unsqueeze(1)                   

        bag_emb, attn = self.att(H_inst, mask=mask)
        logits = self.head(bag_emb).squeeze(-1)

        return {"logits": logits, "attn": attn, "bag_emb": bag_emb}

if __name__ == "__main__":
    # --- Structural Unit Test ---
    print("[INFO] Testing VascuMIL Architecture...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VascuMIL(pretrained=False)
    model.build(device=device)
    
    dummy_bag = torch.randn(2, 10, 4, 224, 224).to(device)
    dummy_lens = torch.tensor([10, 5]).to(device)
    
    out = model(dummy_bag, dummy_lens)
    print(f" Success. Parameters: {count_parameters(model)/1e6:.2f} M")
    print(f"   Output Logits: {out['logits'].shape}")
    print(f"   Attention Map: {out['attn'].shape}")