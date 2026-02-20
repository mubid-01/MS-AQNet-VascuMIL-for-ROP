#!/usr/bin/env python3
"""
Module: preprocessing.py
Description: 
    Core preprocessing utilities for the Context-Aware Asymmetric Ensemble pipeline.
    This module serves as the single source of truth for image transformations, ensuring
    strict consistency between the structural stream (MS-AQNet) and textural stream (VascuMIL).

    The pipeline performs sequential operations:
    1. Luminance normalization via Power-Law (Gamma) transformation.
    2. Morphological artifact removal (thresholding and erosion) to eliminate aperture rings.
    3. Local contrast enhancement via CLAHE in the LAB color space.
    4. Geometry-preserving padding and resolution scaling.
"""

import cv2
import numpy as np
import argparse
from PIL import Image
from pathlib import Path

# --- Global Hyperparameters ---
DEFAULT_GAMMA = 1.5
DEFAULT_CLAHE_CLIP = 2.0
DEFAULT_TILE = 8

def gamma_correct(img_rgb, gamma=DEFAULT_GAMMA):
    """Applies Power Law transform to correct luminance."""
    if img_rgb is None: 
        return None
    # Ensure contiguous array for OpenCV memory safety
    arr = np.ascontiguousarray(img_rgb).astype(np.float32) / 255.0
    corrected = np.power(arr, 1.0 / float(gamma))
    return np.ascontiguousarray((np.clip(corrected, 0.0, 1.0) * 255).astype(np.uint8))

def apply_masked_clahe(img_rgb, clip=DEFAULT_CLAHE_CLIP, tile=DEFAULT_TILE):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) 
    in conjunction with morphological erosion to remove aperture artifacts.
    """
    if img_rgb is None: 
        return None, None
    
    img_rgb = np.ascontiguousarray(img_rgb)
    
    # 1. Generate Binary Mask (Isolate retinal foreground)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # 2. Morphological Erosion (Remove peripheral artifacts)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    
    # 3. Apply CLAHE to the L-Channel of the LAB color space
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    l_cl = clahe.apply(l)
    lab_cl = cv2.merge((l_cl, a, b))
    final_rgb = cv2.cvtColor(lab_cl, cv2.COLOR_LAB2RGB)
    
    # 4. Mask Application (Zero-out noise amplified by CLAHE outside the retina)
    final_rgb = cv2.bitwise_and(final_rgb, final_rgb, mask=mask)
    
    return final_rgb, mask

def pad_to_square(img, mask=None, pad_value=(0,0,0)):
    """
    Pads the shorter spatial dimension to match the longer dimension.
    This ensures anatomical aspect ratios are preserved during resizing.
    """
    if img is None: 
        return None, None
        
    h, w = img.shape
    max_wh = max(w, h)
    
    p_left = (max_wh - w) // 2
    p_right = max_wh - w - p_left
    p_top = (max_wh - h) // 2
    p_bottom = max_wh - h - p_top
    
    img_padded = cv2.copyMakeBorder(
        img, p_top, p_bottom, p_left, p_right, 
        cv2.BORDER_CONSTANT, value=pad_value
    )
    
    if mask is not None:
        mask_padded = cv2.copyMakeBorder(
            mask, p_top, p_bottom, p_left, p_right, 
            cv2.BORDER_CONSTANT, value=0
        )
    else:
        mask_padded = None
    
    return img_padded, mask_padded

def preprocess_full_image(img_rgb,
                          gamma=DEFAULT_GAMMA,
                          clahe_clip=DEFAULT_CLAHE_CLIP,
                          clahe_tile=DEFAULT_TILE,
                          apply_mask_to_img=True, 
                          target_square=None,
                          pad_mode='constant'):
    """
    Main entry point for the preprocessing pipeline.
    
    Returns:
        tuple: (processed_rgb, processed_mask, green_channel_masked)
    """
    if img_rgb is None: 
        return None, None, None
        
    if img_rgb.dtype != np.uint8:
        img_rgb = (np.clip(img_rgb, 0.0, 1.0) * 255).astype(np.uint8)

    # Step 1 & 2: Gamma, Masking, and CLAHE
    gamma_img = gamma_correct(img_rgb, gamma)
    clahe_img, mask = apply_masked_clahe(gamma_img, clahe_clip, clahe_tile)

    # Step 3: Geometry Preservation (Pad to Square)
    square_img, square_mask = pad_to_square(clahe_img, mask)

    # Step 4: Resolution Scaling
    if target_square is not None:
        square_img = cv2.resize(square_img, (target_square, target_square), interpolation=cv2.INTER_LINEAR)
        if square_mask is not None:
            square_mask = cv2.resize(square_mask, (target_square, target_square), interpolation=cv2.INTER_NEAREST)

    # Step 5: Spectral Selection (Green Channel for downstream Vascular Analysis)
    green_masked = square_img.copy()
    if square_mask is not None:
        green_masked = 0

    return square_img, square_mask, green_masked

def crop_from_preprocessed(final_rgb, mask, x, y, w, h, pad_mode='constant'):
    """
    Extracts a localized patch from the preprocessed global image.
    Automatically handles boundary padding if the requested crop exceeds image dimensions.
    """
    if final_rgb is None: 
        return None, None, None
        
    H, W = final_rgb.shape
    x0, y0 = int(round(x)), int(round(y))
    
    def pad_and_crop(arr, fill_val):
        hA, wA = arr.shape
        l = max(0, -x0)
        t = max(0, -y0)
        r = max(0, x0 + w - wA)
        b = max(0, y0 + h - hA)
        
        # Fast path: Crop is strictly internal
        if l == 0 and t == 0 and r == 0 and b == 0:
            return arr.copy()
            
        # Boundary path: Pad before cropping
        border_type = cv2.BORDER_REFLECT_101 if pad_mode == 'reflect' else cv2.BORDER_CONSTANT
        padded = cv2.copyMakeBorder(arr, t, b, l, r, border_type, value=fill_val)
        nx, ny = x0 + l, y0 + t
        return padded.copy()

    crop_rgb = pad_and_crop(final_rgb, (0, 0, 0))
    crop_mask = pad_and_crop(mask, 0) if mask is not None else None
    
    crop_green = crop_rgb.copy()
    if crop_mask is not None:
        crop_green = 0
        
    return crop_rgb, crop_mask, crop_green

def crop_with_reflect_padding(arr, x, y, w, h, pad=0):
    """Legacy helper for patch extraction fallback mechanisms."""
    if arr is None: return None
    hA, wA = arr.shape
    l = max(0, -x)
    t = max(0, -y)
    r = max(0, x + w - wA)
    b = max(0, y + h - hA)
    
    if l == 0 and t == 0 and r == 0 and b == 0:
        return arr.copy()
    
    padded = cv2.copyMakeBorder(arr, t, b, l, r, cv2.BORDER_REFLECT_101)
    nx, ny = x + l, y + t
    return padded.copy()


if __name__ == "__main__":
    # Command-line testing utility for the module
    parser = argparse.ArgumentParser(description="Test the preprocessing pipeline on a single image.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to the input fundus image.")
    parser.add_argument("--out_dir", type=str, default="./output", help="Directory to save the processed test image.")
    parser.add_argument("--target_size", type=int, default=384, help="Target square resolution.")
    args = parser.parse_args()

    if not os.path.exists(args.img_path):
        print(f" File not found: {args.img_path}")
        sys.exit(1)

    print(f" Processing image: {args.img_path}")
    raw_bgr = cv2.imread(args.img_path)
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

    processed_rgb, _, _ = preprocess_full_image(raw_rgb, target_square=args.target_size)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "test_preprocessed.jpg")
    cv2.imwrite(out_path, cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR))
    print(f" Processed image saved to {out_path}")