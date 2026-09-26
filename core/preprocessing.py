"""Fundus image preprocessing utilities."""

from typing import Union

import cv2
import numpy as np

from core.config import AppConfig


def crop_image_from_gray(img: np.ndarray, threshold: int = 7, tol: int = 7) -> np.ndarray:
    """Strip non-informative circular black borders from fundus photographs."""
    if img.ndim == 2:
        mask = img > threshold
    else:
        mask = img[:, :, 1] > threshold

    if not mask.any():
        return img

    row_mask = mask.any(axis=1)
    col_mask = mask.any(axis=0)
    rmin, rmax = np.where(row_mask)[0][[0, -1]]
    cmin, cmax = np.where(col_mask)[0][[0, -1]]

    rmin = max(0, rmin - tol)
    rmax = min(img.shape[0] - 1, rmax + tol)
    cmin = max(0, cmin - tol)
    cmax = min(img.shape[1] - 1, cmax + tol)

    cropped = img[rmin:rmax + 1, cmin:cmax + 1]
    return cropped if cropped.size > 0 and min(cropped.shape[:2]) >= 10 else img


def ben_graham_enhance(img: np.ndarray) -> np.ndarray:
    """Apply Ben Graham spatial illumination normalization."""
    ksize = int(2 * round(4 * AppConfig.BEN_GRAHAM_SIGMA) + 1)
    blurred = cv2.GaussianBlur(img, (ksize, ksize), AppConfig.BEN_GRAHAM_SIGMA)
    enhanced = cv2.addWeighted(
        img,
        AppConfig.BEN_GRAHAM_ALPHA,
        blurred,
        AppConfig.BEN_GRAHAM_BETA,
        AppConfig.BEN_GRAHAM_GAMMA,
    )
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def denoise_fundus(img: np.ndarray, diameter: int = 5, sigma_color: float = 20.0, sigma_space: float = 20.0) -> np.ndarray:
    """Apply conservative edge-preserving bilateral filtering before enhancement.
    
    Preserves fine retinal vessel boundaries and microaneurysms while smoothing
    sensor noise and compression artifacts.
    """
    if img.ndim == 2:
        return cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)
    return cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """Apply Contrast-Limited Adaptive Histogram Equalization (CLAHE).
    
    For RGB images, operates in the CIELAB color space on the Luminance (L) channel
    to optimize microvascular contrast without distorting chromatic diagnostic features.
    For grayscale images, operates directly on the 2D intensity plane.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if img.ndim == 2:
        return clahe.apply(img)
    elif img.ndim == 3 and img.shape[2] == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return img


def enhance_edges(img: np.ndarray, strength: float = 1.2, sigma: float = 3.0) -> np.ndarray:
    """Apply high-boost unsharp masking to sharpen microvascular and lesion boundaries.
    
    Subtracts a low-pass Gaussian blur to isolate high-frequency edge gradients,
    enhancing subtle microaneurysms and exudate boundaries against the fundus background.
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def preprocess_image(
    image_input: Union[str, np.ndarray],
    apply_denoise: bool = False,
    apply_clahe_enhancement: bool = False,
    apply_edge_enhancement: bool = False,
) -> np.ndarray:
    """Convert an input image into a normalized 224x224 RGB tensor.
    
    Applies the comprehensive medical preprocessing pipeline:
    1. Color-space standardization (RGBA/Gray -> RGB)
    2. Circular dark border cropping
    3. Scale-adaptive interpolation resizing (224x224)
    4. Optional bilateral edge-preserving denoising
    5. Optional CLAHE contrast-limited adaptive histogram equalization
    6. Optional high-boost unsharp mask edge enhancement
    7. Ben Graham spatial color/illumination normalization
    8. Intensity normalization to [0.0, 1.0] float32
    """
    if isinstance(image_input, str):
        bgr = cv2.imread(image_input)
        if bgr is None:
            raise ValueError(f"Could not load image: {image_input}")
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        img = np.asarray(image_input)

    if img.size == 0:
        raise ValueError("Input image is empty.")
    if img.ndim not in (2, 3):
        raise ValueError(f"Unsupported image shape: {img.shape}. Expected 2D grayscale or 3D RGB/RGBA array.")
    if img.ndim == 3 and (img.shape[0] == 0 or img.shape[1] == 0):
        raise ValueError(f"Input image has a zero-sized dimension: {img.shape}.")
    if img.ndim == 3 and img.shape[2] not in (1, 3, 4):
        raise ValueError(f"Unsupported channel count: {img.shape[2]}. Expected 1, 3, or 4 channels.")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    cropped = crop_image_from_gray(img)
    if cropped.size == 0 or cropped.shape[0] == 0 or cropped.shape[1] == 0:
        raise ValueError(f"Image crop produced an empty result: {cropped.shape}.")

    h, w = cropped.shape[:2]
    interp = cv2.INTER_AREA if h > AppConfig.IMG_SIZE or w > AppConfig.IMG_SIZE else cv2.INTER_LINEAR
    resized = cv2.resize(cropped, (AppConfig.IMG_SIZE, AppConfig.IMG_SIZE), interpolation=interp)
    if apply_denoise:
        resized = denoise_fundus(resized)
    if apply_clahe_enhancement:
        resized = apply_clahe(resized)
    if apply_edge_enhancement:
        resized = enhance_edges(resized)
    enhanced = ben_graham_enhance(resized)
    return enhanced.astype(np.float32) / 255.0
