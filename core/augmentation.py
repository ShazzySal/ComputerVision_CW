"""Optional NumPy augmentation utilities for RetinaTrace AI.

The notebook's Keras layers are the authoritative training augmentation pipeline.
This module is a separate optional utility and is not used by the notebook training path.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import cv2


class FundusAugmentor:
    """Optional fundus image transforms; label and lesion preservation are not guaranteed.
    """

    def __init__(
        self,
        rotation_range: Tuple[float, float] = (-19.8, 19.8),
        horizontal_flip_prob: float = 0.5,
        vertical_flip_prob: float = 0.5,
        zoom_range: Tuple[float, float] = (0.90, 1.10),
        brightness_range: Tuple[float, float] = (0.85, 1.15),
        contrast_range: Tuple[float, float] = (0.85, 1.15),
        coarse_dropout_prob: float = 0.0,
        num_dropout_holes: int = 4,
        max_dropout_size: int = 16,
        seed: Optional[int] = None,
    ):
        self.rotation_range = rotation_range
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.zoom_range = zoom_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.coarse_dropout_prob = coarse_dropout_prob
        self.num_dropout_holes = num_dropout_holes
        self.max_dropout_size = max_dropout_size
        self.rng = np.random.default_rng(seed)

    def random_rotate(self, img: np.ndarray) -> np.ndarray:
        """Apply planar rotation within the configured degree range.
        
        Clinical Justification: Fundus photographs have circular rotational symmetry;
        patient head tilt and camera angle orientation do not alter the diagnostic stage.
        """
        angle = self.rng.uniform(self.rotation_range[0], self.rotation_range[1])
        h, w = img.shape[:2]
        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    def random_flip(self, img: np.ndarray) -> np.ndarray:
        """Apply configured horizontal and vertical reflections.
        """
        if self.rng.random() < self.horizontal_flip_prob:
            img = cv2.flip(img, 1)
        if self.rng.random() < self.vertical_flip_prob:
            img = cv2.flip(img, 0)
        return img

    def random_zoom(self, img: np.ndarray) -> np.ndarray:
        """Apply slight scale variations [0.90, 1.10].
        
        Clinical Justification: Simulates variations in optical magnification and
        patient working distance (30-50 cm) across different non-mydriatic camera models.
        """
        scale = self.rng.uniform(self.zoom_range[0], self.zoom_range[1])
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if scale > 1.0:
            # Center crop
            top = (new_h - h) // 2
            left = (new_w - w) // 2
            return resized[top:top + h, left:left + w]
        else:
            # Pad
            padded = np.zeros_like(img)
            top = (h - new_h) // 2
            left = (w - new_w) // 2
            padded[top:top + new_h, left:left + new_w] = resized
            return padded

    def random_photometric_jitter(self, img: np.ndarray) -> np.ndarray:
        """Apply subtle brightness and contrast adjustments.
        
        Clinical Justification: Simulates inter-clinic differences in strobe flash
        voltage, pupil dilation quality, and lens transmission properties.
        """
        alpha = self.rng.uniform(self.contrast_range[0], self.contrast_range[1])
        beta = self.rng.uniform(
            self.brightness_range[0] - 1.0,
            self.brightness_range[1] - 1.0,
        ) * 255.0
        adjusted = img.astype(np.float32) * alpha + beta
        return np.clip(adjusted, 0, 255).astype(img.dtype)

    def random_coarse_dropout(self, img: np.ndarray) -> np.ndarray:
        """Apply optional random occlusion patches when coarse dropout is enabled.
        """
        if self.rng.random() >= self.coarse_dropout_prob:
            return img
        out = img.copy()
        h, w = img.shape[:2]
        for _ in range(self.num_dropout_holes):
            hh = self.rng.integers(4, self.max_dropout_size)
            ww = self.rng.integers(4, self.max_dropout_size)
            y = self.rng.integers(0, max(1, h - hh))
            x = self.rng.integers(0, max(1, w - ww))
            out[y:y + hh, x:x + ww] = 0
        return out

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Execute full augmentation sequence on an input image."""
        aug = self.random_rotate(img)
        aug = self.random_flip(aug)
        aug = self.random_zoom(aug)
        aug = self.random_photometric_jitter(aug)
        aug = self.random_coarse_dropout(aug)
        return aug


# ── Dataset Balancing & Weighting Utilities ──────────────────────────────────

def compute_balanced_class_weights(
    labels: Union[List[int], np.ndarray],
    num_classes: int = 5,
    method: str = "balanced",
    beta: float = 0.9999,
) -> Dict[int, float]:
    """Compute normalized class weight penalties to counter dataset imbalance.
    
    Parameters
    ----------
    labels : array-like of shape (N,)
        Integer class targets [0..num_classes-1].
    num_classes : int
        Total number of diagnostic classes (default: 5).
    method : str
        'balanced' (Standard inverse-frequency weighting):
            w_c = N / (C * N_c)
        'effective_samples' (Cui et al., CVPR 2019):
            w_c = (1 - beta) / (1 - beta^{N_c})
    beta : float
        Hyperparameter for effective number of samples (default: 0.9999).
        
    Returns
    -------
    Dict[int, float]
        Mapping from class index to normalized penalty weight.
    """
    labels = np.asarray(labels, dtype=int)
    total_samples = len(labels)
    class_counts = np.bincount(labels, minlength=num_classes)
    
    weights = {}
    if method == "effective_samples":
        # Effective Number of Samples weighting (Cui et al., 2019)
        effective_num = 1.0 - np.power(beta, class_counts)
        raw_weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
        # Normalize weights so mean weight = 1.0
        norm_factor = np.sum(raw_weights) / num_classes
        for c in range(num_classes):
            weights[c] = float(raw_weights[c] / norm_factor)
    else:
        # Standard inverse frequency heuristic (heuristic from scikit-learn)
        for c in range(num_classes):
            count = class_counts[c]
            if count > 0:
                weights[c] = float(total_samples / (num_classes * count))
            else:
                weights[c] = 1.0
                
    return weights


def create_sample_weight_vector(
    labels: Union[List[int], np.ndarray],
    class_weights: Dict[int, float],
) -> np.ndarray:
    """Generate sample-level weight vector matching training batch labels."""
    labels = np.asarray(labels, dtype=int)
    return np.array([class_weights.get(lbl, 1.0) for lbl in labels], dtype=np.float32)


def get_stratified_batch_indices(
    labels: np.ndarray,
    batch_size: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate balanced mini-batch indices sampling equally across all classes."""
    if rng is None:
        rng = np.random.default_rng()
    num_classes = len(np.unique(labels))
    samples_per_class = max(1, batch_size // num_classes)
    selected_indices = []
    
    for c in range(num_classes):
        c_indices = np.where(labels == c)[0]
        if len(c_indices) > 0:
            chosen = rng.choice(c_indices, size=samples_per_class, replace=True)
            selected_indices.extend(chosen)
            
    rng.shuffle(selected_indices)
    return np.array(selected_indices[:batch_size])
