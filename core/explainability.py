"""Explainability, retrieval, and classical computer-vision utilities."""

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import tensorflow as tf

from core.config import AppConfig
from core.models import embedding_extractor, gradcam_model, ref_embeddings, ref_fps, ref_labels, unet_model


def compute_gradcam(img_tensor: np.ndarray, pred_index: int) -> np.ndarray:
    tensor = tf.convert_to_tensor(img_tensor[np.newaxis, ...])
    with tf.GradientTape() as tape:
        tape.watch(tensor)
        conv_outputs, predictions = gradcam_model(tensor, training=False)
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.squeeze(conv_outputs[0] @ pooled_grads[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap.numpy()


def overlay_heatmap(rgb_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = rgb_img.shape[:2]
    heat_uint8 = (cv2.resize(heatmap, (w, h)) * 255).astype(np.uint8)
    heat_color = cv2.cvtColor(cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    base = (np.clip(rgb_img, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.addWeighted(heat_color, alpha, base, 1.0 - alpha, 0)


def segment_retinal_lesions(preproc_img: np.ndarray, heatmap: np.ndarray, stage: int) -> Tuple[np.ndarray, float]:
    base = (np.clip(preproc_img, 0.0, 1.0) * 255).astype(np.uint8)
    if stage == 0:
        return base, 0.0
    raw_mask = unet_model(preproc_img[np.newaxis, ...], training=False).numpy()[0, :, :, 0]
    gated = (raw_mask > 0.35) & (cv2.resize(heatmap, (AppConfig.IMG_SIZE, AppConfig.IMG_SIZE)) > 0.30)
    visible_pixels = np.sum(np.mean(base, axis=2) > 10)
    lesion_ratio = float(np.sum(gated) / max(visible_pixels, 1) * 100.0)
    overlay = base.copy()
    overlay[gated] = [0, 255, 80]
    return cv2.addWeighted(overlay, 0.70, base, 0.30, 0), lesion_ratio


def generate_quadrant_description(heatmap: np.ndarray, stage: int) -> Tuple[str, Dict[str, float], str, float]:
    h, w = heatmap.shape
    mid_y, mid_x = h // 2, w // 2
    quadrants = {
        "Superior-Temporal": float(np.mean(heatmap[:mid_y, :mid_x])),
        "Superior-Nasal": float(np.mean(heatmap[:mid_y, mid_x:])),
        "Inferior-Temporal": float(np.mean(heatmap[mid_y:, :mid_x])),
        "Inferior-Nasal": float(np.mean(heatmap[mid_y:, mid_x:])),
    }
    peak_name, peak_val = max(quadrants.items(), key=lambda item: item[1])
    desc = (f"**Peak Pathological Focus:** Grad-CAM localized maximum lesion density in the **[{peak_name}]** quadrant "
            f"(intensity index: `{peak_val:.2f}`), serving as the primary morphological driver for the **{AppConfig.CLASS_NAMES[stage]}** classification.")
    return desc, quadrants, peak_name, peak_val


def find_similar_cases(query_arr: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
    if len(ref_embeddings) == 0:
        return []
    query_embedding = embedding_extractor(query_arr[np.newaxis, ...], training=False).numpy()
    query_embedding /= np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-10
    similarities = np.dot(ref_embeddings, query_embedding.T).squeeze()
    results = []
    for rank, index in enumerate(np.argsort(similarities)[::-1][:k], start=1):
        stage = int(ref_labels[index])
        results.append({"rank": rank, "filepath": ref_fps[index], "stage": stage,
                        "stage_name": AppConfig.CLASS_NAMES[stage], "similarity": float(similarities[index])})
    return results


def extract_classical_cv_biomarkers(preproc_img: np.ndarray) -> Dict[str, Any]:
    gray = cv2.cvtColor((np.clip(preproc_img, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    sobel_x = cv2.Sobel(equalized, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(equalized, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tophat = cv2.morphologyEx(equalized, cv2.MORPH_TOPHAT, kernel)
    return {"sobel_edge_density": float(np.mean(grad_mag > np.percentile(grad_mag, 85)) * 100.0),
            "morph_candidate_pct": float(np.mean(tophat > 25) * 100.0),
            "mean_gradient": float(np.mean(grad_mag)), "clahe_status": "Calibrated (Clip=2.0)"}
