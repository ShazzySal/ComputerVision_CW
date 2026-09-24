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


def extract_retinal_vessels(preproc_img: np.ndarray) -> Dict[str, Any]:
    """Create an exploratory vessel mask from green-channel morphology."""
    base = (np.clip(preproc_img, 0.0, 1.0) * 255).astype(np.uint8)
    green = base[:, :, 1]
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(green)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    vessel_response = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(vessel_response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cleanup_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cleanup_kernel)
    overlay = base.copy()
    overlay[mask > 0] = [40, 220, 255]
    return {
        "vessel_mask": mask,
        "vessel_overlay": cv2.addWeighted(base, 0.65, overlay, 0.35, 0),
        "vessel_density": float(np.mean(mask > 0) * 100.0),
        "vessel_status": "Green-channel black-hat + Otsu morphology",
        "method_description": "Exploratory green-channel CLAHE, black-hat morphology, Otsu thresholding, and morphological cleanup; visual support only.",
    }


def localize_optic_disc(preproc_img: np.ndarray) -> Dict[str, Any]:
    """Locate a bright optic-disc candidate for visualisation only."""
    base = (np.clip(preproc_img, 0.0, 1.0) * 255).astype(np.uint8)
    green = base[:, :, 1]
    smooth = cv2.GaussianBlur(green, (15, 15), 0)
    threshold = max(170, int(np.percentile(smooth, 99.0)) - 1)
    _, candidates = cv2.threshold(smooth, threshold, 255, cv2.THRESH_BINARY)
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(green.shape[0] * green.shape[1])
    valid = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not 0.002 * image_area <= area <= 0.20 * image_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect = min(w, h) / max(w, h, 1)
        circularity = 4 * np.pi * area / max(cv2.arcLength(contour, True) ** 2, 1.0)
        if aspect >= 0.35 and circularity >= 0.20:
            valid.append(contour)

    empty = {
        "optic_disc_found": False,
        "optic_disc_center": None,
        "optic_disc_radius": None,
        "optic_disc_box": None,
        "optic_disc_mask": np.zeros(green.shape, dtype=np.uint8),
        "optic_disc_overlay": base,
        "optic_disc_score": 0.0,
        "method_description": "Bright-region thresholding with contour size, shape, and position validation; no reliable candidate found.",
        "optic_disc_method_description": "Bright-region thresholding with contour size, shape, and position validation; no reliable candidate found.",
    }
    if not valid:
        return empty

    height, width = green.shape
    def candidate_score(candidate: np.ndarray) -> float:
        x, y, w, h = cv2.boundingRect(candidate)
        area = cv2.contourArea(candidate)
        circularity = 4 * np.pi * area / max(cv2.arcLength(candidate, True) ** 2, 1.0)
        center_x = (x + w / 2) / width
        position_score = 1.0 - min(abs(center_x - 0.30) / 0.70, 1.0)
        return float(min(area / image_area * 8.0, 1.0) * min(circularity, 1.0) * position_score)

    contour = max(valid, key=candidate_score)
    area = cv2.contourArea(contour)
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return empty
    center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
    x, y, w, h = cv2.boundingRect(contour)
    circularity = float(4 * np.pi * area / max(cv2.arcLength(contour, True) ** 2, 1.0))
    score = candidate_score(contour)
    if score < 0.02:
        return empty
    radius = int(round(max(w, h) / 2.0))
    disc_mask = np.zeros(green.shape, dtype=np.uint8)
    cv2.drawContours(disc_mask, [contour], -1, 255, -1)
    overlay = base.copy()
    cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)
    cv2.circle(overlay, center, 4, (255, 0, 255), -1)
    return {"optic_disc_found": True, "optic_disc_overlay": overlay, "optic_disc_center": center,
            "optic_disc_radius": radius, "optic_disc_box": (x, y, w, h), "optic_disc_mask": disc_mask,
            "optic_disc_score": score,
            "method_description": "Bright green-channel candidate selected with contour size, elliptical-shape, and nasal-position heuristics; visual support only.",
            "optic_disc_method_description": "Bright green-channel candidate selected with contour size, elliptical-shape, and nasal-position heuristics; visual support only."}


def remove_optic_disc(preproc_img: np.ndarray, optic_disc: Dict[str, Any]) -> np.ndarray:
    """Return a visual comparison with the detected disc inpainted, never a classifier input."""
    base = (np.clip(preproc_img, 0.0, 1.0) * 255).astype(np.uint8)
    if not AppConfig.ENABLE_OPTIC_DISC_REMOVAL or not optic_disc.get("optic_disc_found"):
        return base
    mask = optic_disc.get("optic_disc_mask")
    if mask is None or not np.any(mask):
        return base
    return cv2.inpaint(base, mask, 3, cv2.INPAINT_TELEA)
