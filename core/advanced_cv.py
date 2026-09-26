"""Advanced Computer Vision Analysis Modules for RetinaTrace AI.

Includes:
1. Lesion-Grad-CAM Overlap Analysis (AI Attention-Lesion Agreement)
2. Retinal Severity Map (4-Quadrant Anatomical Visual Abnormality Map)
3. Quantitative Retinal Biomarker Panel Compilation
4. Prediction-Evidence Consistency Analysis
5. Longitudinal Retinal Image Comparison (Registration & Metric Deltas)

All metrics are research/visual-support features and do not alter classifier predictions.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from core.config import AppConfig


def explain_similar_cases(similar_cases: List[Dict[str, Any]], predicted_stage: int) -> Dict[str, Any]:
    """Summarize CBR similarity metadata without presenting it as clinical evidence."""
    if not similar_cases:
        return {
            "available": False,
            "average_similarity": 0.0,
            "stage_agreement_count": 0,
            "stage_agreement_total": 0,
            "interpretation": "No verified reference cases were available for visual comparison.",
        }
    agreement = sum(case.get("stage") == predicted_stage for case in similar_cases)
    return {
        "available": True,
        "average_similarity": float(np.mean([case.get("similarity", 0.0) for case in similar_cases])),
        "stage_agreement_count": agreement,
        "stage_agreement_total": len(similar_cases),
        "interpretation": "Retrieved cases provide visual feature comparisons only; they are not independent clinical evidence.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. LESION–GRAD-CAM OVERLAP ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def analyze_attention_lesion_agreement(
    preproc_img: np.ndarray,
    heatmap: np.ndarray,
    raw_unet_mask: Optional[np.ndarray],
    stage: int,
    cam_threshold: float = 0.30,
    lesion_threshold: float = 0.35,
) -> Dict[str, Any]:
    """Computes spatial overlap (IoU, Dice, % lesions in attention) between Grad-CAM & U-Net.
    
    Research/explainability metric only. Does not alter classifier outputs.
    """
    base = (np.clip(preproc_img, 0.0, 1.0) * 255).astype(np.uint8)
    h, w = AppConfig.IMG_SIZE, AppConfig.IMG_SIZE

    # Resize Grad-CAM to image dimensions
    if heatmap is not None and heatmap.size > 0:
        cam_resized = cv2.resize(heatmap, (w, h))
        cam_binary = cam_resized > cam_threshold
    else:
        cam_resized = np.zeros((h, w), dtype=np.float32)
        cam_binary = np.zeros((h, w), dtype=bool)

    # U-Net lesion mask
    if raw_unet_mask is not None and stage > 0:
        if raw_unet_mask.shape != (h, w):
            raw_unet_mask = cv2.resize(raw_unet_mask, (w, h))
        lesion_binary = raw_unet_mask > lesion_threshold
    else:
        lesion_binary = np.zeros((h, w), dtype=bool)

    # Overlap metrics
    intersection = int(np.logical_and(cam_binary, lesion_binary).sum())
    union = int(np.logical_or(cam_binary, lesion_binary).sum())
    cam_area = int(cam_binary.sum())
    lesion_area = int(lesion_binary.sum())

    iou = float(intersection / max(union, 1)) if union > 0 else 0.0
    dice = float((2.0 * intersection) / max(cam_area + lesion_area, 1)) if (cam_area + lesion_area) > 0 else 0.0
    lesion_in_cam_pct = float(intersection / max(lesion_area, 1) * 100.0) if lesion_area > 0 else 0.0

    # Qualitative interpretation
    if stage == 0 or lesion_area == 0:
        interpretation = (
            "No significant retinal lesions segmented by U-Net (Stage 0 / normal parenchyma). "
            "Model attention is distributed over normal structural landmarks."
        )
        agreement_level = "Baseline (Normal Retinal Parenchyma)"
    elif lesion_in_cam_pct >= 65.0 or dice >= 0.50:
        interpretation = (
            "Substantial spatial agreement: The classifier's high-attention regions "
            "show strong overlap with independently segmented retinal microvascular lesions."
        )
        agreement_level = "High Spatial Agreement"
    elif lesion_in_cam_pct >= 30.0 or dice >= 0.25:
        interpretation = (
            "Moderate spatial agreement: The classifier attends partly to identified lesions, "
            "while also incorporating surrounding parenchymal and vascular context."
        )
        agreement_level = "Moderate Spatial Agreement"
    else:
        interpretation = (
            "Low spatial agreement: Classifier attention diverges from segmented lesion candidates. "
            "May reflect subtle diffuse retinopathy features, retinal background texture, or non-lesion cues. "
            "Secondary clinical review recommended."
        )
        agreement_level = "Divergent / Low Agreement"

    # Composite visualization: Base + Attention (Amber/Orange) + Lesions (Cyan) + Overlap (White/Bright Yellow)
    overlay = base.copy()
    # High attention area in translucent amber
    overlay[cam_binary] = [255, 170, 0]
    # Lesion candidates in translucent cyan
    overlay[lesion_binary] = [0, 230, 255]
    # Exact overlap in bright lime-yellow
    overlap_mask = np.logical_and(cam_binary, lesion_binary)
    overlay[overlap_mask] = [230, 255, 50]
    combined_vis = cv2.addWeighted(base, 0.40, overlay, 0.60, 0)

    # Lesion mask visualization
    lesion_vis = base.copy()
    lesion_vis[lesion_binary] = [0, 255, 100]
    lesion_vis = cv2.addWeighted(base, 0.50, lesion_vis, 0.50, 0)

    return {
        "iou": iou,
        "dice": dice,
        "lesion_in_cam_pct": lesion_in_cam_pct,
        "intersection_px": intersection,
        "union_px": union,
        "cam_area_px": cam_area,
        "lesion_area_px": lesion_area,
        "interpretation": interpretation,
        "agreement_level": agreement_level,
        "combined_vis": combined_vis,
        "lesion_vis": lesion_vis,
        "cam_binary": cam_binary,
        "lesion_binary": lesion_binary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. RETINAL SEVERITY MAP (4-QUADRANT ANATOMICAL ABNORMALITY MAP)
# ─────────────────────────────────────────────────────────────────────────────
def compute_retinal_severity_map(
    preproc_img: np.ndarray,
    lesion_binary: np.ndarray,
    vessel_binary: np.ndarray,
    heatmap: np.ndarray,
) -> Dict[str, Any]:
    """Calculates normalized visual-abnormality indicators across the 4 anatomical quadrants.
    
    Quadrants: Superior-Temporal (ST), Superior-Nasal (SN), Inferior-Temporal (IT), Inferior-Nasal (IN).
    Visual support only — not clinical severity grades.
    """
    base = (np.clip(preproc_img, 0.0, 1.0) * 255).astype(np.uint8)
    h, w = AppConfig.IMG_SIZE, AppConfig.IMG_SIZE
    mid_y, mid_x = h // 2, w // 2

    # Retinal parenchyma mask (ignore black border)
    gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
    parenchyma = gray > 15

    # Connected components for lesion counting
    lesion_uint8 = (lesion_binary.astype(np.uint8)) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(lesion_uint8, connectivity=8)

    quadrant_slices = {
        "Superior-Temporal": (slice(0, mid_y), slice(0, mid_x)),
        "Superior-Nasal": (slice(0, mid_y), slice(mid_x, w)),
        "Inferior-Temporal": (slice(mid_y, h), slice(0, mid_x)),
        "Inferior-Nasal": (slice(mid_y, h), slice(mid_x, w)),
    }

    quadrant_results = {}
    affected_count = 0

    for qname, (yslice, xslice) in quadrant_slices.items():
        q_parenchyma = parenchyma[yslice, xslice]
        q_visible = max(int(np.sum(q_parenchyma)), 1)

        q_lesions = lesion_binary[yslice, xslice]
        q_vessels = vessel_binary[yslice, xslice] if vessel_binary is not None else np.zeros((mid_y, mid_x), dtype=bool)

        # Lesion burden in quadrant
        lesion_burden_pct = float(np.sum(q_lesions) / q_visible * 100.0)

        # Lesion count in quadrant (count centroids landing in quadrant)
        q_lesion_count = 0
        for i in range(1, num_labels):
            cx, cy = centroids[i]
            if xslice.start <= cx < xslice.stop and yslice.start <= cy < yslice.stop:
                q_lesion_count += 1

        # Vessel density in quadrant
        vessel_density_pct = float(np.sum(q_vessels) / q_visible * 100.0)

        has_lesions = q_lesion_count > 0 or lesion_burden_pct > 0.05
        if has_lesions:
            affected_count += 1

        quadrant_results[qname] = {
            "lesion_burden_pct": lesion_burden_pct,
            "lesion_count": q_lesion_count,
            "vessel_density_pct": vessel_density_pct,
            "has_lesions": has_lesions,
        }

    max_burden = max((item["lesion_burden_pct"] for item in quadrant_results.values()), default=0.0)
    max_count = max((item["lesion_count"] for item in quadrant_results.values()), default=0)
    for item in quadrant_results.values():
        burden_component = item["lesion_burden_pct"] / max_burden if max_burden > 0 else 0.0
        count_component = item["lesion_count"] / max_count if max_count > 0 else 0.0
        item["abnormality_indicator"] = float(0.65 * burden_component + 0.35 * count_component)

    ranked = sorted(quadrant_results, key=lambda name: quadrant_results[name]["abnormality_indicator"])
    for rank, qname in enumerate(ranked):
        item = quadrant_results[qname]
        if item["abnormality_indicator"] == 0.0 or len(ranked) == 1 or rank == 0:
            label, color, background = "Low visual abnormality", "#10b981", "#ecfdf5"
        elif rank == len(ranked) - 1:
            label, color, background = "Higher visual abnormality", "#e11d48", "#fff1f2"
        else:
            label, color, background = "Moderate visual abnormality", "#d97706", "#fffbeb"
        item.update({"abnormality_label": label, "badge_color": color, "badge_bg": background})

    # Overlay showing the 4 anatomical regions on the retinal image
    annotated_img = base.copy()
    # Draw quadrant crosshair lines
    cv2.line(annotated_img, (mid_x, 0), (mid_x, h), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(annotated_img, (0, mid_y), (w, mid_y), (255, 255, 255), 1, cv2.LINE_AA)

    # Label text positions
    labels_pos = {
        "ST": (10, 22),
        "SN": (mid_x + 10, 22),
        "IT": (10, mid_y + 22),
        "IN": (mid_x + 10, mid_y + 22),
    }
    for qkey, (px, py) in labels_pos.items():
        cv2.putText(annotated_img, qkey, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated_img, qkey, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    # Highlight lesion locations in green
    if lesion_binary is not None and np.any(lesion_binary):
        annotated_img[lesion_binary] = [0, 255, 120]

    quadrant_overlay = cv2.addWeighted(base, 0.60, annotated_img, 0.40, 0)

    return {
        "quadrants": quadrant_results,
        "affected_quadrants_count": affected_count,
        "total_lesion_count": max(0, num_labels - 1),
        "quadrant_overlay": quadrant_overlay,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. QUANTITATIVE RETINAL BIOMARKER PANEL
# ─────────────────────────────────────────────────────────────────────────────
def compile_retinal_biomarkers(
    lesion_pct: float,
    lesion_count: int,
    vessel_density: float,
    affected_quadrants: int,
    optic_disc: Dict[str, Any],
    qc: Dict[str, Any],
    classical_cv: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Compiles consolidated list of all actually computed computer vision measurements.
    
    Only displays measurements that are actually generated — does not fabricate missing values.
    """
    biomarkers = []

    # 1. Lesion burden
    biomarkers.append({
        "name": "Lesion Area Burden",
        "value": f"{lesion_pct:.2f}%",
        "tooltip": "Percentage of visible retinal parenchyma occupied by segmented lesion candidates (microaneurysms/exudates).",
        "category": "Pathology Burden",
        "status": "No candidate area" if lesion_pct == 0.0 else "Measured visual-support value",
    })

    # 2. Lesion count
    biomarkers.append({
        "name": "Candidate Lesion Count",
        "value": str(lesion_count),
        "tooltip": "Estimated number of discrete contiguous morphological lesion clusters identified via 8-connectivity labeling.",
        "category": "Morphology",
        "status": "No candidates" if lesion_count == 0 else "Candidate clusters measured",
    })

    # 3. Vessel density
    biomarkers.append({
        "name": "Retinal Vessel Density",
        "value": f"{vessel_density:.2f}%",
        "tooltip": "Vascular bed area percentage extracted via green-channel CLAHE + black-hat morphological filtering + Otsu thresholding.",
        "category": "Vasculature",
        "status": "Measured",
    })

    # 4. Affected quadrants
    biomarkers.append({
        "name": "Affected Anatomical Quadrants",
        "value": f"{affected_quadrants} / 4",
        "tooltip": "Number of four retinal quadrants (ST, SN, IT, IN) displaying detectable lesion candidates.",
        "category": "Spatial Distribution",
        "status": "Measured spatial distribution",
    })

    # 5. Optic disc area / radius
    if optic_disc.get("optic_disc_found"):
        radius = optic_disc.get("optic_disc_radius", 0)
        area_px = int(np.pi * (radius ** 2)) if radius else 0
        biomarkers.append({
            "name": "Optic-Disc Area",
            "value": f"{area_px:,} px² (r={radius}px)",
            "tooltip": "Morphological boundary size of localized optic-disc candidate in 224x224 coordinate frame.",
            "category": "Anatomical Landmark",
            "status": "Localized",
        })
    else:
        biomarkers.append({
            "name": "Optic-Disc Area",
            "value": "Not reliably detected",
            "tooltip": "Optic disc heuristic did not identify a candidate satisfying nasal aspect-ratio criteria.",
            "category": "Anatomical Landmark",
            "status": "Unresolved",
        })

    # 6. Image Quality & Blur
    blur_score = qc.get("blur_score", 0.0)
    brightness = qc.get("brightness", 0.0)
    green_energy = qc.get("central_green_energy", 0.0)
    biomarkers.append({
        "name": "Focus Score (Laplacian Var.)",
        "value": f"{blur_score:.1f}",
        "tooltip": "High-frequency variance of Laplacian operator. Values >= 80 indicate acceptable photographic focus.",
        "category": "Image Quality",
        "status": "Pass" if blur_score >= 80.0 else "Blur Flagged",
    })

    biomarkers.append({
        "name": "Luminance & Green Energy",
        "value": f"{brightness:.0f}/255 (G: {green_energy:.1f})",
        "tooltip": "Mean retinal brightness and central green-channel luminance ensuring adequate optical exposure.",
        "category": "Optical Illumination",
        "status": "Adequate" if 30 <= brightness <= 220 else "Sub-optimal",
    })

    # 7. Classical CV Biomarkers
    if classical_cv:
        sobel_edge = classical_cv.get("sobel_edge_density", 0.0)
        morph_pct = classical_cv.get("morph_candidate_pct", 0.0)
        biomarkers.append({
            "name": "Sobel Edge Gradient Density",
            "value": f"{sobel_edge:.1f}%",
            "tooltip": "Density of high-gradient vascular and textural transitions calculated via 3x3 Sobel convolution.",
            "category": "Classical CV",
            "status": "Calibrated",
        })
        biomarkers.append({
            "name": "Top-Hat Bright-Lesion Index",
            "value": f"{morph_pct:.1f}%",
            "tooltip": "Percentage of pixels showing positive morphological top-hat response for small bright retinal deposits.",
            "category": "Classical CV",
            "status": "Calibrated",
        })

    return biomarkers


# ─────────────────────────────────────────────────────────────────────────────
# 4. PREDICTION–EVIDENCE CONSISTENCY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_prediction_evidence_consistency(
    diagnosis: Dict[str, Any],
    lesion_pct: float,
    affected_quadrants: int,
    overlap_data: Dict[str, Any],
    qc: Dict[str, Any],
) -> Dict[str, Any]:
    """Compares classifier stage prediction with independent visual evidence.
    
    Transparent decision support signal only — NEVER overrides the classifier prediction.
    """
    stage = diagnosis.get("stage", 0)
    stage_name = diagnosis.get("stage_name", "Unknown")
    confidence = diagnosis.get("confidence", 0.0)
    qc_passed = qc.get("passed", True)

    # If image quality failed, evidence is fundamentally insufficient
    if not qc_passed:
        return {
            "status": "INSUFFICIENT EVIDENCE",
            "status_color": "#64748b",
            "status_bg": "#f1f5f9",
            "icon": "⚪",
            "reason": (
                f"Image-quality pre-check flagged optical deficiencies ({'; '.join(qc.get('issues', []))}). "
                "Independent visual evidence cannot be reliably evaluated; manual ophthalmic examination required."
            ),
            "evidence_summary": {
                "stage_name": stage_name,
                "confidence": confidence,
                "lesion_pct": lesion_pct,
                "affected_quadrants": affected_quadrants,
                "agreement_level": overlap_data.get("agreement_level", "Unknown"),
            },
        }

    reasons = []
    inconsistencies = 0

    # Rule checks per stage
    if stage == 0:
        if lesion_pct > 2.0:
            reasons.append(f"Model classified No DR ({confidence*100:.1f}%), but U-Net detected {lesion_pct:.2f}% lesion burden candidates.")
            inconsistencies += 2
        elif lesion_pct > 0.5:
            reasons.append(f"Minor lesion candidates ({lesion_pct:.2f}%) present despite Stage 0 prediction.")
            inconsistencies += 1
        else:
            reasons.append(f"No DR classification is fully congruent with clean parenchyma (lesion burden {lesion_pct:.2f}%).")

    elif stage == 1:  # Mild NPDR
        if lesion_pct > 4.0:
            reasons.append(f"Elevated lesion burden ({lesion_pct:.2f}%) exceeds typical Mild NPDR isolated microaneurysm expectations.")
            inconsistencies += 1
        elif affected_quadrants > 2:
            reasons.append(f"Lesions dispersed across {affected_quadrants} quadrants exceeds typical localized Mild NPDR.")
            inconsistencies += 1
        else:
            reasons.append(f"Mild NPDR prediction aligns with localized lesion burden ({lesion_pct:.2f}%, {affected_quadrants} quadrant).")

    elif stage in (2, 3, 4):  # Moderate, Severe, Proliferative
        if lesion_pct < 0.20:
            reasons.append(
                f"High-stage prediction ({stage_name}, {confidence*100:.1f}%) has minimal independently segmented lesion burden ({lesion_pct:.2f}%). "
                "Classifier may rely on subtle diffuse textures or vessel attenuation."
            )
            inconsistencies += 2
        elif affected_quadrants < 2 and stage >= 3:
            reasons.append(f"Stage {stage} typically presents multifocal pathology, but lesions localized to {affected_quadrants} quadrant(s).")
            inconsistencies += 1
        else:
            reasons.append(f"Lesion distribution ({lesion_pct:.2f}% across {affected_quadrants} quadrants) supports {stage_name} severity.")

        # Overlap check
        dice = overlap_data.get("dice", 0.0)
        if dice >= 0.35 or overlap_data.get("lesion_in_cam_pct", 0.0) >= 50.0:
            reasons.append("Grad-CAM attention map exhibits substantial spatial concordance with segmented lesions.")
        else:
            reasons.append("Grad-CAM attention partially diverges from U-Net lesion clusters.")

    # Synthesize consistency status
    if inconsistencies == 0:
        status = "CONSISTENT"
        status_color = "#10b981"
        status_bg = "#ecfdf5"
        icon = "🟢"
        explanation = "The independently extracted visual evidence (lesion burden, spatial dispersion, Grad-CAM focus) is consistent with the model's prediction."
    elif inconsistencies == 1:
        status = "PARTIALLY CONSISTENT"
        status_color = "#d97706"
        status_bg = "#fffbeb"
        icon = "🟡"
        explanation = "Partial consistency: Core diagnostic indicators align, but secondary metrics show slight divergence. Routine clinical verification advised."
    else:
        status = "POTENTIALLY INCONSISTENT"
        status_color = "#e11d48"
        status_bg = "#fff1f2"
        icon = "🟠"
        explanation = "Potential divergence between classifier confidence and independently segmented lesion evidence. Manual slit-lamp ophthalmic review strongly recommended."

    full_reason = f"{explanation} Details: {' '.join(reasons)}"

    return {
        "status": status,
        "status_color": status_color,
        "status_bg": status_bg,
        "icon": icon,
        "reason": full_reason,
        "inconsistencies_count": inconsistencies,
        "evidence_summary": {
            "stage_name": stage_name,
            "confidence": confidence,
            "lesion_pct": lesion_pct,
            "affected_quadrants": affected_quadrants,
            "agreement_level": overlap_data.get("agreement_level", "Unknown"),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. LONGITUDINAL RETINAL IMAGE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
def compare_longitudinal_examinations(
    prev_preproc: np.ndarray,
    curr_preproc: np.ndarray,
    prev_diag: Dict[str, Any],
    curr_diag: Dict[str, Any],
    prev_expl: Dict[str, Any],
    curr_expl: Dict[str, Any],
) -> Dict[str, Any]:
    """Compares two sequential examinations (Previous vs Current) and calculates metric deltas.
    
    Attempts feature-based registration. If unreliable, safely degrades to non-spatial comparison.
    Research/visual support only — does not claim clinical progression.
    """
    prev_base = (np.clip(prev_preproc, 0.0, 1.0) * 255).astype(np.uint8)
    curr_base = (np.clip(curr_preproc, 0.0, 1.0) * 255).astype(np.uint8)
    h, w = AppConfig.IMG_SIZE, AppConfig.IMG_SIZE

    # Stages & confidence
    prev_stage = prev_diag.get("stage", 0)
    curr_stage = curr_diag.get("stage", 0)
    stage_delta = curr_stage - prev_stage

    # Lesion burden deltas
    prev_lesion = prev_expl.get("lesion_pct", 0.0)
    curr_lesion = curr_expl.get("lesion_pct", 0.0)
    lesion_delta = curr_lesion - prev_lesion

    # Vessel density deltas
    prev_vessel = prev_expl.get("vessel_density", 0.0)
    curr_vessel = curr_expl.get("vessel_density", 0.0)
    vessel_delta = curr_vessel - prev_vessel

    # Affected quadrants deltas
    prev_quads = prev_expl.get("affected_quadrants_count", 0)
    curr_quads = curr_expl.get("affected_quadrants_count", 0)
    quads_delta = curr_quads - prev_quads

    # Attempt spatial registration via ORB
    registration_reliable = False
    diff_vis = None
    try:
        prev_gray = cv2.cvtColor(prev_base, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_base, cv2.COLOR_RGB2GRAY)

        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(curr_gray, None)

        if des1 is not None and des2 is not None and len(kp1) >= 8 and len(kp2) >= 8:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) >= 8:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:30]]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:30]]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if M is not None and np.sum(mask) >= 6:
                    aligned_prev = cv2.warpPerspective(prev_base, M, (w, h))
                    abs_diff = cv2.absdiff(curr_base, aligned_prev)
                    diff_gray = cv2.cvtColor(abs_diff, cv2.COLOR_RGB2GRAY)
                    diff_heat = cv2.applyColorMap(diff_gray, cv2.COLORMAP_MAGMA)
                    diff_vis = cv2.addWeighted(curr_base, 0.50, diff_heat, 0.50, 0)
                    registration_reliable = True
    except Exception:
        registration_reliable = False

    if not registration_reliable:
        # Fallback side-by-side or simple difference without misleading warping
        abs_simple = cv2.absdiff(curr_base, prev_base)
        diff_vis = cv2.applyColorMap(cv2.cvtColor(abs_simple, cv2.COLOR_RGB2GRAY), cv2.COLORMAP_JET)

    # Narrative summary
    direction = "stable" if stage_delta == 0 else ("escalated" if stage_delta > 0 else "reduced")
    summary = (
        f"Visual comparison detected a stage delta of {stage_delta:+d} (from Stage {prev_stage} to Stage {curr_stage}), "
        f"lesion burden shift of {lesion_delta:+.2f} pp, and vessel density shift of {vessel_delta:+.2f} pp. "
        "Visual support only — does not represent a confirmed clinical disease progression course."
    )

    return {
        "prev_stage": prev_stage,
        "curr_stage": curr_stage,
        "stage_delta": stage_delta,
        "prev_lesion": prev_lesion,
        "curr_lesion": curr_lesion,
        "lesion_delta": lesion_delta,
        "prev_vessel": prev_vessel,
        "curr_vessel": curr_vessel,
        "vessel_delta": vessel_delta,
        "prev_quads": prev_quads,
        "curr_quads": curr_quads,
        "quads_delta": quads_delta,
        "registration_reliable": registration_reliable,
        "diff_vis": diff_vis,
        "prev_img": prev_base,
        "curr_img": curr_base,
        "summary": summary,
        "direction": direction,
    }
