"""Clinical decision pipeline agents."""

from typing import Any, Dict, Optional

import cv2
import numpy as np

from core.config import AppConfig
from core.explainability import (
    compute_gradcam,
    extract_classical_cv_biomarkers,
    extract_retinal_vessels,
    find_similar_cases,
    generate_quadrant_description,
    localize_optic_disc,
    overlay_heatmap,
    segment_retinal_lesions_with_mask,
)
from core.advanced_cv import (
    analyze_attention_lesion_agreement,
    compute_retinal_severity_map,
    evaluate_prediction_evidence_consistency,
    explain_similar_cases,
)
from core.models import full_model, unet_model


class DiagnosisAgent:
    def process(self, preproc_img: np.ndarray) -> Dict[str, Any]:
        probabilities = full_model(preproc_img[np.newaxis, ...], training=False).numpy()[0]
        stage = int(np.argmax(probabilities))
        return {"stage": stage, "stage_name": AppConfig.CLASS_NAMES[stage],
                "confidence": float(probabilities[stage]),
                "probabilities": {AppConfig.CLASS_NAMES[i]: float(probabilities[i]) for i in range(5)}}


class ExplainabilityAgent:
    def process(self, preproc_img: np.ndarray, diagnosis: Dict[str, Any]) -> Dict[str, Any]:
        stage = diagnosis["stage"]
        try:
            heatmap = compute_gradcam(preproc_img, stage)
        except Exception:
            heatmap = np.zeros((7, 7), dtype=np.float32)

        quadrant_desc, quadrant_scores, peak_quadrant, peak_value = generate_quadrant_description(heatmap, stage)

        try:
            lesion_segmentation, lesion_pct, raw_mask = segment_retinal_lesions_with_mask(preproc_img, heatmap, stage)
        except Exception:
            base = (np.clip(preproc_img, 0.0, 1.0) * 255).astype(np.uint8)
            lesion_segmentation, lesion_pct, raw_mask = base, 0.0, None

        try:
            vessel_analysis = extract_retinal_vessels(preproc_img)
        except Exception:
            vessel_analysis = {
                "vessel_mask": np.zeros((AppConfig.IMG_SIZE, AppConfig.IMG_SIZE), dtype=np.uint8),
                "vessel_overlay": (np.clip(preproc_img, 0.0, 1.0) * 255).astype(np.uint8),
                "vessel_density": 0.0,
                "vessel_status": "Unavailable",
                "method_description": "Vessel analysis unavailable; classifier output unaffected.",
            }
        try:
            optic_disc = localize_optic_disc(preproc_img)
        except Exception:
            optic_disc = {
                "optic_disc_found": False,
                "optic_disc_overlay": (np.clip(preproc_img, 0.0, 1.0) * 255).astype(np.uint8),
                "optic_disc_center": None,
                "optic_disc_score": 0.0,
                "method_description": "Optic-disc analysis unavailable; classifier output unaffected.",
            }

        h, w = AppConfig.IMG_SIZE, AppConfig.IMG_SIZE
        cam_resized = cv2.resize(heatmap, (w, h)) if (heatmap is not None and heatmap.size > 0) else np.zeros((h, w), dtype=np.float32)
        lesion_binary = (raw_mask > 0.35) & (cam_resized > 0.30) if (raw_mask is not None and stage > 0) else np.zeros((h, w), dtype=bool)

        # 1. Feature 1: Lesion-Grad-CAM Overlap Analysis (Agreement)
        try:
            overlap_analysis = analyze_attention_lesion_agreement(preproc_img, heatmap, raw_mask, stage)
        except Exception:
            overlap_analysis = {"iou": 0.0, "dice": 0.0, "lesion_in_cam_pct": 0.0,
                                "interpretation": "Overlap analysis unavailable.", "agreement_level": "Insufficient evidence",
                                "combined_vis": (np.clip(preproc_img, 0.0, 1.0) * 255).astype(np.uint8)}

        # 2. Feature 2: Retinal Severity Map (4 Anatomical Quadrants)
        try:
            severity_map = compute_retinal_severity_map(preproc_img, lesion_binary, vessel_analysis.get("vessel_mask"), heatmap)
        except Exception:
            severity_map = {"quadrants": {}, "affected_quadrants_count": 0, "total_lesion_count": 0,
                            "quadrant_overlay": (np.clip(preproc_img, 0.0, 1.0) * 255).astype(np.uint8)}

        similar_cases = find_similar_cases(preproc_img)
        cbr_explanation = explain_similar_cases(similar_cases, stage)

        return {
            "overlay_cam": overlay_heatmap(preproc_img, heatmap),
            "lesion_seg": lesion_segmentation,
            "lesion_pct": lesion_pct,
            "raw_mask": raw_mask,
            "lesion_binary": lesion_binary,
            "quadrant_desc": quadrant_desc,
            "quadrant_scores": quadrant_scores,
            "peak_quadrant": peak_quadrant,
            "peak_val": peak_value,
            "similar_cases": similar_cases,
            "cbr_explanation": cbr_explanation,
            "classical_cv": extract_classical_cv_biomarkers(preproc_img),
            "vessel_method_description": vessel_analysis.get("method_description", "Exploratory classical-CV vessel analysis."),
            "optic_disc_method_description": optic_disc.get("method_description", "Exploratory optic-disc heuristic."),
            "overlap_analysis": overlap_analysis,
            "severity_map": severity_map,
            "affected_quadrants_count": severity_map.get("affected_quadrants_count", 0),
            **vessel_analysis,
            **optic_disc,
        }


class AdvisoryAgent:
    GUIDANCE = {
        0: ("Routine Screening", "No diabetic microvascular abnormalities observed. Recommend annual dilated retinal examination and continued glycemic management (HbA1c < 7.0%).", "12 Months"),
        1: ("Non-Urgent Clinical Monitoring", "Mild NPDR (isolated microaneurysms). Primary care management: optimize blood pressure, cholesterol, and glycemic control.", "6–9 Months"),
        2: ("Comprehensive Specialist Referral", "Moderate NPDR (dot/blot hemorrhages, hard exudates). Significant risk of macular edema; schedule dilated examination and optical coherence tomography (OCT).", "3–6 Months"),
        3: ("Urgent Specialist Evaluation", "Severe NPDR (fulfills '4-2-1 rule'). High progression risk to proliferative retinopathy. Immediate ophthalmologist evaluation required.", "2–4 Weeks"),
        4: ("EMERGENCY Vitreoretinal Intervention", "Proliferative DR (active neovascularization, vitreous hemorrhage). Immediate retina specialist referral for panretinal photocoagulation (PRP) or intravitreal anti-VEGF therapy.", "24–48 Hours"),
    }
    DISCLAIMER = "CLINICAL DISCLAIMER: RetinaGuard AI is an investigational decision-support tool. It does not replace independent clinical judgment or formal diagnostic verification by a licensed ophthalmologist."

    def process(self, stage: int) -> Dict[str, str]:
        urgency, plan, followup = self.GUIDANCE.get(stage, ("Unknown", "Manual ophthalmological review mandatory.", "Immediate"))
        return {"urgency": urgency, "plan": plan, "followup": followup, "disclaimer": self.DISCLAIMER}


class GovernanceAgent:
    def __init__(self, threshold: float = AppConfig.DEFAULT_CONFIDENCE_THRESHOLD):
        self.threshold = threshold

    def evaluate(
        self,
        diagnosis: Dict[str, Any],
        explanation: Dict[str, Any],
        advisory: Dict[str, str],
        qc: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        confidence = diagnosis["confidence"]
        flagged = confidence < self.threshold

        # Feature 4: Prediction-Evidence Consistency Analysis
        consistency_analysis = evaluate_prediction_evidence_consistency(
            diagnosis=diagnosis,
            lesion_pct=explanation.get("lesion_pct", 0.0),
            affected_quadrants=explanation.get("affected_quadrants_count", 0),
            overlap_data=explanation.get("overlap_analysis", {}),
            qc=qc or {"passed": True},
        )

        # Flag for review if confidence is low OR if evidence is potentially inconsistent
        recommend_manual_review = flagged or (consistency_analysis.get("status") in ("POTENTIALLY INCONSISTENT", "INSUFFICIENT EVIDENCE"))

        if flagged:
            message = (f"SAFETY INTERCEPTION ACTIVATED: Model confidence ({confidence*100:.1f}%) is BELOW the clinical safety threshold "
                       f"({self.threshold*100:.0f}%). Automated treatment recommendations have been WITHHELD to eliminate hallucination risks. "
                       "The patient case has been flagged for mandatory specialist review.")
            controlled_advisory = {"urgency": "HUMAN SPECIALIST TRIAGE MANDATORY", "plan": message,
                                   "followup": "Withheld — Manual Slit-Lamp Examination Required Immediately",
                                   "disclaimer": advisory["disclaimer"]}
        else:
            if consistency_analysis.get("status") == "POTENTIALLY INCONSISTENT":
                message = (f"Safety Verified with Review Recommendation: Model confidence ({confidence*100:.1f}%) satisfies threshold ({self.threshold*100:.0f}%), "
                           "but independent visual evidence divergence was noted. Clinician review recommended.")
            else:
                message = f"Safety Verified: Model confidence ({confidence*100:.1f}%) satisfies the clinical safety threshold ({self.threshold*100:.0f}%)."
            controlled_advisory = advisory

        return {
            "flagged": flagged,
            "flagged_for_review": recommend_manual_review,
            "confidence": confidence,
            "message": message,
            "diagnosis": diagnosis,
            "explanation": explanation,
            "advisory": controlled_advisory,
            "consistency_analysis": consistency_analysis,
        }


def run_pipeline(
    preproc_img: np.ndarray,
    threshold: float = AppConfig.DEFAULT_CONFIDENCE_THRESHOLD,
    qc: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    diagnosis = DiagnosisAgent().process(preproc_img)
    explanation = ExplainabilityAgent().process(preproc_img, diagnosis)
    advisory = AdvisoryAgent().process(diagnosis["stage"])
    return GovernanceAgent(threshold).evaluate(diagnosis, explanation, advisory, qc=qc)
