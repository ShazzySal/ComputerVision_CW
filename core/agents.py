"""Clinical decision pipeline agents."""

from typing import Any, Dict

import numpy as np

from core.config import AppConfig
from core.explainability import (
    compute_gradcam,
    extract_classical_cv_biomarkers,
    find_similar_cases,
    generate_quadrant_description,
    overlay_heatmap,
    segment_retinal_lesions,
)
from core.models import full_model


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
        heatmap = compute_gradcam(preproc_img, stage)
        quadrant_desc, quadrant_scores, peak_quadrant, peak_value = generate_quadrant_description(heatmap, stage)
        lesion_segmentation, lesion_pct = segment_retinal_lesions(preproc_img, heatmap, stage)
        return {"overlay_cam": overlay_heatmap(preproc_img, heatmap), "lesion_seg": lesion_segmentation,
                "lesion_pct": lesion_pct, "quadrant_desc": quadrant_desc,
                "quadrant_scores": quadrant_scores, "peak_quadrant": peak_quadrant,
                "peak_val": peak_value, "similar_cases": find_similar_cases(preproc_img),
                "classical_cv": extract_classical_cv_biomarkers(preproc_img)}


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

    def evaluate(self, diagnosis: Dict[str, Any], explanation: Dict[str, Any], advisory: Dict[str, str]) -> Dict[str, Any]:
        confidence = diagnosis["confidence"]
        flagged = confidence < self.threshold
        if flagged:
            message = (f"SAFETY INTERCEPTION ACTIVATED: Model confidence ({confidence*100:.1f}%) is BELOW the clinical safety threshold "
                       f"({self.threshold*100:.0f}%). Automated treatment recommendations have been WITHHELD to eliminate hallucination risks. "
                       "The patient case has been flagged for mandatory specialist review.")
            controlled_advisory = {"urgency": "HUMAN SPECIALIST TRIAGE MANDATORY", "plan": message,
                                   "followup": "Withheld — Manual Slit-Lamp Examination Required Immediately",
                                   "disclaimer": advisory["disclaimer"]}
        else:
            message = f"Safety Verified: Model confidence ({confidence*100:.1f}%) satisfies the clinical safety threshold ({self.threshold*100:.0f}%)."
            controlled_advisory = advisory
        return {"flagged": flagged, "flagged_for_review": flagged, "confidence": confidence,
                "message": message, "diagnosis": diagnosis, "explanation": explanation,
                "advisory": controlled_advisory}


def run_pipeline(preproc_img: np.ndarray, threshold: float = AppConfig.DEFAULT_CONFIDENCE_THRESHOLD) -> Dict[str, Any]:
    diagnosis = DiagnosisAgent().process(preproc_img)
    explanation = ExplainabilityAgent().process(preproc_img, diagnosis)
    advisory = AdvisoryAgent().process(diagnosis["stage"])
    return GovernanceAgent(threshold).evaluate(diagnosis, explanation, advisory)
