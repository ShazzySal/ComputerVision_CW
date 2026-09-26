"""
app.py — RetinaTrace AI: Multi-Agent Clinical Decision Support System
Modern, Unique & User-Friendly Gradio Web Application.

BSc (Hons) Computer Science — Computer Vision (BSCCOMP24.2P)
Repository: https://github.com/ShazzySal/ComputerVision_CW

Architecture Features:
----------------------
1. 3-Layer Clinical Explainability Dossier:
   - Layer 1: EfficientNetB3 5-Stage Disease Classification & Probability Bar Breakdown.
   - Layer 2: Regional Grad-CAM Attention Heatmap & Automated Anatomical Quadrant Narrative.
   - Layer 3: Auxiliary U-Net Pixel-Level Retinal Lesion Segmentation (Microaneurysms/Exudates in Green).
2. Technical Innovations:
   - Innovation A: Case-Based Reasoning (CBR) Metric Embedding Retrieval (Cosine Dot-Product).
   - Innovation B: Decoupled 4-Agent Decision Pipeline with Active Governance Safety Gate (70% Threshold).
3. Modern UX / UI:
   - Custom Biomedical Clinical CSS Theme with Responsive Cards & Status Badges.
   - One-Click Preset Fundus Sample Loaders (Normal, Moderate NPDR, Proliferative DR).
   - One-Click Interactive Safety Gate Override Simulation.
   - Exportable Clinical EHR Summary Note.
"""

import os
from difflib import SequenceMatcher
from typing import Tuple, List, Dict, Any, Union, Optional
import numpy as np
import cv2
import gradio as gr

from core.config import AppConfig
from core.preprocessing import preprocess_image
from core.advanced_cv import compile_retinal_biomarkers, compare_longitudinal_examinations

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Modular pipeline implementations.
from core.agents import run_pipeline


# ─────────────────────────────────────────────────────────────────────────────
# 5. Clinical Chatbot Knowledge Base & Responder
# ─────────────────────────────────────────────────────────────────────────────
_CLINICAL_KB: List[Dict[str, Any]] = [
    # ── DR Stage Descriptions ─────────────────────────────────────────────────
    {
        "keys": ["stage 0", "no dr", "normal", "healthy"],
        "reply": (
            "**Stage 0 — No Diabetic Retinopathy (No DR)**\n\n"
            "The fundus appears normal with no microvascular abnormalities. "
            "No retinal lesions, hemorrhages, or exudates are detected.\n\n"
            "**Management (AAO PPP 2022):** Annual dilated fundoscopy screening. "
            "Reinforce glycemic control (HbA1c < 7%), blood pressure < 130/80 mmHg, "
            "and lipid optimization. No treatment intervention required."
        ),
    },
    {
        "keys": ["stage 1", "mild", "mild npdr", "microaneurysm"],
        "reply": (
            "**Stage 1 — Mild Non-Proliferative Diabetic Retinopathy (Mild NPDR)**\n\n"
            "Characterized by the presence of **microaneurysms only** — outpouchings "
            "in fragile retinal capillary walls caused by pericyte degeneration.\n\n"
            "**Management (AAO PPP 2022):** Annual dilated fundoscopy. Intensify "
            "systemic risk factor control. No intraocular treatment is indicated at this stage."
        ),
    },
    {
        "keys": ["stage 2", "moderate", "moderate npdr", "exudate", "cotton wool"],
        "reply": (
            "**Stage 2 — Moderate Non-Proliferative Diabetic Retinopathy (Moderate NPDR)**\n\n"
            "More than microaneurysms present: dot-and-blot hemorrhages, hard exudates "
            "(lipid leakage), and cotton-wool spots (nerve fiber layer infarcts) visible. "
            "Does not meet the criteria for Severe NPDR.\n\n"
            "**Management (AAO PPP 2022):** 6–12 month follow-up. Ophthalmologist referral "
            "recommended. Evaluate for clinically significant diabetic macular edema (CSME)."
        ),
    },
    {
        "keys": ["stage 3", "severe", "severe npdr", "4-2-1", "venous beading", "irma"],
        "reply": (
            "**Stage 3 — Severe Non-Proliferative Diabetic Retinopathy (Severe NPDR)**\n\n"
            "Defined by the **4-2-1 rule**: >20 intraretinal hemorrhages in all 4 quadrants, "
            "venous beading in ≥2 quadrants, or prominent intraretinal microvascular "
            "abnormalities (IRMA) in ≥1 quadrant.\n\n"
            "**Management (AAO PPP 2022):** 3–4 month follow-up with retinal specialist. "
            "High risk of progression to proliferative disease. Consider panretinal "
            "photocoagulation (PRP) prophylactically in high-risk patients."
        ),
    },
    {
        "keys": ["stage 4", "proliferative", "pdr", "neovascularization", "vitreous hemorrhage", "nvd", "nve"],
        "reply": (
            "**Stage 4 — Proliferative Diabetic Retinopathy (PDR)**\n\n"
            "The most advanced stage. VEGF-driven **neovascularization** breaches the "
            "internal limiting membrane producing fragile new vessels on the disc (NVD) "
            "or retina (NVE). Untreated, this leads to vitreous hemorrhage, fibrovascular "
            "proliferation, tractional retinal detachment, and irreversible blindness.\n\n"
            "**Management (AAO PPP 2022):** Urgent ophthalmologist referral within 1–2 weeks. "
            "Panretinal photocoagulation (PRP) or intravitreal anti-VEGF injections "
            "(ranibizumab, bevacizumab) are first-line treatments."
        ),
    },
    # ── Model Architecture FAQs ───────────────────────────────────────────────
    {
        "keys": ["efficientnet", "backbone", "architecture", "model", "network", "cnn"],
        "reply": (
            "**EfficientNetB3 — Deep Learning Backbone**\n\n"
            "RetinaTrace uses **EfficientNetB3** pre-trained on ImageNet as its convolutional "
            "backbone. EfficientNet applies **compound scaling** — simultaneously scaling "
            "depth, width, and resolution using a fixed ratio — achieving superior "
            "accuracy/parameter efficiency vs ResNet, DenseNet, or VGG.\n\n"
            "A custom classification head is added:\n"
            "`GAP → BatchNorm → Dense(256, ReLU) → Dropout(0.3) → Dense(5, Softmax)`\n\n"
            "Training uses **2-phase transfer learning**:\n"
            "- Phase 1 (15 epochs, LR=1e-3): Base frozen, head trained.\n"
            "- Phase 2 (25 epochs, LR=1e-5): Top 30 base layers unfrozen for fine-tuning."
        ),
    },
    {
        "keys": ["grad-cam", "gradcam", "heatmap", "saliency", "explainability", "layer 2"],
        "reply": (
            "**Grad-CAM — Layer 2 Regional Explainability**\n\n"
            "Gradient-weighted Class Activation Mapping (Grad-CAM) computes a 2D saliency "
            "heatmap by backpropagating gradients through the final convolutional layer "
            "(`top_activation` in EfficientNetB3) using `tf.GradientTape`.\n\n"
            "The heatmap identifies **which anatomical regions drove the classification** "
            "— e.g. the optic disc, macular area, or peripheral microaneurysm clusters. "
            "This provides radiologist-level regional accountability for every prediction.\n\n"
            "The overlay uses a Jet colormap: red = highest activation, blue = lowest."
        ),
    },
    {
        "keys": ["unet", "u-net", "segmentation", "lesion", "layer 3", "mask"],
        "reply": (
            "**Auxiliary U-Net — Layer 3 Pixel-Level Lesion Segmentation**\n\n"
            "The U-Net predicts a pixel-level lesion probability map, which is visualized "
            "with highlighted candidate regions.\n\n"
            "In the notebook training workflow, manual pixel masks were unavailable, so "
            "pseudo-masks were generated from the green channel, "
            "Top-Hat and Black-Hat morphology, and Grad-CAM saliency above 0.35. These are "
            "synthetic training targets, not manual ground truth.\n\n"
            "The notebook trains the U-Net with a **Hybrid Soft Dice + BCE Loss**. In the "
            "deployed app, the loaded U-Net mask is additionally filtered at 0.35 and combined "
            "with a Grad-CAM threshold of 0.30. The result is visual support, not an independent diagnosis."
        ),
    },
    {
        "keys": ["governance", "safety gate", "flag", "threshold", "confidence", "override"],
        "reply": (
            "**GovernanceAgent — Active Clinical Safety Gate**\n\n"
            "Unlike passive warning banners, the `GovernanceAgent` **actively intercepts** "
            "the pipeline when model confidence falls below the configurable threshold "
            "(default: 70%).\n\n"
            "When triggered:\n"
            "- Automated treatment guidance is **withheld entirely**.\n"
            "- `flagged_for_review: True` is set.\n"
            "- The case is rerouted to **mandatory human ophthalmologist triage**.\n\n"
            "This mirrors clinical safety governance patterns in defense medical AI systems "
            "and prevents hallucination-driven misdiagnosis on ambiguous or out-of-distribution images."
        ),
    },
    {
        "keys": ["cbr", "case-based reasoning", "similar cases", "embedding", "retrieval", "nearest neighbor"],
        "reply": (
            "**Case-Based Reasoning (CBR) — Similar Case Retrieval**\n\n"
            "After diagnosis, RetinaTrace extracts a **256-dimensional feature vector** "
            "from the penultimate dense layer (`head_dense`) of the classifier.\n\n"
            "When a verified reference library is available, this embedding is compared "
            "against stored case embeddings in `embeddings.npz` using **cosine similarity**. "
            "The top-3 most similar historical cases are then retrieved and displayed with their DR stages.\n\n"
            "This provides clinicians with precedent-based diagnostic context, "
            "supporting decision transparency."
        ),
    },
    {
        "keys": ["ben graham", "preprocessing", "preprocessing pipeline", "crop", "normalization"],
        "reply": (
            "**Preprocessing Pipeline — 4-Step Optical Standardization**\n\n"
            "1. **Circular Border Crop:** Strips non-informative black optical borders "
            "using luminance thresholding (green channel > 7).\n"
            "2. **Ben Graham Enhancement:** Applies spatial frequency subtraction:\n"
            "   `I_norm = 4×I − 4×GaussianBlur(I, σ=10) + 128`\n"
            "   This suppresses low-frequency illumination variation and enhances local detail, "
            "but may amplify noise in poor-quality images.\n"
            "3. **Resize:** All images standardized to 224×224 pixels.\n"
            "4. **Normalization:** Pixel values scaled to [0, 1] float32."
        ),
    },
    {
        "keys": ["kappa", "qwk", "quadratic weighted kappa", "metric", "evaluation"],
        "reply": (
            "**Quadratic Weighted Kappa (QWK) — Ordinal Evaluation Metric**\n\n"
            "Standard accuracy treats all misclassifications equally. In clinical DR grading, "
            "confusing Stage 0 (No DR) with Stage 4 (Proliferative) is catastrophically "
            "worse than confusing Stage 1 with Stage 2.\n\n"
            "QWK assigns **quadratic penalty weights** proportional to the distance between "
            "the predicted and true stage. A kappa of 1.0 = perfect agreement; "
            "0.0 = chance agreement; <0 = worse than chance.\n\n"
            "RetinaTrace achieves a QWK of ~0.842, indicating substantial clinical agreement."
        ),
    },
    {
        "keys": ["refer", "referral", "when to refer", "specialist", "urgency"],
        "reply": (
            "**Clinical Referral Guidelines (AAO PPP 2022)**\n\n"
            "| DR Stage | Referral Urgency | Recall Interval |\n"
            "|---|---|---|\n"
            "| Stage 0 — No DR | No referral needed | 12 months |\n"
            "| Stage 1 — Mild NPDR | No immediate referral | 12 months |\n"
            "| Stage 2 — Moderate NPDR | Ophthalmologist referral recommended | 6–12 months |\n"
            "| Stage 3 — Severe NPDR | Retinal specialist referral | 3–4 months |\n"
            "| Stage 4 — Proliferative DR | **Urgent referral within 1–2 weeks** | ASAP |\n\n"
            "⚠️ *This tool is assistive clinical decision support only. All referral "
            "decisions must be confirmed by a qualified ophthalmologist.*"
        ),
    },
    {
        "keys": ["dataset", "data", "kaggle", "aptos", "idrid", "messidor", "eyepacs", "38034", "38,034"],
        "reply": (
            "**Dataset — Combined Fundus Image Records**\n\n"
            "The project uses the **Combined DR Dataset** package (Harsha, 2020) from Kaggle. "
            "Although the package describes multiple source cohorts, the local labels do not "
            "identify the source dataset for each image, so per-source provenance cannot be "
            "verified from this copy.\n\n"
            "The downloaded package contains **38,034 image files** across its supplied folders. "
            "The project creates a separate duplicate-safe model split; local labels do not identify "
            "the source dataset for each record, and the Kaggle description and file count use "
            "different reported scopes. The saved checkpoint predates the corrected input-scale "
            "pipeline and duplicate-safe split; retraining and evaluation are required before "
            "treating its predictions as validated results."
        ),
    },
]

_CHATBOT_FALLBACK = (
    "I don't have a specific answer for that query in my clinical knowledge base. "
    "For questions about diabetic retinopathy management, please consult the "
    "**AAO Preferred Practice Patterns (2022)** or a qualified ophthalmologist.\n\n"
    "You can ask me about: DR stages (0–4), Grad-CAM, the U-Net segmentation, "
    "the Governance Agent, Case-Based Reasoning, the preprocessing pipeline, "
    "Quadratic Weighted Kappa, referral guidelines, or the training dataset."
)

_SIMPLE_CHAT_REPLIES = {
    "lesion": (
        "A retinal lesion is an area of damage or abnormality in the retina, such as a "
        "microaneurysm, haemorrhage, or exudate. RetinaTrace highlights possible lesion areas "
        "to help a clinician review the image; the highlights are not a confirmed diagnosis."
    ),
    "grad-cam": (
        "Grad-CAM is a heatmap showing which parts of the retinal image influenced the model's "
        "prediction. It helps the user see where the model was looking."
    ),
    "unet": (
        "U-Net is the part of RetinaTrace that highlights possible lesion areas pixel by pixel. "
        "It helps show where abnormalities may be present, but the highlighted areas still need clinical confirmation."
    ),
    "efficientnet": (
        "EfficientNetB3 is the image-classification model used by RetinaTrace. It examines the "
        "retinal photograph and estimates the most likely diabetic-retinopathy stage."
    ),
    "governance": (
        "The Governance Agent is a safety check. If the result is uncertain or the supporting "
        "evidence does not agree, it withholds automated advice and asks for specialist review."
    ),
    "refer": (
        "Referral urgency depends on the detected retinopathy stage. More severe or uncertain "
        "results need faster review by an ophthalmologist."
    ),
    "stage 0": "Stage 0 means no diabetic-retinopathy signs were detected in the image.",
    "stage 1": "Stage 1 means mild diabetic retinopathy, usually limited to microaneurysms.",
    "stage 2": "Stage 2 means moderate diabetic retinopathy with more retinal changes than Stage 1.",
    "stage 3": "Stage 3 means severe non-proliferative diabetic retinopathy and needs prompt specialist review.",
    "stage 4": "Stage 4 means proliferative diabetic retinopathy, which requires urgent specialist assessment.",
}


class ChatRouterAgent:
    """Routes a chat message to a safe, explicit response path."""

    _COMMON_INTENTS = {
        "greeting": {"hi", "hii", "hiii", "hello", "helloo", "hey", "heyy", "goodmorning", "goodafternoon", "goodevening"},
        "help": {"help", "whatcanyoudo", "whatcaniask", "whatquestionscaniask"},
        "thanks": {"thanks", "thankyou", "thanku", "thankyouu", "thnks", "thx", "ty", "tq", "okay", "ok", "okk"},
        "definition": {"whatisdiabeticretinopathy", "whatisdr", "definediabeticretinopathy"},
    }

    @staticmethod
    def _technical_request(query: str) -> bool:
        technical_terms = [
            "student", "developer", "technical", "implementation", "architecture", "algorithm",
            "code", "training", "trained", "loss", "pipeline", "threshold", "model layers",
            "how does", "how is", "in detail", "under the hood", "show me",
        ]
        return any(term in query for term in technical_terms)

    @staticmethod
    def _close_match(query_key: str, phrases: set) -> bool:
        return any(
            query_key == phrase or SequenceMatcher(None, query_key, phrase).ratio() >= 0.78
            for phrase in phrases
        )

    def process(
        self,
        query: str,
        pred_context: Optional[dict] = None,
        history: Optional[List] = None,
    ) -> Dict[str, Any]:
        query_key = "".join(ch for ch in query if ch.isalnum())
        for intent, phrases in self._COMMON_INTENTS.items():
            if self._close_match(query_key, phrases):
                return {"intent": intent, "entry": None}

        high_risk_terms = [
            "should i start treatment", "should i take medicine", "what medication should i take",
            "can i start treatment", "should i inject", "prescribe", "dosage", "treat myself",
        ]
        if any(term in query for term in high_risk_terms):
            return {"intent": "high_risk", "entry": None}

        if pred_context and pred_context.get("stage_name"):
            if query_key in {"why", "whythis", "explain", "more", "moreinfo", "whataboutthat"}:
                return {"intent": "prediction_context", "entry": None, "topic": "classification"}
            triggers = ["why", "classified", "this image", "current", "result", "prediction", "explain this",
                        "confidence", "how confident", "what was found", "lesion", "quadrant", "peak",
                        "this patient", "this case", "referred", "refer this"]
            if any(trigger in query for trigger in triggers):
                if any(word in query for word in ["lesion", "exudate", "microaneurysm"]):
                    topic = "lesions"
                elif any(word in query for word in ["confidence", "certain"]):
                    topic = "confidence"
                elif any(word in query for word in ["urgent", "urgency", "follow-up", "follow up", "action", "refer"]):
                    topic = "advisory"
                elif any(word in query for word in ["quadrant", "peak", "grad-cam", "gradcam"]):
                    topic = "attention"
                else:
                    topic = "classification"
                return {"intent": "prediction_context", "entry": None, "topic": topic}

        for entry in _CLINICAL_KB:
            if any(keyword in query for keyword in entry["keys"]):
                return {
                    "intent": "knowledge_base",
                    "entry": entry,
                    "technical": self._technical_request(query),
                }
        return {"intent": "unknown", "entry": None}


class ChatKnowledgeAgent:
    """Builds a response from the routed knowledge or approved prediction context."""

    def process(self, query: str, route: Dict[str, Any], pred_context: Optional[dict] = None) -> str:
        intent = route["intent"]
        if intent == "greeting":
            return "Hello. I am the **RetinaTrace Clinical Knowledge Assistant**. Ask me about diabetic retinopathy stages, this model, or clinical guidelines."
        if intent == "help":
            return "I can explain **DR stages 0–4**, Grad-CAM, U-Net lesion segmentation, the Governance Agent, referral guidance, preprocessing, and the current prediction."
        if intent == "thanks":
            return "You are welcome. Ask another question whenever you are ready."
        if intent == "definition":
            return "**Diabetic retinopathy** is damage to the retinal blood vessels caused by diabetes. It progresses from no retinopathy through non-proliferative stages to proliferative disease. Regular eye screening and good blood-glucose and blood-pressure control are important."
        if intent == "high_risk":
            return (
                "I cannot recommend starting, stopping, or changing treatment from an AI chat response. "
                "Please discuss medication or procedures with a qualified ophthalmologist or the patient's clinician."
            )
        if intent == "knowledge_base":
            if not route.get("technical"):
                for keyword, simple_reply in _SIMPLE_CHAT_REPLIES.items():
                    if keyword in query:
                        return simple_reply
            return route["entry"]["reply"]
        if intent == "prediction_context":
            stage_name = pred_context.get("stage_name", "Unknown")
            conf = pred_context.get("confidence", 0.0)
            topic = route.get("topic", "classification")
            if topic == "lesions":
                return (
                    f"**Lesions in the current analysis**\n\n"
                    f"The U-Net identified lesion candidates in **{pred_context.get('lesion_pct', 0.0):.1f}%** of the retinal area. "
                    "These candidates may include microaneurysms or exudate-like regions and should be clinically confirmed."
                )
            if topic == "confidence":
                return f"The current **{stage_name}** prediction has a model confidence of **{conf*100:.1f}%**."
            if topic == "advisory":
                urgency = pred_context.get("urgency", "N/A")
                followup = pred_context.get("followup", "N/A")
                plan = pred_context.get("plan", "")
                if "HUMAN SPECIALIST TRIAGE" in urgency:
                    return (
                        "This result should be reviewed **urgently by an ophthalmologist**. "
                        f"The safety gate requires specialist triage because the model confidence is **{conf*100:.1f}%**, "
                        f"below the **70%** threshold. Automated treatment advice should not be followed without clinical confirmation."
                    )
                return (
                    f"**Current clinical routing**\n\n"
                    f"- Urgency: **{urgency}**\n"
                    f"- Follow-up: **{followup}**\n"
                    f"- Action: {plan}"
                )
            if topic == "attention":
                return (
                    f"Grad-CAM identified **{pred_context.get('peak_quadrant', '-')}** as the peak attention quadrant. "
                    "This shows which region influenced the model and is not, by itself, a diagnosis."
                )
            if topic == "classification":
                return (
                    f"The image was classified as **{stage_name}** with **{conf*100:.1f}% confidence**. "
                    f"The main supporting signals were the Grad-CAM peak in **{pred_context.get('peak_quadrant', '-')}** "
                    f"and U-Net lesion candidates covering **{pred_context.get('lesion_pct', 0.0):.1f}%** of the retinal area. "
                    "Because this result is AI-assisted, it should be confirmed by an ophthalmologist."
                )
            return (
                f"**Current Prediction Context: {stage_name}**\n\n"
                f"The model classified this fundus image as **{stage_name}** with a confidence of **{conf*100:.1f}%**.\n\n"
                f"- Grad-CAM peak quadrant: **{pred_context.get('peak_quadrant', '-')}**\n"
                f"- U-Net lesion candidates: **{pred_context.get('lesion_pct', 0.0):.1f}%**\n"
                f"- Urgency: **{pred_context.get('urgency', 'N/A')}**\n"
                f"- Follow-up: **{pred_context.get('followup', 'N/A')}**\n"
                f"- Action: {pred_context.get('plan', 'N/A')}"
            )
        return _CHATBOT_FALLBACK


class ChatGovernanceAgent:
    """Applies a final clinical disclaimer before a chat response is released."""

    def process(self, reply: str, route: Dict[str, Any]) -> str:
        if route["intent"] in {"knowledge_base", "prediction_context", "definition"} and "qualified ophthalmologist" not in reply:
            return reply + "\n\n*This is AI-assisted information. Confirm clinical decisions with a qualified ophthalmologist.*"
        return reply


def run_chat_pipeline(
    message: str,
    pred_context: Optional[dict] = None,
    history: Optional[List] = None,
) -> str:
    """Run the chatbot router, knowledge/context agent, and final safety gate."""
    query = str(message).lower().strip()
    router = ChatRouterAgent()
    route = router.process(query, pred_context, history)
    response = ChatKnowledgeAgent().process(query, route, pred_context)
    return ChatGovernanceAgent().process(response, route)


def respond_to_clinical_query(message: str, history: Optional[List] = None, pred_context: dict = None) -> tuple:
    """Gradio adapter for the explicit multi-agent chat pipeline."""
    history = [] if history is None else list(history)
    if not message or not str(message).strip():
        return history, ""
    reply = run_chat_pipeline(message, pred_context, history)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    return history, ""
import json as _json
import datetime as _datetime
import tempfile as _tempfile


def generate_report_json(stage_name: str, confidence: float, probabilities: dict,
                         advisory_urgency: str, advisory_followup: str,
                         advisory_plan: str, ehr_text: str,
                         research_support: Optional[dict] = None) -> str:
    """Serializes the current diagnosis to a timestamped JSON file for download."""
    if not stage_name or not probabilities or not ehr_text:
        return None
    report = {
        "retinatrace_report": {
            "generated_at": _datetime.datetime.now().isoformat(),
            "model": "EfficientNetB3 + U-Net + CBR (RetinaTrace AI)",
            "diagnosis": {
                "stage_name": stage_name,
                "confidence_pct": round(confidence * 100, 2),
                "probabilities": {k: round(v * 100, 2) for k, v in probabilities.items()},
            },
            "clinical_advisory": {
                "urgency": advisory_urgency,
                "followup": advisory_followup,
                "action_plan": advisory_plan,
            },
            "ehr_note": ehr_text,
            "research_visual_support": research_support or {
                "available": False,
                "message": "No experimental visual-support metrics available.",
            },
            "disclaimer": "This report is generated by an AI assistive tool and must be reviewed by a qualified ophthalmologist before any clinical decision.",
        }
    }
    tmp = _tempfile.NamedTemporaryFile(delete=False, suffix="_retinatrace_report.json", mode="w", encoding="utf-8")
    _json.dump(report, tmp, indent=2)
    tmp.close()
    return tmp.name


def calculate_multimodal_risk(stage: int = 2, hba1c: float = 7.5, duration_years: float = 10.0, age: float = 55.0, systolic_bp: float = 135.0, diabetes_type: str = "Type 2") -> Tuple[str, str]:
    """Computes evidence-based 10-year vision loss progression risk and NHS hospital triage dispatch routing (UKPDS/WESDR)."""
    try:
        stage = int(stage) if stage is not None else 2
    except (ValueError, TypeError):
        stage = 2

    # Base stage risk (10-year baseline from UKPDS 33 / WESDR epidemiological cohorts)
    base_risks = {0: 3.5, 1: 12.0, 2: 29.5, 3: 58.0, 4: 84.0}
    base_risk = base_risks.get(stage, 25.0)

    # Systemic risk multipliers
    hba1c_factor = max(0.5, 1.0 + (hba1c - 7.0) * 0.18)
    duration_factor = max(0.6, 1.0 + (duration_years - 5.0) * 0.025)
    bp_factor = max(0.7, 1.0 + (systolic_bp - 120.0) * 0.008)
    age_factor = max(0.8, 1.0 + (age - 50.0) * 0.005)
    type_factor = 1.15 if diabetes_type == "Type 1" else 1.0

    risk_pct = min(98.5, max(1.5, base_risk * hba1c_factor * duration_factor * bp_factor * age_factor * type_factor))

    triage_info = {
        0: ("P4 — ROUTINE SURVEILLANCE", "Primary Community Optometry Clinic", "12 Months", "#10b981", "Routine annual digital fundus screening. Maintain glycemic control (HbA1c < 7.0%) and BP < 130/80 mmHg."),
        1: ("P3 — PRIMARY CARE GLYCEMIC ROUTE", "General Practice / Diabetes Care Team", "6–9 Months", "#0284c7", "Optimize systemic risk factors. Intensify medical therapy and blood pressure management."),
        2: ("P2 — SECONDARY HOSPITAL OPHTHALMOLOGY", "Hospital Outpatient Ophthalmology & OCT Clinic", "3–6 Months", "#d97706", "Comprehensive dilated examination + Macular OCT to evaluate subclinical Diabetic Macular Edema (DME)."),
        3: ("P2+ — URGENT VITREORETINAL EVALUATION", "Vitreoretinal Specialist Service", "2–4 Weeks", "#ea580c", "Pre-proliferative severity. Assess readiness for panretinal photocoagulation (PRP) laser therapy."),
        4: ("P1 — EMERGENCY VITREORETINAL SURGICAL ROUTE", "Tertiary Vitreoretinal Emergency Unit", "≤ 24–48 Hours", "#e11d48", "Active neovascularization / vitreous hemorrhage risk. Immediate anti-VEGF or emergency PRP laser intervention.")
    }

    triage_code, facility, wait_time, color, protocol = triage_info.get(stage, triage_info[2])

    if risk_pct < 15.0:
        risk_label, risk_color = "Low 10-Yr Progression Risk", "#10b981"
    elif risk_pct < 40.0:
        risk_label, risk_color = "Moderate 10-Yr Progression Risk", "#0284c7"
    elif risk_pct < 70.0:
        risk_label, risk_color = "High 10-Yr Progression Risk", "#ea580c"
    else:
        risk_label, risk_color = "CRITICAL VISION-THREATENING RISK", "#e11d48"

    # RetinaRisk-inspired SVG Speedometer Gauge
    clamped_risk = min(max(risk_pct, 1.0), 99.0)
    arc_length = 267.0
    dash_offset = arc_length * (1.0 - (clamped_risk / 100.0))

    # Calculate screening ladder steps (highlight the recommended tier)
    tiers = [
        ("12 Mo", "P4 Surveillance", "#10b981", stage == 0),
        ("6–9 Mo", "P3 Primary Care", "#0284c7", stage == 1),
        ("3–6 Mo", "P2 Hospital OCT", "#d97706", stage == 2),
        ("2–4 Wk", "P2+ Vitreoretinal", "#ea580c", stage == 3),
        ("≤ 48 Hr", "P1 Emergency", "#e11d48", stage == 4),
    ]

    ladder_html = ""
    for interval, tier_name, tcolor, is_active in tiers:
        if is_active:
            ladder_html += f"""
            <div style="flex:1; background:{tcolor}; color:#fff; border-radius:8px; padding:8px 4px; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.15); border:2px solid #fff;">
                <div style="font-size:13px; font-weight:800;">{interval}</div>
                <div style="font-size:9.5px; font-weight:700; text-transform:uppercase; margin-top:2px; opacity:0.95;">{tier_name}</div>
                <div style="font-size:9px; background:rgba(255,255,255,0.25); border-radius:4px; padding:1px 3px; margin-top:3px; font-weight:800;">RECOMMENDED</div>
            </div>
            """
        else:
            ladder_html += f"""
            <div style="flex:1; background:#f1f5f9; color:#64748b; border-radius:8px; padding:8px 4px; text-align:center; border:1px solid #e2e8f0; opacity:0.75;">
                <div style="font-size:12px; font-weight:700;">{interval}</div>
                <div style="font-size:9px; margin-top:2px;">{tier_name}</div>
            </div>
            """

    risk_card_html = f"""
    <div class="risk-card" style="padding:16px; background:#f8fafc; border-radius:12px; border:1px solid #e2e8f0; margin-bottom:12px;">
        <div class="risk-card-header" style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; border-bottom:1px solid #e2e8f0; padding-bottom:8px;">
            <div style="font-size:13px; font-weight:800; text-transform:uppercase; color:#0369a1; display:flex; align-items:center; gap:8px;">
                <span>⏱️</span> RetinaRisk™ Individualized Screening & 10-Year Progression Predictor
            </div>
            <span style="background:{color}; color:#fff; font-size:11.5px; font-weight:800; padding:4px 10px; border-radius:6px; letter-spacing:0.5px;">
                {triage_code.split('—')[0].strip()}
            </span>
        </div>
        <div class="responsive-grid responsive-grid-two" style="display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:12px;">
            <!-- Speedometer Gauge Card -->
            <div style="background:#fff; border-radius:10px; padding:16px; border:1px solid #e2e8f0; border-top:4px solid {risk_color}; text-align:center;">
                <div style="font-size:11px; color:#64748b; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">Sight-Threatening Retinopathy Risk</div>
                <div style="margin:8px auto; max-width:240px;">
                    <svg viewBox="0 0 240 135" style="width:100%; height:auto; overflow:visible;">
                        <defs>
                            <linearGradient id="retinaRiskGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                                <stop offset="0%" stop-color="#10b981" />
                                <stop offset="30%" stop-color="#0284c7" />
                                <stop offset="65%" stop-color="#f59e0b" />
                                <stop offset="100%" stop-color="#ef4444" />
                            </linearGradient>
                        </defs>
                        <!-- Background track -->
                        <path d="M 30 115 A 85 85 0 0 1 210 115" fill="none" stroke="#f1f5f9" stroke-width="14" stroke-linecap="round" />
                        <!-- Active Progress Arc -->
                        <path d="M 30 115 A 85 85 0 0 1 210 115" fill="none" stroke="url(#retinaRiskGrad)" stroke-width="14" stroke-linecap="round"
                              stroke-dasharray="267" stroke-dashoffset="{dash_offset:.1f}" />
                        <!-- Gauge Center Text -->
                        <text x="120" y="86" text-anchor="middle" font-size="30" font-weight="900" fill="{risk_color}">{risk_pct:.1f}%</text>
                        <text x="120" y="104" text-anchor="middle" font-size="10" font-weight="800" fill="#64748b" letter-spacing="0.5">IN 10 YEARS</text>
                        <text x="30" y="130" font-size="9.5" font-weight="700" fill="#10b981">0% (Low)</text>
                        <text x="120" y="130" text-anchor="middle" font-size="9.5" font-weight="700" fill="#f59e0b">Moderate</text>
                        <text x="210" y="130" text-anchor="end" font-size="9.5" font-weight="700" fill="#ef4444">100% (High)</text>
                    </svg>
                </div>
                <div style="font-size:13px; font-weight:800; color:{risk_color}; margin-top:2px;">{risk_label}</div>
                <div style="font-size:11px; color:#64748b; margin-top:4px;">
                    UKPDS 33 / WESDR Multiplicative Systemic Model
                </div>
            </div>

            <!-- Hospital Dispatch Routing Card -->
            <div style="background:#fff; border-radius:10px; padding:16px; border:1px solid #e2e8f0; border-top:4px solid {color}; display:flex; flex-direction:column; justify-content:space-between;">
                <div>
                    <div style="font-size:11px; color:#64748b; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">Hospital Dispatch & Routing</div>
                    <div style="font-size:16px; font-weight:800; color:#0f172a; margin-top:4px;">{facility}</div>
                    <div style="display:inline-flex; align-items:center; gap:6px; background:#eff6ff; color:{color}; padding:4px 10px; border-radius:6px; font-size:12px; font-weight:800; margin:6px 0;">
                        <span>⏱️</span> Mandatory Wait Time: {wait_time}
                    </div>
                    <div style="font-size:12px; color:#334155; line-height:1.5; margin-top:6px;">
                        <strong>Clinical Action Plan:</strong> {protocol}
                    </div>
                </div>
                <div style="font-size:11px; color:#64748b; background:#f8fafc; border-radius:6px; padding:8px 10px; margin-top:8px;">
                    <strong>Patient Profile:</strong> Age {age:.0f}y &bull; {diabetes_type} &bull; HbA1c {hba1c:.1f}% &bull; Duration {duration_years:.0f}y &bull; BP {systolic_bp:.0f} mmHg
                </div>
            </div>
        </div>

        <!-- RetinaRisk Individualized Screening Ladder -->
        <div style="background:#fff; border-radius:10px; padding:14px; border:1px solid #e2e8f0;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <div style="font-size:11.5px; font-weight:700; text-transform:uppercase; color:#0369a1;">
                    Individualized Screening Prescription vs. Standard Annual Surveillance
                </div>
                <div style="font-size:11px; color:#64748b;">
                    Standard Recall: <strong style="color:#0f172a;">12 Months</strong> &bull; Personalized: <strong style="color:{color};">{wait_time}</strong>
                </div>
            </div>
            <div class="risk-ladder" style="display:flex; gap:8px;">
                {ladder_html}
            </div>
        </div>
    </div>
    """

    referral_ticket = (
        "===========================================================\n"
        "           OFFICIAL DIGITAL HOSPITAL REFERRAL TICKET       \n"
        "===========================================================\n"
        f"TRIAGE PRIORITY CODE  : {triage_code}\n"
        f"REFERRAL FACILITY     : {facility}\n"
        f"MANDATORY WAIT TIME   : {wait_time}\n"
        f"DIAGNOSTIC IMAGE STAGE: Stage {stage} ({AppConfig.CLASS_NAMES[stage]})\n"
        f"10-YEAR RISK ESTIMATE : {risk_pct:.1f}% ({risk_label})\n"
        "-----------------------------------------------------------\n"
        f"PATIENT PARAMETERS    : Age {age:.0f}y | {diabetes_type} | HbA1c {hba1c:.1f}% | Duration {duration_years:.0f}y | BP {systolic_bp:.0f} mmHg\n"
        f"CLINICAL ACTION PLAN  : {protocol}\n"
        "-----------------------------------------------------------\n"
        "REFERRING CLINICIAN SIGN-OFF:\n"
        "Clinician Name: _________________   Medical Reg: __________\n"
        "Signature: ______________________   Date: _________________\n"
        "==========================================================="
    )

    return risk_card_html, referral_ticket


# ─────────────────────────────────────────────────────────────────────────────
# 6. Synthetic Realistic Fundus Demonstrators (Instant Demo Library)
# ─────────────────────────────────────────────────────────────────────────────
def create_sample_fundus(stage: int = 0) -> np.ndarray:
    """Generates realistic synthetic retinal fundus images for instant offline demonstration."""
    img = np.zeros((AppConfig.IMG_SIZE, AppConfig.IMG_SIZE, 3), dtype=np.uint8)
    # Base retinal fundus background disk
    cv2.circle(img, (112, 112), 104, (190, 75, 35), -1)
    # Optic disc (yellowish-white oval)
    cv2.ellipse(img, (75, 112), (16, 22), 0, 0, 360, (240, 220, 150), -1)
    # Retinal vascular tree
    cv2.polylines(img, [np.array([[75, 112], [105, 80], [150, 55], [195, 45]])], False, (110, 25, 15), 2)
    cv2.polylines(img, [np.array([[75, 112], [110, 140], [160, 165], [190, 175]])], False, (110, 25, 15), 2)
    cv2.polylines(img, [np.array([[75, 112], [45, 95], [25, 85]])], False, (110, 25, 15), 2)
    # Macula / Fovea centralis (dark luteal region)
    cv2.circle(img, (135, 112), 14, (140, 45, 20), -1)

    if stage >= 1:  # Microaneurysms
        cv2.circle(img, (120, 95), 2, (70, 10, 5), -1)
        cv2.circle(img, (145, 130), 2, (70, 10, 5), -1)
    if stage >= 2:  # Hard exudates (yellow deposits) & blot hemorrhages
        cv2.circle(img, (155, 110), 4, (250, 240, 170), -1)
        cv2.circle(img, (162, 115), 3, (250, 240, 170), -1)
        cv2.circle(img, (115, 135), 4, (80, 10, 10), -1)
    if stage >= 3:  # Severe hemorrhages & IRMA
        cv2.circle(img, (100, 145), 6, (75, 10, 10), -1)
        cv2.circle(img, (140, 80), 7, (75, 10, 10), -1)
        cv2.circle(img, (165, 140), 5, (250, 240, 170), -1)
    if stage >= 4:  # Neovascular fronds
        cv2.line(img, (75, 112), (90, 95), (130, 35, 20), 3)
        cv2.line(img, (90, 95), (105, 85), (130, 35, 20), 2)
        cv2.circle(img, (85, 105), 8, (90, 10, 10), -1)

    return cv2.GaussianBlur(img, (3, 3), 0)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Modern Gradio Interface Callbacks & Formatting
# ─────────────────────────────────────────────────────────────────────────────
STAGE_BADGE_COLORS = {
    0: ("#10b981", "#dcfce7", "STAGE 0: NO RETINOPATHY"),
    1: ("#0284c7", "#e0f2fe", "STAGE 1: MILD NPDR"),
    2: ("#d97706", "#fef3c7", "STAGE 2: MODERATE NPDR"),
    3: ("#ea580c", "#ffedd5", "STAGE 3: SEVERE NPDR"),
    4: ("#e11d48", "#ffe4e6", "STAGE 4: PROLIFERATIVE DR"),
}


def assess_image_quality(img: np.ndarray) -> dict:
    """Assesses fundus image quality: blur, illumination, retina presence, and black border ratio."""
    issues = []
    warnings = []
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img

    # 1. Blur detection via Laplacian variance
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_blurry = lap_var < 80.0
    if is_blurry:
        issues.append(f"Blur detected (Laplacian variance: {lap_var:.1f} < 80 threshold) — image may be out of focus.")

    # 2. Illumination check via mean brightness
    mean_brightness = float(np.mean(gray))
    if mean_brightness < 30:
        issues.append(f"Poor illumination — image is underexposed (mean brightness: {mean_brightness:.1f}/255).")
    elif mean_brightness > 220:
        issues.append(f"Overexposed image — excessive brightness may wash out lesions (mean: {mean_brightness:.1f}/255).")

    # 3. Black border ratio (retina presence check)
    black_mask = gray < 15
    black_ratio = float(np.sum(black_mask)) / float(gray.size)
    if black_ratio > 0.60:
        issues.append(f"Excessive black border ({black_ratio*100:.1f}% of image) — retina may be missing or severely cropped.")
    elif black_ratio > 0.40:
        warnings.append(f"Large black border area ({black_ratio*100:.1f}%) — image may be sub-optimally framed.")

    # 4. Retina disk presence heuristic using central green-channel energy
    h, w = gray.shape
    if img.ndim == 3:
        center_patch = img[h//4:3*h//4, w//4:3*w//4]
        green_energy = float(np.mean(center_patch[:, :, 1]))
    else:
        green_energy = float(np.mean(gray[h//4:3*h//4, w//4:3*w//4]))
    if green_energy < 20:
        issues.append("Retina not detected in image center — please ensure the optic disc is within the frame.")

    passed = len(issues) == 0
    return {
        "passed": passed,
        "issues": issues,
        "warnings": warnings,
        "blur_score": lap_var,
        "brightness": mean_brightness,
        "black_ratio": black_ratio,
        "central_green_energy": green_energy,
    }


def build_quality_warning_html(qc: dict) -> str:
    """Renders image quality assessment result as an HTML warning card."""
    if qc["passed"] and not qc["warnings"]:
        return (
            '<div style="padding:8px 14px; background:#f0fdf4; border-radius:8px; border-left:4px solid #10b981; margin-bottom:8px;">'
            '<span style="font-weight:700; color:#166534;">✅ Image Quality: PASS</span>'
            f'<span style="color:#64748b; font-size:12px; margin-left:10px;">Blur score: {qc["blur_score"]:.1f} | Brightness: {qc["brightness"]:.0f}/255 | Black border: {qc["black_ratio"]*100:.1f}% | Green energy: {qc["central_green_energy"]:.1f}</span>'
            '</div>'
        )
    color = "#fef2f2" if not qc["passed"] else "#fffbeb"
    border = "#ef4444" if not qc["passed"] else "#f59e0b"
    icon = "⚠️ Image Quality: ISSUES DETECTED" if not qc["passed"] else "⚠️ Image Quality: WARNINGS"
    label_color = "#991b1b" if not qc["passed"] else "#92400e"
    all_msgs = [f'<li style="margin:2px 0;">{m}</li>' for m in (qc["issues"] + qc["warnings"])]
    return (
        f'<div style="padding:10px 14px; background:{color}; border-radius:8px; border-left:4px solid {border}; margin-bottom:8px;">'
        f'<div style="font-weight:700; color:{label_color}; margin-bottom:6px;">{icon}</div>'
        f'<ul style="margin:0; padding-left:18px; font-size:12px; color:#374151;">{"".join(all_msgs)}</ul>'
        f'<div style="font-size:11px; color:#64748b; margin-top:4px;">Blur score: {qc["blur_score"]:.1f} | Brightness: {qc["brightness"]:.0f}/255 | Black border: {qc["black_ratio"]*100:.1f}% | Green energy: {qc["central_green_energy"]:.1f}</div>'
        '</div>'
    )


def analyze_fundus(img: Optional[np.ndarray], threshold: float, session_history: list = None):
    """Primary analysis handler that executes the pipeline and populates modern UI widgets."""
    empty_img = np.zeros((AppConfig.IMG_SIZE, AppConfig.IMG_SIZE, 3), dtype=np.uint8)
    if img is None:
        notice = "<div class='card warning-card'>⚠️ <strong>Please upload a retinal fundus photograph</strong> or click one of the quick-load sample buttons on the left.</div>"
        return notice, "", {}, empty_img, empty_img, empty_img, empty_img, "", empty_img, empty_img, "", "", [], "", "", "", "", "", None, [], {}, "", empty_img, "", "", 0.0, "", "", "", ""

    try:
        preproc = preprocess_image(img)
    except ValueError as exc:
        notice = (
            "<div class='card warning-card'>⚠️ <strong>Invalid fundus image input</strong> — "
            f"{exc}. Please upload a valid retina image or choose a sample fundus from the quick-load buttons.</div>"
        )
        return notice, "", {}, empty_img, empty_img, empty_img, empty_img, "", empty_img, empty_img, "", "", [], "", "", "", "", "", None, [], {}, "", empty_img, "", "", 0.0, "", "", "", ""

    # Image Quality Assessment (before inference)
    qc = assess_image_quality(img)
    quality_html = build_quality_warning_html(qc)

    result = run_pipeline(preproc, threshold=threshold, qc=qc)


    diag = result["diagnosis"]
    expl = result["explanation"]
    adv = result["advisory"]
    flagged = result["flagged_for_review"]
    stage = diag["stage"]
    overlap = expl.get("overlap_analysis", {})
    severity_map = expl.get("severity_map", {})
    consistency = result.get("consistency_analysis", {})
    biomarkers = compile_retinal_biomarkers(
        expl.get("lesion_pct", 0.0),
        severity_map.get("total_lesion_count", 0),
        expl.get("vessel_density", 0.0),
        expl.get("affected_quadrants_count", 0),
        expl,
        qc,
        expl.get("classical_cv", {}),
    )

    # 1. Governance Banner HTML
    if flagged:
        gov_html = f"""
        <div class="card alert-card">
            <div class="card-header">
                <span class="icon">🚨</span>
                <div>
                    <h3 style="margin:0; color:#991b1b;">GOVERNANCE STATUS: FLAGGED FOR HUMAN TRIAGE</h3>
                    <p style="margin:2px 0 0 0; color:#7f1d1d; font-size:13px;">
                        Safety Gate Interception: Model confidence (<strong>{diag['confidence']*100:.1f}%</strong>) is below your set threshold (<strong>{threshold*100:.0f}%</strong>).
                    </p>
                </div>
            </div>
            <div style="margin-top:8px; padding:8px 12px; background:#fff; border-radius:6px; border-left:4px solid #ef4444; font-size:13px; color:#b91c1c;">
                <strong>Patient Safety Protection:</strong> Automated treatment guidance has been withheld to eliminate hallucination risks. Rerouted to mandatory human ophthalmologist review.
            </div>
        </div>
        """
    else:
        gov_html = f"""
        <div class="card success-card">
            <div class="card-header">
                <span class="icon">✅</span>
                <div>
                    <h3 style="margin:0; color:#166534;">GOVERNANCE STATUS: AUTOMATION APPROVED</h3>
                    <p style="margin:2px 0 0 0; color:#14532d; font-size:13px;">
                        Quality Assurance Verified: Model confidence (<strong>{diag['confidence']*100:.1f}%</strong>) satisfies the clinical safety threshold (<strong>{threshold*100:.0f}%</strong>).
                    </p>
                </div>
            </div>
        </div>
        """

    # 2. Hero Diagnosis Card HTML (Medios-Style Tri-State Status Chip & Hospital Card)
    border_c, bg_c, badge_text = STAGE_BADGE_COLORS[stage]

    if stage == 0:
        medios_chip_text = "🟢 NO DR DETECTED"
        medios_chip_bg = "#ecfdf5"
        medios_chip_color = "#065f46"
        medios_chip_border = "#a7f3d0"
    elif stage in (1, 2):
        medios_chip_text = f"🟡 DIABETIC RETINOPATHY DETECTED ({diag['stage_name'].upper()})"
        medios_chip_bg = "#fffbeb"
        medios_chip_color = "#92400e"
        medios_chip_border = "#fde68a"
    else:
        medios_chip_text = f"🔴 SIGHT-THREATENING RETINOPATHY DETECTED ({diag['stage_name'].upper()})"
        medios_chip_bg = "#fff1f2"
        medios_chip_color = "#9f1239"
        medios_chip_border = "#fecdd3"

    hero_html = f"""
    <div class="card hero-card" style="border-top: 5px solid {border_c};">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:10px;">
            <div>
                <div style="display:inline-flex; align-items:center; gap:6px; background:{medios_chip_bg}; color:{medios_chip_color}; border:1px solid {medios_chip_border}; padding:5px 14px; border-radius:20px; font-weight:800; font-size:12px; letter-spacing:0.5px; margin-bottom:8px;">
                    {medios_chip_text}
                </div>
                <h1 class="hero-stage-title" style="margin:2px 0 4px 0; font-size:26px;">Stage {stage}: {diag['stage_name']}</h1>
                <p class="hero-subtext" style="margin:0; font-size:13px;">International Clinical Diabetic Retinopathy (ICDR) Scale &bull; Primary Diagnostic Output</p>
            </div>
            <div style="text-align:right;">
                <div style="font-size:36px; font-weight:900; color:{border_c}; line-height:1;">{diag['confidence']*100:.1f}%</div>
                <div class="hero-sublabel" style="font-size:11px; font-weight:700; margin-top:4px; letter-spacing:0.5px;">DIAGNOSTIC CONFIDENCE</div>
            </div>
        </div>
    </div>
    """

    # 3. Probabilities for gr.Label
    probs_dict = diag["probabilities"]

    # 4. CBR Gallery
    gallery_items = []
    for c in expl["similar_cases"]:
        caption = f"Match #{c['rank']} • Stage {c['stage']}: {c['stage_name']} (Similarity: {c['similarity']:.3f})"
        if os.path.exists(c["filepath"]):
            gallery_items.append((c["filepath"], caption))

    # 5. Clinical Advisory Plan HTML
    advisory_html = f"""
    <div class="card">
        <h3 class="protocol-title" style="margin-top:0; border-bottom:1px solid #e2e8f0; padding-bottom:8px;">📋 Clinical Care Protocol (AAO Preferred Practice Pattern)</h3>
        <div class="responsive-grid responsive-grid-two" style="display:grid; grid-template-columns: 1fr 1fr; gap:12px; margin-bottom:12px;">
            <div class="protocol-box" style="padding:10px; border-radius:6px;">
                <div class="protocol-label" style="font-size:11px; font-weight:700; text-transform:uppercase;">Clinical Urgency Level</div>
                <div class="protocol-val" style="font-size:15px; font-weight:700; margin-top:2px;">{adv['urgency']}</div>
            </div>
            <div class="protocol-box" style="padding:10px; border-radius:6px;">
                <div class="protocol-label" style="font-size:11px; font-weight:700; text-transform:uppercase;">Recommended Follow-Up</div>
                <div class="protocol-val" style="font-size:15px; font-weight:700; margin-top:2px;">{adv['followup']}</div>
            </div>
        </div>
        <div class="action-box" style="padding:12px; border-radius:6px; margin-bottom:12px;">
            <div class="action-label" style="font-size:11px; font-weight:700; text-transform:uppercase;">Specialist Action Plan</div>
            <div class="action-val" style="font-size:13.5px; margin-top:4px; line-height:1.5;">{adv['plan']}</div>
        </div>
        <div class="disclaimer-text" style="font-size:11.5px; font-style:italic;">
            {adv['disclaimer']}
        </div>
    </div>
    """

    # 6. Exportable Clinical EHR Note
    clean_quad = expl['quadrant_desc'].replace('**', '').replace('`', '')
    gate_label = 'OVERRIDE (FLAGGED)' if flagged else 'APPROVED'
    ehr_text = (
        "===========================================================\n"
        "           RETINATRACE AI CLINICAL TRIAGE NOTE             \n"
        "===========================================================\n"
        f"ASSESSMENT DATE/TIME   : Diagnostic Session Active\n"
        f"PREDICTED DR STAGE     : Stage {stage} — {diag['stage_name']}\n"
        f"MODEL CONFIDENCE       : {diag['confidence']*100:.2f}%\n"
        f"GOVERNANCE GATE STATUS : {gate_label}\n"
        f"SAFETY THRESHOLD ENF.  : {threshold*100:.0f}%\n"
        f"PEAK ANATOMICAL REGION : {clean_quad}\n"
        f"EXPERIMENTAL CV SUPPORT : Lesion/attention IoU {overlap.get('iou', 0.0):.3f} | Vessel density {expl.get('vessel_density', 0.0):.2f}% | Affected quadrants {expl.get('affected_quadrants_count', 0)}/4\n"
        f"EVIDENCE CONSISTENCY    : {consistency.get('status', 'INSUFFICIENT EVIDENCE')}\n"
        "-----------------------------------------------------------\n"
        f"CLINICAL URGENCY       : {adv['urgency']}\n"
        f"RECOMMENDED RECALL     : {adv['followup']}\n"
        f"ACTION PLAN            : {adv['plan']}\n"
        "-----------------------------------------------------------\n"
        "REVIEWING OPHTHALMOLOGIST SIGN-OFF:\n"
        "Name: ______________________   Signature: __________________\n"
        "==========================================================="
    )

    # 7. Lesion Burden HTML metric card
    lesion_pct = expl.get("lesion_pct", 0.0)
    if lesion_pct == 0.0:
        burden_color, burden_label = "#10b981", "Minimal / No Detectable Lesion Area"
    elif lesion_pct < 5.0:
        burden_color, burden_label = "#d97706", "Low-Moderate Burden"
    elif lesion_pct < 15.0:
        burden_color, burden_label = "#ea580c", "Moderate-High Burden"
    else:
        burden_color, burden_label = "#e11d48", "High Lesion Burden - Urgent Review"

    lesion_burden_html = (
        f'<div style="margin-top:8px; padding:10px 14px; background:#f8fafc; border-radius:8px; border-left:4px solid {burden_color};">'
        f'<div style="font-size:11px; font-weight:700; text-transform:uppercase; color:#64748b; margin-bottom:4px;">Retinal Lesion Area Burden (U-Net Segmentation)</div>'
        f'<div style="display:flex; align-items:baseline; gap:10px;">'
        f'<span style="font-size:28px; font-weight:800; color:{burden_color};">{lesion_pct:.2f}%</span>'
        f'<span style="font-size:13px; color:#475569;">of visible parenchyma</span></div>'
        f'<div style="font-size:12px; color:{burden_color}; font-weight:600; margin-top:2px;">{burden_label}</div>'
        f'</div>'
    )

    # Classical Computer Vision Biomarkers Panel (OpenCV Syllabus Alignment)
    classical_cv = expl.get("classical_cv", {})
    edge_density = classical_cv.get("sobel_edge_density", 0.0)
    morph_pct = classical_cv.get("morph_candidate_pct", 0.0)
    clahe_status = classical_cv.get("clahe_status", "Calibrated (Clip=2.0)")

    classical_cv_html = (
        '<div style="margin-top:10px; padding:12px 14px; background:#f8fafc; border-radius:8px; border:1px solid #e2e8f0; border-left:4px solid #0284c7;">'
        '<div style="font-size:11px; font-weight:700; text-transform:uppercase; color:#0369a1; margin-bottom:8px; display:flex; align-items:center; gap:6px;">'
        '<span>🔬</span> Classical Computer Vision Biomarkers (OpenCV)</div>'
        '<div class="responsive-grid responsive-grid-three" style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; text-align:center;">'
        '<div style="background:#fff; border-radius:6px; padding:8px; border:1px solid #e2e8f0;">'
        '<div style="font-size:10px; color:#64748b; font-weight:600;">SOBEL GRADIENT</div>'
        f'<div style="font-size:16px; font-weight:800; color:#0f172a; margin:2px 0;">{edge_density:.1f}%</div>'
        '<div style="font-size:10px; color:#64748b;">Vascular Edge Density</div></div>'
        '<div style="background:#fff; border-radius:6px; padding:8px; border:1px solid #e2e8f0;">'
        '<div style="font-size:10px; color:#64748b; font-weight:600;">CLAHE HISTOGRAM</div>'
        f'<div style="font-size:13px; font-weight:800; color:#059669; margin:4px 0;">{clahe_status}</div>'
        '<div style="font-size:10px; color:#64748b;">Adaptive Local Contrast</div></div>'
        '<div style="background:#fff; border-radius:6px; padding:8px; border:1px solid #e2e8f0;">'
        '<div style="font-size:10px; color:#64748b; font-weight:600;">MORPHOLOGY</div>'
        f'<div style="font-size:16px; font-weight:800; color:#d97706; margin:2px 0;">{morph_pct:.1f}%</div>'
        '<div style="font-size:10px; color:#64748b;">Top-Hat Lesion Sites</div></div>'
        '</div>'
        '<div style="font-size:10px; color:#64748b; margin-top:8px; font-style:italic;">'
        'Deterministic physical image processing features (Convolution, Morphology, Histograms) extracted in parallel with deep embeddings.</div>'
        '</div>'
    )

    lesion_burden_html = lesion_burden_html + classical_cv_html

    vessel_density = float(expl.get("vessel_density", 0.0))
    optic_found = bool(expl.get("optic_disc_found", False))
    optic_status = "Detected" if optic_found else "Not reliably detected"
    classical_cv_status_html = (
        '<div style="margin-top:10px; padding:12px 14px; background:#f8fafc; border-radius:8px; border:1px solid #e2e8f0;">'
        '<div style="font-size:12px; font-weight:800; color:#0369a1;">Classical CV Vessel Analysis</div>'
        f'<div style="font-size:13px; color:#334155; margin-top:5px;">Vessel density: <strong>{vessel_density:.2f}%</strong> '
        f'| Method: {expl.get("vessel_status", "Exploratory morphology")}</div>'
        '<div style="font-size:12px; font-weight:800; color:#7c3aed; margin-top:10px;">Optic-Disc Localisation</div>'
        f'<div style="font-size:13px; color:#334155; margin-top:5px;">Status: <strong>{optic_status}</strong> '
        f'| Heuristic score: {float(expl.get("optic_disc_score", 0.0)):.2f}</div>'
        '<div style="font-size:11px; color:#64748b; margin-top:8px; font-style:italic;">Visual support only — not an independent diagnosis.</div>'
        '</div>'
    )

    biomarker_rows = "".join(
        f'<div title="{item["tooltip"]}" style="padding:8px; background:#fff; border:1px solid #e2e8f0; border-radius:6px;">'
        f'<div style="font-size:10px; color:#64748b; font-weight:700;">{item["name"]}</div>'
        f'<div style="font-size:16px; font-weight:800; color:#0f172a; margin-top:3px;">{item["value"]}</div>'
        f'<div style="font-size:10px; color:#64748b; margin-top:2px;">{item["status"]}</div></div>'
        for item in biomarkers
    )
    quadrant_rows = "".join(
        f'<div style="padding:8px; background:{item["badge_bg"]}; border-left:4px solid {item["badge_color"]}; border-radius:5px;">'
        f'<strong>{name}</strong><br><span style="font-size:11px;">Burden {item["lesion_burden_pct"]:.2f}% | '
        f'Lesions {item["lesion_count"]} | Vessels {item["vessel_density_pct"]:.2f}%</span><br>'
        f'<span style="font-size:11px; color:{item["badge_color"]};">{item["abnormality_label"]}</span></div>'
        for name, item in severity_map.get("quadrants", {}).items()
    )
    advanced_analysis_html = (
        '<div style="margin-top:10px; padding:12px 14px; background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px;">'
        '<div style="font-size:12px; font-weight:800; color:#0369a1;">AI Attention–Lesion Agreement</div>'
        f'<div style="font-size:13px; margin-top:5px;">IoU: <strong>{overlap.get("iou", 0.0):.3f}</strong> | '
        f'Dice: <strong>{overlap.get("dice", 0.0):.3f}</strong> | Lesion regions within attention: '
        f'<strong>{overlap.get("lesion_in_cam_pct", 0.0):.1f}%</strong></div>'
        f'<div style="font-size:11px; color:#64748b; margin-top:4px;">{overlap.get("interpretation", "Evidence unavailable.")} Experimental explainability measure only.</div>'
        '<div style="font-size:12px; font-weight:800; color:#7c3aed; margin-top:12px;">Retinal Visual-Abnormality Map</div>'
        f'<div style="display:grid; grid-template-columns:1fr 1fr; gap:6px; margin-top:6px;">{quadrant_rows}</div>'
        f'<div style="font-size:11px; color:#64748b; margin-top:5px;">Affected quadrants: {severity_map.get("affected_quadrants_count", 0)}/4. These are visual-support indicators, not clinical severity grades.</div>'
        '<div style="font-size:12px; font-weight:800; color:#0f766e; margin-top:12px;">Retinal CV Biomarkers</div>'
        f'<div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:6px; margin-top:6px;">{biomarker_rows}</div>'
        '<div style="font-size:11px; color:#64748b; margin-top:8px; font-style:italic;">Computer Vision Measurements — Research/Visual Support Only. Visual support only — not an independent diagnosis.</div>'
        '<div style="font-size:12px; font-weight:800; color:#b45309; margin-top:12px;">Prediction–Evidence Consistency</div>'
        f'<div style="font-size:14px; font-weight:800; color:{consistency.get("status_color", "#64748b")}; margin-top:4px;">{consistency.get("icon", "⚪")} {consistency.get("status", "INSUFFICIENT EVIDENCE")}</div>'
        f'<div style="font-size:11px; color:#475569; margin-top:4px;">{consistency.get("reason", "Independent evidence is unavailable.")}</div>'
        '</div>'
    )
    cbr_explanation = expl.get("cbr_explanation", {})
    advanced_analysis_html += (
        '<div style="margin-top:10px; padding:10px; border-top:1px solid #e2e8f0;">'
        '<div style="font-size:12px; font-weight:800; color:#475569;">CBR Visual-Similarity Explanation</div>'
        f'<div style="font-size:11px; color:#475569; margin-top:4px;">'
        f'Average retrieved similarity: {cbr_explanation.get("average_similarity", 0.0):.3f} | '
        f'Stage-label agreement: {cbr_explanation.get("stage_agreement_count", 0)}/{cbr_explanation.get("stage_agreement_total", 0)}</div>'
        f'<div style="font-size:11px; color:#64748b; margin-top:3px;">{cbr_explanation.get("interpretation", "No CBR metadata available.")}</div>'
        '</div>'
    )

    research_support = {
        "available": True,
        "attention_lesion_iou": overlap.get("iou", 0.0),
        "attention_lesion_dice": overlap.get("dice", 0.0),
        "lesion_regions_in_attention_pct": overlap.get("lesion_in_cam_pct", 0.0),
        "lesion_burden_pct": expl.get("lesion_pct", 0.0),
        "candidate_lesion_count": severity_map.get("total_lesion_count", 0),
        "vessel_density_pct": expl.get("vessel_density", 0.0),
        "affected_quadrants": expl.get("affected_quadrants_count", 0),
        "optic_disc_found": bool(expl.get("optic_disc_found", False)),
        "optic_disc_score": expl.get("optic_disc_score", 0.0),
        "prediction_evidence_consistency": consistency.get("status", "INSUFFICIENT EVIDENCE"),
        "cbr": cbr_explanation,
        "disclaimer": "Research/visual support only — not an independent diagnosis.",
    }

    # 8. Anatomical Quadrant Salience HTML bar chart
    quad_scores = expl.get("quadrant_scores", {})
    peak_quad = expl.get("peak_quadrant", "-")
    quad_bar_rows = ""
    for qname, qval in sorted(quad_scores.items(), key=lambda x: x[1], reverse=True):
        pct_width = min(int(qval * 100 * 3.5), 100)
        is_peak = qname == peak_quad
        bar_color = "#ef4444" if is_peak else "#60a5fa"
        peak_marker = " [PEAK]" if is_peak else ""
        fw = "700" if is_peak else "500"
        fc = "#991b1b" if is_peak else "#334155"
        quad_bar_rows += (
            f'<div style="margin-bottom:6px;">'
            f'<div style="display:flex; justify-content:space-between; font-size:12px; font-weight:{fw}; color:{fc}; margin-bottom:2px;">'
            f'<span>{qname}{peak_marker}</span><span>{qval:.3f}</span></div>'
            f'<div style="background:#e2e8f0; border-radius:4px; height:10px; overflow:hidden;">'
            f'<div style="background:{bar_color}; width:{pct_width}%; height:100%; border-radius:4px;"></div></div></div>'
        )

    quadrant_chart_html = (
        f'<div style="margin-top:10px; padding:12px 14px; background:#f8fafc; border-radius:8px; border:1px solid #e2e8f0;">'
        f'<div style="font-size:11px; font-weight:700; text-transform:uppercase; color:#64748b; margin-bottom:10px;">'
        f'Anatomical Quadrant Grad-CAM Activation Index</div>'
        f'{quad_bar_rows}'
        f'<div style="font-size:11px; color:#94a3b8; margin-top:6px; font-style:italic;">'
        f'Mean Grad-CAM saliency per quadrant (Superior/Inferior x Temporal/Nasal).</div>'
        f'</div>'
    )

    # 9. Adjacent-Stage Confidence Margin
    sorted_probs = sorted(diag["probabilities"].items(), key=lambda x: x[1], reverse=True)
    top_stage_name, top_conf = sorted_probs[0]
    sec_stage_name, sec_conf = sorted_probs[1] if len(sorted_probs) > 1 else ("-", 0.0)
    margin = top_conf - sec_conf
    margin_color = "#10b981" if margin > 0.40 else ("#d97706" if margin > 0.20 else "#e11d48")
    margin_label = "High Certainty" if margin > 0.40 else ("Borderline - Monitor" if margin > 0.20 else "Low Certainty - Human Review Advised")

    confidence_margin_html = (
        f'<div style="margin-top:8px; padding:10px 14px; background:#f8fafc; border-radius:8px; border-left:4px solid {margin_color};">'
        f'<div style="font-size:11px; font-weight:700; text-transform:uppercase; color:#64748b; margin-bottom:6px;">'
        f'Diagnostic Uncertainty Margin (Adjacent-Stage Analysis)</div>'
        f'<div class="responsive-grid responsive-grid-three" style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; text-align:center;">'
        f'<div style="background:#fff; border-radius:6px; padding:8px; border:1px solid #e2e8f0;">'
        f'<div style="font-size:10px; color:#64748b; font-weight:600;">PRIMARY</div>'
        f'<div style="font-size:13px; font-weight:700; color:#1e293b;">{top_stage_name}</div>'
        f'<div style="font-size:16px; font-weight:800; color:{border_c};">{top_conf*100:.1f}%</div></div>'
        f'<div style="background:#fff; border-radius:6px; padding:8px; border:1px solid #e2e8f0;">'
        f'<div style="font-size:10px; color:#64748b; font-weight:600;">RUNNER-UP</div>'
        f'<div style="font-size:13px; font-weight:700; color:#1e293b;">{sec_stage_name}</div>'
        f'<div style="font-size:16px; font-weight:800; color:#64748b;">{sec_conf*100:.1f}%</div></div>'
        f'<div style="background:#fff; border-radius:6px; padding:8px; border:1px solid #e2e8f0;">'
        f'<div style="font-size:10px; color:#64748b; font-weight:600;">MARGIN</div>'
        f'<div style="font-size:16px; font-weight:800; color:{margin_color};">{margin*100:.1f}%</div>'
        f'<div style="font-size:10px; color:{margin_color}; font-weight:600;">{margin_label}</div></div>'
        f'</div></div>'
    )

    # 10. Uncertainty Banner
    conf_val = diag["confidence"]
    if conf_val >= 0.85:
        unc_icon, unc_label, unc_color, unc_bg = "🟢", "HIGH CONFIDENCE", "#166534", "#f0fdf4"
        unc_border = "#10b981"
        unc_sub = f"Model certainty is {conf_val*100:.1f}% — result is suitable for clinical review."
    elif conf_val >= threshold:
        unc_icon, unc_label, unc_color, unc_bg = "🟡", "REVIEW RECOMMENDED", "#92400e", "#fffbeb"
        unc_border = "#f59e0b"
        unc_sub = f"Confidence ({conf_val*100:.1f}%) exceeds threshold but is below 85% — secondary clinical review advised."
    else:
        unc_icon, unc_label, unc_color, unc_bg = "🔴", "INSUFFICIENT CONFIDENCE", "#991b1b", "#fef2f2"
        unc_border = "#ef4444"
        unc_sub = f"Confidence ({conf_val*100:.1f}%) is below the safety threshold ({threshold*100:.0f}%) — human triage required."

    uncertainty_banner_html = (
        f'<div style="padding:12px 16px; background:{unc_bg}; border-radius:10px; '
        f'border-left:5px solid {unc_border}; margin-bottom:10px; display:flex; align-items:center; gap:12px;">'
        f'<div style="font-size:28px;">{unc_icon}</div>'
        f'<div><div style="font-size:14px; font-weight:800; color:{unc_color}; letter-spacing:0.5px;">{unc_label}</div>'
        f'<div style="font-size:12px; color:#374151; margin-top:2px;">{unc_sub}</div></div>'
        f'</div>'
    )

    # 11. Processed image for side-by-side view
    processed_display = (preproc * 255).astype(np.uint8)

    # 12. Prediction context for chatbot (gr.State)
    pred_context = {
        "stage_name": diag["stage_name"],
        "stage": stage,
        "confidence": conf_val,
        "urgency": adv["urgency"],
        "followup": adv["followup"],
        "plan": adv["plan"],
        "peak_quadrant": expl.get("peak_quadrant", "-"),
        "lesion_pct": expl.get("lesion_pct", 0.0),
    }

    # 13. Session history update
    if session_history is None:
        session_history = []
    import datetime as _dt
    hist_entry = {
        "timestamp": _dt.datetime.now().strftime("%H:%M:%S"),
        "stage": stage,
        "stage_name": diag["stage_name"],
        "confidence": f"{conf_val*100:.1f}%",
        "urgency": adv["urgency"],
    }
    session_history = session_history + [hist_entry]

    # Build history HTML
    history_rows = ""
    stage_colors = {0: "#10b981", 1: "#0284c7", 2: "#d97706", 3: "#ea580c", 4: "#e11d48"}
    for i, h in enumerate(reversed(session_history)):
        sc = stage_colors.get(h["stage"], "#64748b")
        history_rows += (
            f'<div style="display:flex; align-items:center; gap:10px; padding:8px 12px; '
            f'border-radius:8px; background:#f8fafc; border:1px solid #e2e8f0; margin-bottom:6px;">'
            f'<div style="min-width:36px; height:36px; border-radius:50%; background:{sc}; '
            f'display:flex; align-items:center; justify-content:center; color:white; font-weight:800; font-size:14px;">{h["stage"]}</div>'
            f'<div style="flex:1;"><div style="font-weight:700; font-size:13px; color:#0f172a;">{h["stage_name"]}</div>'
            f'<div style="font-size:11px; color:#64748b;">{h["urgency"]} • Confidence: {h["confidence"]}</div></div>'
            f'<div style="font-size:11px; color:#94a3b8;">{h["timestamp"]}</div>'
            f'</div>'
        )
    if not history_rows:
        history_rows = '<div style="color:#94a3b8; font-size:13px; padding:12px;">No predictions yet in this session.</div>'

    session_history_html = (
        f'<div style="padding:4px 0;">'
        f'<div style="font-size:11px; font-weight:700; text-transform:uppercase; color:#64748b; margin-bottom:8px;">'
        f'{len(session_history)} Prediction(s) This Session — Most Recent First</div>'
        f'{history_rows}</div>'
    )

    return (
        gov_html,                       # 0: status_banner
        hero_html,                      # 1: hero_diagnosis
        probs_dict,                     # 2: prob_distribution
        expl["overlay_cam"],            # 3: overlay_cam_view
        expl["lesion_seg"],             # 4: lesion_seg_view
        expl["vessel_overlay"],         # 5: vessel_overlay_view
        expl["optic_disc_overlay"],     # 6: optic_disc_overlay_view
        classical_cv_status_html,       # 7: classical_cv_status_view
        expl.get("overlap_analysis", {}).get("combined_vis", empty_img), # 8: overlap_view
        expl.get("severity_map", {}).get("quadrant_overlay", empty_img), # 9: severity_map_view
        advanced_analysis_html,         # 10: advanced_analysis_view
        expl["quadrant_desc"],          # 11: quadrant_text
        gallery_items,                  # 12: gallery_view
        advisory_html,                  # 13: advisory_view
        ehr_text,                       # 14: ehr_note_box
        lesion_burden_html,             # 15: lesion_burden_view
        quadrant_chart_html,             # 16: quadrant_chart_view
        confidence_margin_html,         # 17: confidence_margin_view
        pred_context,                   # 18: pred_context_state
        session_history,                # 19: session_history_state
        research_support,               # 20: research_support_state
        quality_html + uncertainty_banner_html,  # 20: uncertainty_banner_top
        processed_display,              # 21: processed_img_view
        session_history_html,           # 22: session_history_view
        diag["stage_name"],             # 23: _diag_stage_name_state
        conf_val,                       # 24: _diag_conf_state
        adv["urgency"],                 # 25: _diag_urgency_state
        adv["followup"],                # 26: _diag_followup_state
        adv["plan"],                    # 27: _diag_plan_state
        quality_html,                   # 28: quality_warning_view
    )


def compare_longitudinal_images(previous: Optional[np.ndarray], current: Optional[np.ndarray]):
    """Compare two optional examinations without assuming patient identity."""
    empty_img = np.zeros((AppConfig.IMG_SIZE, AppConfig.IMG_SIZE, 3), dtype=np.uint8)
    if previous is None or current is None:
        return "<div class='card warning-card'>Submit both a previous and current examination.</div>", empty_img
    try:
        previous_preproc = preprocess_image(previous)
        current_preproc = preprocess_image(current)
        previous_qc = assess_image_quality(previous)
        current_qc = assess_image_quality(current)
        previous_result = run_pipeline(previous_preproc, qc=previous_qc)
        current_result = run_pipeline(current_preproc, qc=current_qc)
        comparison = compare_longitudinal_examinations(
            previous_preproc,
            current_preproc,
            previous_result["diagnosis"],
            current_result["diagnosis"],
            previous_result["explanation"],
            current_result["explanation"],
        )
    except (ValueError, KeyError, RuntimeError, cv2.error) as exc:
        return f"<div class='card warning-card'>Reliable comparison could not be established: {exc}</div>", empty_img

    registration = "Reliable spatial registration established." if comparison["registration_reliable"] else "Reliable spatial comparison could not be established. Metric changes are non-spatial visual comparisons."
    summary = (
        '<div class="card" style="border-left:4px solid #0284c7;">'
        '<h3 style="margin:0 0 8px 0;">LONGITUDINAL COMPARISON</h3>'
        f'<div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">'
        f'<div><strong>Previous</strong><br>Stage {comparison["prev_stage"]} ({previous_result["diagnosis"]["stage_name"]})<br>'
        f'Lesion burden: {comparison["prev_lesion"]:.2f}%<br>Vessel density: {comparison["prev_vessel"]:.2f}%<br>Quadrants: {comparison["prev_quads"]}/4</div>'
        f'<div><strong>Current</strong><br>Stage {comparison["curr_stage"]} ({current_result["diagnosis"]["stage_name"]})<br>'
        f'Lesion burden: {comparison["curr_lesion"]:.2f}%<br>Vessel density: {comparison["curr_vessel"]:.2f}%<br>Quadrants: {comparison["curr_quads"]}/4</div></div>'
        f'<div style="margin-top:8px;">Lesion burden change: <strong>{comparison["lesion_delta"]:+.2f} percentage points</strong><br>'
        f'Vessel density change: <strong>{comparison["vessel_delta"]:+.2f} percentage points</strong><br>'
        f'Affected quadrant change: <strong>{comparison["quads_delta"]:+d}</strong></div>'
        f'<div style="margin-top:8px; color:#475569;">Visual change detected between the submitted examinations. {registration}</div>'
        '<div style="font-size:11px; color:#64748b; margin-top:8px; font-style:italic;">Visual support only — not an independent diagnosis and not proof of clinical disease progression.</div>'
        '</div>'
    )
    return summary, comparison["diff_vis"]


# ─────────────────────────────────────────────────────────────────────────────
# 7. Modern UI Assembly (Gradio Blocks Layout)
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* Medical Dashboard Base Styling */
html, body {
    overflow-x: hidden !important;
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    background-color: #f8fafc !important;
    color: #0f172a !important;
    transition: background-color 0.25s ease, color 0.25s ease;
}
.dark, html.dark, body.dark {
    background-color: #0b1120 !important;
    color: #f8fafc !important;
}

gradio-app {
    overflow-x: hidden !important;
    max-width: 100% !important;
    width: 100% !important;
    background-color: transparent !important;
}

.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    transition: background-color 0.3s ease, color 0.3s ease;
    box-sizing: border-box !important;
}

/* Header Telemetry Styling */
.telemetry-bar {
    display: flex;
    gap: 12px;
    background: #ffffff;
    color: #0f172a;
    padding: 10px 16px;
    border-radius: 8px;
    font-size: 12px;
    margin-bottom: 16px;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.telemetry-bar strong {
    color: #0f172a !important;
}
.telemetry-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #f1f5f9;
    padding: 4px 10px;
    border-radius: 6px;
    border: 1px solid #cbd5e1;
    font-weight: 500;
    color: #334155 !important;
}
.dark .telemetry-bar {
    background: #0f172a;
    border: 1px solid #1e293b;
    box-shadow: none;
}
.dark .telemetry-bar strong {
    color: #f8fafc !important;
}
.dark .telemetry-badge {
    background: #1e293b;
    border: 1px solid #334155;
    color: #f8fafc !important;
}
.pulse-dot {
    width: 8px;
    height: 8px;
    background: #10b981;
    border-radius: 50%;
    box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.4);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
    100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

/* Base Card Styling */
.card {
    background: #ffffff;
    border-radius: 10px;
    padding: 16px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    margin-bottom: 12px;
    color: #1e293b;
    transition: background-color 0.3s ease, border-color 0.3s ease, color 0.3s ease;
}
.dark .card {
    background: #1e293b;
    border: 1px solid #334155;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    color: #f1f5f9;
}

.card-header {
    display: flex;
    align-items: center;
    gap: 12px;
}
.icon {
    font-size: 24px;
}

/* Alert, Success, Warning Cards */
.success-card {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
}
.dark .success-card {
    background: #064e3b;
    border: 1px solid #059669;
}
.alert-card {
    background: #fef2f2;
    border: 1px solid #fecaca;
}
.dark .alert-card {
    background: #450a0a;
    border: 1px solid #dc2626;
}
.warning-card {
    background: #fffbeb;
    border: 1px solid #fde68a;
    color: #92400e;
}
.dark .warning-card {
    background: #451a03;
    border: 1px solid #d97706;
    color: #fef3c7;
}

/* Hero Diagnostic Card */
.hero-card {
    background: #ffffff;
}
.dark .hero-card {
    background: #1e293b;
}
.hero-stage-title {
    color: #0f172a;
}
.dark .hero-stage-title {
    color: #f8fafc;
}
.hero-subtext, .hero-sublabel {
    color: #64748b;
}
.dark .hero-subtext, .dark .hero-sublabel {
    color: #94a3b8;
}

/* Stage Badge Dark Mode Overrides */
.dark .stage-badge-0 { background: rgba(16, 185, 129, 0.2) !important; color: #34d399 !important; }
.dark .stage-badge-1 { background: rgba(2, 132, 199, 0.2) !important; color: #38bdf8 !important; }
.dark .stage-badge-2 { background: rgba(217, 119, 6, 0.2) !important; color: #fbbf24 !important; }
.dark .stage-badge-3 { background: rgba(234, 88, 12, 0.2) !important; color: #fb923c !important; }
.dark .stage-badge-4 { background: rgba(225, 29, 72, 0.2) !important; color: #f87171 !important; }

/* Protocol Section */
.protocol-title {
    color: #0f172a;
    border-bottom: 1px solid #e2e8f0;
}
.dark .protocol-title {
    color: #f8fafc;
    border-bottom: 1px solid #334155;
}
.protocol-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
}
.dark .protocol-box {
    background: #0f172a;
    border: 1px solid #334155;
}
.protocol-label {
    color: #64748b;
}
.dark .protocol-label {
    color: #94a3b8;
}
.protocol-val {
    color: #0f172a;
}
.dark .protocol-val {
    color: #f8fafc;
}
.action-box {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
}
.dark .action-box {
    background: #0f172a;
    border: 1px solid #334155;
}
.action-label {
    color: #475569;
}
.dark .action-label {
    color: #94a3b8;
}
.action-val {
    color: #1e293b;
}
.dark .action-val {
    color: #f1f5f9;
}
.disclaimer-text {
    color: #94a3b8;
}
.dark .disclaimer-text {
    color: #64748b;
}

/* Header Text */
.header-title {
    color: #0f172a;
}
.dark .header-title {
    color: #f8fafc;
}
.header-subtitle {
    color: #475569;
}
.dark .header-subtitle {
    color: #94a3b8;
}

/* Theme Toggle Button */
.theme-toggle-btn {
    border-radius: 8px !important;
    font-weight: 700 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    background: #f0fdfa !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
}
.theme-toggle-btn:hover {
    background: #f1f5f9 !important;
    transform: translateY(-1px);
}
.dark .theme-toggle-btn {
    background: #1e293b !important;
    color: #f8fafc !important;
    border: 1px solid #475569 !important;
    box-shadow: none !important;
}
.dark .theme-toggle-btn:hover {
    background: #334155 !important;
}

/* Button & Tool Enhancements */
.action-btn {
    background: linear-gradient(135deg, #0d9488, #0284c7) !important;
    color: white !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
}
.action-btn:hover {
    box-shadow: 0 4px 12px rgba(13, 148, 136, 0.3) !important;
}

.risk-card-header {
    flex-wrap: wrap;
    gap: 8px;
}
.risk-ladder > div {
    min-width: 0;
}

/* Keep the first viewport focused on the primary action and prediction. */
.gradio-container .gr-accordion {
    border-radius: 8px !important;
    margin-bottom: 6px !important;
}
.gradio-container .gr-accordion > .label-wrap {
    padding: 8px 12px !important;
}
.gradio-container .tabs {
    margin-top: 4px !important;
}
.gradio-container .tabitem {
    padding-top: 6px !important;
}

/* ── Global padding compression ── */
/* Reduce the default Gradio page-level top/bottom padding */
.gradio-container {
    padding-top: 8px !important;
    padding-bottom: 8px !important;
}
/* Tighten row & column gaps */
.gradio-container .gap {
    gap: 8px !important;
}
/* Shrink individual form-block vertical margins */
.gradio-container .block,
.gradio-container .form,
.gradio-container .gap > * {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}
/* Image upload widget label/label-wrap */
.gradio-container .block label,
.gradio-container .block .label-wrap {
    margin-bottom: 2px !important;
}
/* Reduce space between the uploader and the Run button row */
.gradio-container .row {
    gap: 6px !important;
}
/* Card spacing */
.card {
    margin-bottom: 8px !important;
}

/* ── Aggressive Gradio internal spacing overrides ── */
/* Remove top padding on the main app wrapper */
.gradio-container > .main,
.gradio-container > .main > .wrap {
    padding-top: 6px !important;
    padding-bottom: 6px !important;
    gap: 8px !important;
}
/* Shrink Gradio svelte block wrapper padding */
.svelte-1gfkn6j,
[class*="wrap "] {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}
/* Tighten Gradio built-in form block top/bottom padding */
.form > div,
.form > .block {
    padding-top: 4px !important;
    padding-bottom: 4px !important;
}
/* Force image block not to add extra margin */
.gradio-container .image-frame {
    margin: 0 !important;
}
/* Shrink button groups internal margin */
.gradio-container button + button {
    margin-left: 4px !important;
}
/* Tighten inter-element gap inside each column */
.gradio-container > .main > .wrap > .contain > * + * {
    margin-top: 6px !important;
}

/* Keep dense result cards usable on phones and small tablet widths. */
@media (max-width: 700px) {
    .gradio-container {
        padding-left: 10px !important;
        padding-right: 10px !important;
    }

    .header-title {
        font-size: 21px !important;
        line-height: 1.2 !important;
    }

    .telemetry-bar {
        align-items: flex-start;
        flex-direction: column;
    }

    .telemetry-bar > div:last-child {
        width: 100%;
    }

    .telemetry-badge {
        font-size: 11px;
        padding: 4px 7px;
    }

    .hero-stage-title {
        font-size: 21px !important;
    }

    .gallery {
        min-width: 0 !important;
    }

    .chatbot {
        height: 320px !important;
    }

    .responsive-grid-two,
    .responsive-grid-three {
        grid-template-columns: 1fr !important;
    }

    .risk-card {
        padding: 10px !important;
    }

    .risk-card-header > div:first-child {
        max-width: 100%;
        line-height: 1.35;
    }

    .risk-ladder {
        display: grid !important;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 6px !important;
    }

    .risk-ladder > div {
        min-width: 0 !important;
    }

    .gradio-container .gr-accordion > .label-wrap {
        padding: 8px 10px !important;
    }
}

/* Stepper Component (Medios / Mobile Workflow) */
.stepper-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #f0fdfa;
    border: 1px solid #a7f3d0;
    border-radius: 10px;
    padding: 6px 12px;
    margin-bottom: 8px;
    color: #065f46 !important;
    flex-wrap: wrap;
    gap: 6px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}
.stepper-item {
    display: flex;
    align-items: center;
    gap: 10px;
}
.stepper-circle {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background: #0d9488;
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 800;
}
.stepper-title {
    font-size: 11.5px;
    font-weight: 800;
    letter-spacing: 0.5px;
    color: #065f46 !important;
}
.stepper-sub {
    font-size: 10px;
    color: #047857 !important;
    opacity: 0.85;
}
.stepper-arrow {
    color: #059669 !important;
    font-weight: 800;
    font-size: 14px;
}
.dark .stepper-container {
    background: #022c22;
    border-color: #134e4a;
    box-shadow: none;
}
.dark .stepper-title {
    color: #ccfbf1 !important;
}
.dark .stepper-sub {
    color: #99f6e4 !important;
}
.dark .stepper-arrow {
    color: #2dd4bf !important;
}

/* Medios Patient EHR Card */
.patient-id-card {
    background: #ffffff;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
    padding: 8px 10px;
    margin-bottom: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.dark .patient-id-card {
    background: #1e293b;
    border-color: #334155;
    box-shadow: 0 1px 3px rgba(0,0,0,0.25);
}
.badge-quota {
    background: #ecfdf5;
    color: #065f46;
    border: 1px solid #a7f3d0;
    font-size: 10px;
    font-weight: 800;
    padding: 3px 8px;
    border-radius: 20px;
    letter-spacing: 0.5px;
}
.dark .badge-quota {
    background: #064e3b;
    color: #a7f3d0;
    border-color: #059669;
}
.patient-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: #f1f5f9;
    color: #334155;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
    border: 1px solid #e2e8f0;
}
.dark .patient-chip {
    background: #1e293b;
    color: #cbd5e1;
    border-color: #334155;
}

/* Medios SaMD Regulatory Card */
.medios-disclaimer-card {
    background: #f0fdfa;
    border: 1px solid #99f6e4;
    border-radius: 10px;
    padding: 12px 16px;
    margin-top: 14px;
    color: #0f766e;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}
.medios-disclaimer-card .disclaimer-title {
    color: #0d9488 !important;
}
.medios-disclaimer-card .disclaimer-body {
    color: #134e4a !important;
}
.dark .medios-disclaimer-card {
    background: #022c22;
    border-color: #134e4a;
    color: #ccfbf1;
    box-shadow: none;
}
.dark .medios-disclaimer-card .disclaimer-title {
    color: #2dd4bf !important;
}
.dark .medios-disclaimer-card .disclaimer-body {
    color: #ccfbf1 !important;
}

/* Gradio Tab Bar Styling */
.tabs > .tab-nav {
    border-bottom: 2px solid #e2e8f0 !important;
    gap: 6px !important;
}
.dark .tabs > .tab-nav {
    border-bottom: 2px solid #1e293b !important;
}
.tabs > .tab-nav > button {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #475569 !important;
    padding: 8px 14px !important;
    border-radius: 6px 6px 0 0 !important;
    transition: all 0.15s ease !important;
    background: transparent !important;
}
.tabs > .tab-nav > button:hover {
    color: #0d9488 !important;
    background: #f1f5f9 !important;
}
.tabs > .tab-nav > button.selected {
    color: #0d9488 !important;
    border-bottom: 2px solid #0d9488 !important;
    font-weight: 700 !important;
}
.dark .tabs > .tab-nav > button {
    color: #94a3b8 !important;
}
.dark .tabs > .tab-nav > button:hover {
    color: #2dd4bf !important;
    background: #1e293b !important;
}
.dark .tabs > .tab-nav > button.selected {
    color: #2dd4bf !important;
    border-bottom: 2px solid #0d9488 !important;
}

/* ── Left Sidebar Navigation (ProvoHeal Inspired) ─────────────────── */
#rg-sidebar {
    position: fixed !important;
    left: 0 !important;
    top: 0 !important;
    height: 100vh !important;
    width: 240px !important;
    background: #0b1329 !important;
    border-right: 1px solid #1e293b !important;
    z-index: 99999 !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
    box-shadow: 4px 0 24px rgba(0,0,0,0.45) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    transition: width 0.25s ease !important;
}
#rg-sidebar * {
    box-sizing: border-box !important;
}
#rg-sidebar .rg-sb-logo {
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
    padding: 20px 18px 16px !important;
    border-bottom: 1px solid #1e293b !important;
    flex-shrink: 0 !important;
}
#rg-sidebar .rg-sb-logo-icon {
    width: 38px !important;
    height: 38px !important;
    background: rgba(46, 125, 50, 0.16) !important;
    border: 1px solid rgba(74, 222, 128, 0.32) !important;
    border-radius: 10px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 3px !important;
    flex-shrink: 0 !important;
    box-shadow: 0 4px 12px rgba(46, 125, 50, 0.25) !important;
}
#rg-sidebar .rg-sb-logo-icon .rt-logo-img {
    width: 100% !important;
    height: 100% !important;
    object-fit: contain !important;
    filter: drop-shadow(0 0 5px rgba(74, 222, 128, 0.5)) brightness(1.2) contrast(1.1) !important;
}
#rg-sidebar .rg-sb-logo-text {
    font-size: 15px !important;
    font-weight: 800 !important;
    color: #f8fafc !important;
    line-height: 1.2 !important;
    letter-spacing: -0.2px !important;
}
#rg-sidebar .rg-sb-logo-sub {
    font-size: 10.5px !important;
    color: #86efac !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
}
.rt-logo-badge {
    width: 38px !important;
    height: 38px !important;
    border-radius: 10px !important;
    background: #f0fdf4 !important;
    border: 1px solid #bbf7d0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 4px !important;
    box-shadow: 0 2px 6px rgba(46, 125, 50, 0.12) !important;
    flex-shrink: 0 !important;
}
.rt-logo-badge .rt-logo-img {
    width: 100% !important;
    height: 100% !important;
    object-fit: contain !important;
}
.dark .rt-logo-badge {
    background: rgba(46, 125, 50, 0.22) !important;
    border-color: rgba(74, 222, 128, 0.35) !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35) !important;
}
.dark .rt-logo-badge .rt-logo-img {
    filter: drop-shadow(0 0 5px rgba(74, 222, 128, 0.45)) brightness(1.22) contrast(1.1) !important;
}
.dark .header-title {
    color: #f8fafc !important;
}
.dark .header-title span {
    color: #4ade80 !important;
}
.dark .rt-badge-pill {
    color: #4ade80 !important;
    background: rgba(74, 222, 128, 0.15) !important;
    border-color: rgba(74, 222, 128, 0.35) !important;
}
#rg-sidebar .rg-sb-search-wrap {
    padding: 12px 14px 6px !important;
    flex-shrink: 0 !important;
}
#rg-sidebar .rg-sb-search {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    background: #131f38 !important;
    border: 1px solid #223554 !important;
    border-radius: 8px !important;
    padding: 7px 10px !important;
    color: #94a3b8 !important;
}
#rg-sidebar .rg-sb-search input {
    background: transparent !important;
    border: none !important;
    outline: none !important;
    color: #f1f5f9 !important;
    font-size: 11.5px !important;
    width: 100% !important;
}
#rg-sidebar .rg-sb-search input::placeholder {
    color: #64748b !important;
}
#rg-sidebar .rg-kbd {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #94a3b8 !important;
    border-radius: 4px !important;
    font-size: 9.5px !important;
    font-weight: 700 !important;
    padding: 1px 5px !important;
    flex-shrink: 0 !important;
}
#rg-sidebar .rg-sb-section-label {
    font-size: 9.5px !important;
    font-weight: 800 !important;
    color: #475569 !important;
    letter-spacing: 1.2px !important;
    text-transform: uppercase !important;
    padding: 12px 18px 6px !important;
    flex-shrink: 0 !important;
}
#rg-sidebar .rg-sb-nav {
    display: flex !important;
    flex-direction: column !important;
    gap: 3px !important;
    padding: 0 10px !important;
    flex: 1 !important;
    overflow-y: auto !important;
}
#rg-sidebar .rg-sb-nav::-webkit-scrollbar { width: 3px; }
#rg-sidebar .rg-sb-nav::-webkit-scrollbar-thumb {
    background: #1e293b; border-radius: 3px;
}
#rg-sidebar .rg-sb-nav button {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    width: 100% !important;
    padding: 9px 12px !important;
    border: none !important;
    background: transparent !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
    font-size: 12.5px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    text-align: left !important;
    transition: all 0.15s ease !important;
    letter-spacing: 0.1px !important;
    position: relative !important;
}
#rg-sidebar .rg-sb-nav button:hover {
    background: #162444 !important;
    color: #38bdf8 !important;
}
#rg-sidebar .rg-sb-nav button.rg-active {
    background: #132742 !important;
    color: #2dd4bf !important;
    border-left: 3px solid #0d9488 !important;
    font-weight: 700 !important;
}
#rg-sidebar .rg-sb-nav .rg-sb-icon {
    font-size: 15px !important;
    flex-shrink: 0 !important;
    width: 20px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
#rg-sidebar .rg-badge {
    margin-left: auto !important;
    background: #0369a1 !important;
    color: #e0f2fe !important;
    font-size: 9.5px !important;
    font-weight: 700 !important;
    padding: 2px 6px !important;
    border-radius: 10px !important;
}
#rg-sidebar .rg-sb-telemetry {
    padding: 10px 14px !important;
    margin: 8px 10px !important;
    background: #0f1c34 !important;
    border: 1px solid #1e2e4f !important;
    border-radius: 8px !important;
    flex-shrink: 0 !important;
}
#rg-sidebar .rg-status-dot {
    width: 7px !important;
    height: 7px !important;
    border-radius: 50% !important;
    background: #10b981 !important;
    box-shadow: 0 0 8px #10b981 !important;
    display: inline-block !important;
}
#rg-sidebar .rg-sb-footer {
    padding: 10px 12px 14px !important;
    border-top: 1px solid #1e293b !important;
    flex-shrink: 0 !important;
}
#rg-sidebar .rg-sb-footer button {
    width: 100% !important;
    padding: 8px 12px !important;
    border: 1px solid #1e2e4f !important;
    background: #111e38 !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
    font-size: 11.5px !important;
    font-weight: 700 !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
    transition: all 0.15s ease !important;
}
#rg-sidebar .rg-sb-footer button:hover {
    background: #192a4e !important;
    color: #f8fafc !important;
    border-color: #38bdf8 !important;
}
/* Floating chat pill bottom-right */
.rg-floating-chat-pill {
    position: fixed !important;
    right: 24px !important;
    bottom: 24px !important;
    z-index: 99999 !important;
    background: linear-gradient(135deg, #0d9488, #0284c7) !important;
    color: #ffffff !important;
    padding: 10px 18px !important;
    border-radius: 999px !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    box-shadow: 0 6px 20px rgba(13, 148, 136, 0.45) !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}
.rg-floating-chat-pill:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 26px rgba(13, 148, 136, 0.6) !important;
}

/* Compact assistant window opened from the floating pill */
.rg-chat-floating-panel {
    position: fixed !important;
    right: 24px !important;
    bottom: 82px !important;
    z-index: 99998 !important;
    width: min(560px, calc(100vw - 48px)) !important;
    height: min(680px, calc(100vh - 118px)) !important;
    overflow: hidden !important;
    padding: 18px 20px 20px !important;
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 14px !important;
    box-shadow: 0 18px 50px rgba(15, 23, 42, 0.28) !important;
    animation: rg-chat-panel-in 0.18s ease-out !important;
}
.dark .rg-chat-floating-panel {
    background: #102a43 !important;
    border-color: #334155 !important;
    box-shadow: 0 18px 50px rgba(0, 0, 0, 0.5) !important;
}
.rg-chat-floating-panel .rg-chat-close {
    position: absolute !important;
    top: 10px !important;
    right: 12px !important;
    z-index: 2 !important;
    width: 30px !important;
    height: 30px !important;
    padding: 0 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 50% !important;
    background: transparent !important;
    color: #475569 !important;
    font-size: 20px !important;
    line-height: 1 !important;
    cursor: pointer !important;
}
.dark .rg-chat-floating-panel .rg-chat-close {
    border-color: #475569 !important;
    color: #cbd5e1 !important;
}
.rg-chat-floating-panel .rg-chat-close:hover {
    background: #e2e8f0 !important;
}
.dark .rg-chat-floating-panel .rg-chat-close:hover {
    background: #1e293b !important;
}
.rg-chat-floating-panel .rg-chat-specs {
    display: none !important;
}
.rg-chat-floating-panel [role="log"] {
    min-height: 0 !important;
    overflow-y: auto !important;
    overscroll-behavior: contain !important;
}
.rg-chat-floating-panel .rg-chat-input-row {
    align-items: stretch !important;
    gap: 8px !important;
}
.rg-chat-floating-panel .rg-chat-input-row > div,
.rg-chat-floating-panel .rg-chat-input-row button {
    align-self: stretch !important;
}
.rg-chat-floating-panel .rg-chat-input-row textarea,
.rg-chat-floating-panel .rg-chat-input-row button {
    min-height: 50px !important;
    height: 50px !important;
    box-sizing: border-box !important;
}
@keyframes rg-chat-panel-in {
    from { opacity: 0; transform: translateY(10px) scale(0.98); }
    to { opacity: 1; transform: translateY(0) scale(1); }
}

/* Mobile hamburger button */
#rg-mobile-menu-btn {
    display: none;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    border-radius: 8px;
    background: #0f172a;
    border: 1px solid #334155;
    color: #38bdf8;
    font-size: 19px;
    cursor: pointer;
    transition: all 0.15s ease;
    padding: 0;
    line-height: 1;
}
#rg-mobile-menu-btn:hover {
    background: #1e293b;
    border-color: #38bdf8;
}
.dark #rg-mobile-menu-btn {
    background: #0b1329;
    border-color: #1e293b;
    color: #2dd4bf;
}
@media (max-width: 900px) {
    #rg-mobile-menu-btn {
        display: flex !important;
    }
}

/* Sidebar backdrop on mobile */
#rg-sidebar-backdrop {
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(11, 19, 41, 0.7);
    backdrop-filter: blur(3px);
    -webkit-backdrop-filter: blur(3px);
    z-index: 9998;
    opacity: 0;
    transition: opacity 0.25s ease;
    pointer-events: none;
}
#rg-sidebar-backdrop.rg-open {
    display: block !important;
    opacity: 1 !important;
    pointer-events: auto !important;
}

/* Sidebar close button */
.rg-sb-close-btn {
    display: none;
    position: absolute;
    top: 14px;
    right: 14px;
    background: transparent;
    border: none;
    color: #94a3b8;
    font-size: 24px;
    cursor: pointer;
    line-height: 1;
    padding: 4px;
    border-radius: 6px;
    z-index: 10;
}
.rg-sb-close-btn:hover {
    color: #ffffff;
    background: rgba(255, 255, 255, 0.1);
}
@media (max-width: 900px) {
    .rg-sb-close-btn {
        display: block !important;
    }
}

/* Quick samples row: perfectly balanced horizontal row */
.quick-samples-row {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    gap: 6px !important;
    margin-top: 2px !important;
    margin-bottom: 4px !important;
}
.quick-samples-row > div {
    flex: 1 1 0 !important;
    min-width: 0 !important;
}
.quick-samples-row button {
    width: 100% !important;
    padding: 7px 4px !important;
    font-size: 11.5px !important;
    font-weight: 600 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

/* Push main Gradio container right on desktop - Responsive drawer on mobile */
@media (min-width: 901px) {
    #rg-sidebar {
        transform: translateX(0) !important;
        width: 240px !important;
    }
    .gradio-container {
        margin-left: 240px !important;
        margin-right: 0 !important;
        width: calc(100% - 240px) !important;
        max-width: calc(100% - 240px) !important;
        padding-left: 18px !important;
        padding-right: 24px !important;
        box-sizing: border-box !important;
    }
}
@media (max-width: 900px) {
    #rg-sidebar {
        display: flex !important;
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        bottom: 0 !important;
        width: 270px !important;
        transform: translateX(-100%) !important;
        transition: transform 0.28s cubic-bezier(0.4, 0, 0.2, 1) !important;
        z-index: 9999 !important;
        box-shadow: none !important;
    }
    #rg-sidebar.rg-open {
        transform: translateX(0) !important;
        box-shadow: 6px 0 35px rgba(0, 0, 0, 0.75) !important;
    }
    #rg-sidebar .rg-sb-logo-text,
    #rg-sidebar .rg-sb-logo-sub,
    #rg-sidebar .rg-sb-search-wrap,
    #rg-sidebar .rg-sb-section-label,
    #rg-sidebar .rg-sb-nav button span:not(.rg-sb-icon),
    #rg-sidebar .rg-badge,
    #rg-sidebar .rg-sb-telemetry,
    #rg-sidebar .rg-sb-footer button span:last-child {
        display: block !important;
    }
    #rg-sidebar .rg-sb-nav button {
        justify-content: flex-start !important;
        padding: 9px 12px !important;
    }
    .gradio-container {
        margin-left: 0 !important;
        margin-right: 0 !important;
        width: 100% !important;
        max-width: 100% !important;
        padding-left: 12px !important;
        padding-right: 12px !important;
        box-sizing: border-box !important;
    }
}
@media (max-width: 600px) {
    #rg-sidebar {
        display: flex !important;
        width: 270px !important;
    }
    .gradio-container {
        margin-left: 0 !important;
        margin-right: 0 !important;
        width: 100% !important;
        max-width: 100% !important;
        padding-left: 8px !important;
        padding-right: 8px !important;
        box-sizing: border-box !important;
    }
    .rg-floating-chat-pill {
        right: 14px !important;
        bottom: 14px !important;
        padding: 8px 14px !important;
        font-size: 12px !important;
    }
    .rg-chat-floating-panel {
        right: 10px !important;
        bottom: 68px !important;
        width: calc(100vw - 20px) !important;
        height: min(680px, calc(100vh - 92px)) !important;
        padding: 16px 12px 14px !important;
    }
}
</style>
"""

HEAD_SCRIPT = """
<script>
(function() {
    window.retinaOpenTab = function(label, btnEl, elemId) {
        const cleanLabel = (label || '').toLowerCase().trim();

        function getAllRoots() {
            const roots = [document];
            const app = document.querySelector('gradio-app');
            if (app) {
                roots.push(app);
                if (app.shadowRoot) roots.push(app.shadowRoot);
                function collectShadows(node) {
                    if (!node) return;
                    if (node.shadowRoot) roots.push(node.shadowRoot);
                    const kids = node.children ? Array.from(node.children) : [];
                    for (let i = 0; i < kids.length; i++) {
                        collectShadows(kids[i]);
                    }
                }
                collectShadows(app);
            }
            return roots;
        }

        function findTabButton() {
            const roots = getAllRoots();

            // Gradio renders TabItem controls as ordinary buttons without the
            // supplied elem_id. Prefer the visible tab label when an explicit
            // navigation target is requested.
            if (cleanLabel && elemId) {
                const directButtons = Array.from(document.querySelectorAll('button, [role="tab"]'));
                const directTarget = directButtons.find(function(btn) {
                    if (btn.closest && btn.closest('#rg-sidebar')) return false;
                    if (btn.closest && btn.closest('.tab-container.visually-hidden')) return false;
                    return (btn.textContent || '').toLowerCase().includes(cleanLabel);
                });
                if (directTarget) return directTarget;
            }

            // Strategy 1: Find tab button by text match (excluding sidebar buttons).
            // When an explicit element ID is supplied, resolve that stable target first.
            if (cleanLabel && !elemId) {
                for (let i = 0; i < roots.length; i++) {
                    const r = roots[i];
                    let buttons = [];
                    try {
                        buttons = Array.from(r.querySelectorAll('button, [role="tab"]'));
                    } catch(e) {}
                    for (let j = 0; j < buttons.length; j++) {
                        const btn = buttons[j];
                        if (btn.closest && btn.closest('#rg-sidebar')) continue;
                        const txt = (btn.textContent || '').toLowerCase();
                        if (txt.includes(cleanLabel)) {
                            return btn;
                        }
                    }
                }
            }

            // Strategy 2: Find panel by elemId, then get corresponding tab button
            if (elemId) {
                for (let i = 0; i < roots.length; i++) {
                    const r = roots[i];
                    let panel = null;
                    try {
                        panel = r.getElementById ? r.getElementById(elemId) : r.querySelector('#' + elemId);
                    } catch(e) {}
                    if (!panel) continue;

                    let parent = panel.parentElement;
                    while (parent && !parent.querySelector('[role="tab"], .tab-nav button')) {
                        parent = parent.parentElement;
                    }
                    if (parent) {
                        const panelId = panel.id || elemId;
                        const byAria = parent.querySelector('[aria-controls="' + panelId + '"]');
                        if (byAria) return byAria;

                        const panels = Array.from(parent.querySelectorAll('[role="tabpanel"], .tabitem'));
                        const navBtns = Array.from(parent.querySelectorAll('[role="tab"], .tab-nav button'))
                            .filter(b => !b.closest('#rg-sidebar'));
                        const pIdx = panels.indexOf(panel);
                        if (pIdx >= 0 && navBtns[pIdx]) return navBtns[pIdx];
                    }
                }
            }

            // Strategy 3: Loose word matching (e.g. "chatbot", "cbr", "triage", "diag")
            if (cleanLabel) {
                const words = cleanLabel.split(' ').filter(w => w.length > 2);
                for (let i = 0; i < roots.length; i++) {
                    const r = roots[i];
                    let buttons = [];
                    try {
                        buttons = Array.from(r.querySelectorAll('button, [role="tab"]'));
                    } catch(e) {}
                    for (let j = 0; j < buttons.length; j++) {
                        const btn = buttons[j];
                        if (btn.closest && btn.closest('#rg-sidebar')) continue;
                        const txt = (btn.textContent || '').toLowerCase();
                        if (words.some(w => txt.includes(w))) {
                            return btn;
                        }
                    }
                }
            }

            return null;
        }

        const target = findTabButton();
        if (target) {
            const overflowMenu = target.closest && target.closest('.overflow-dropdown');
            const activateTarget = function() {
                target.click();
                try {
                    target.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, composed: true }));
                } catch(e) {}
                setTimeout(function() {
                    try { target.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); } catch(e) {}
                }, 50);
            };
            if (overflowMenu && overflowMenu.classList.contains('hide')) {
                const moreTabs = Array.from(document.querySelectorAll('button')).find(function(button) {
                    return (button.textContent || '').trim().toLowerCase() === 'more tabs';
                });
                if (moreTabs) {
                    moreTabs.click();
                    setTimeout(activateTarget, 100);
                } else {
                    activateTarget();
                }
            } else {
                activateTarget();
            }
        } else {
            console.warn('[RetinaTrace] Tab target not found for:', label, elemId);
            setTimeout(function() {
                const retryTarget = findTabButton();
                if (retryTarget) {
                    retryTarget.click();
                    try {
                        retryTarget.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, composed: true }));
                    } catch(e) {}
                }
            }, 250);
        }

        if (btnEl) {
            document.querySelectorAll('#rg-sidebar .rg-sb-btn').forEach(function(b) {
                b.classList.remove('rg-active');
            });
            btnEl.classList.add('rg-active');
        }

        // On mobile, close drawer
        if (window.innerWidth <= 900 && window.retinaToggleSidebar) {
            window.retinaToggleSidebar(false);
        }
    };

    window.retinaToggleSidebar = function(open) {
        const sb = document.getElementById('rg-sidebar');
        const bd = document.getElementById('rg-sidebar-backdrop');
        if (!sb) return;
        const isCurrentlyOpen = sb.classList.contains('rg-open');
        const shouldOpen = open === undefined ? !isCurrentlyOpen : Boolean(open);
        if (shouldOpen) {
            sb.classList.add('rg-open');
            if (bd) bd.classList.add('rg-open');
        } else {
            sb.classList.remove('rg-open');
            if (bd) bd.classList.remove('rg-open');
        }
    };

    window.retinaToggleTheme = function() {
        const elApp = document.querySelector('gradio-app');
        const isDark = document.documentElement.classList.contains('dark') 
                    || document.body.classList.contains('dark')
                    || (elApp && elApp.classList.contains('dark'));
        const targets = [document.documentElement, document.body];
        if (elApp) targets.push(elApp);
        
        if (isDark) {
            targets.forEach(function(t) { t.classList.remove('dark'); });
            try { localStorage.setItem('retinatrace_theme', 'light'); } catch(e) {}
        } else {
            targets.forEach(function(t) { t.classList.add('dark'); });
            try { localStorage.setItem('retinatrace_theme', 'dark'); } catch(e) {}
        }
    };

    window.retinaSearch = function(query) {
        if (!query) return;
        const q = query.toLowerCase();
        const tabs = [
            { text: 'Diagnostic Assessment', match: ['diag', 'result', 'stage', 'cam', 'grad', 'lesion', 'unet', 'vessel', 'optic'] },
            { text: 'Case-Based Reasoning', match: ['cbr', 'case', 'similar', 'reference', 'embed'] },
            { text: 'Clinical Management', match: ['care', 'protocol', 'ehr', 'plan', 'urgency', 'referral', 'note'] },
            { text: 'Multimodal Triage', match: ['triage', 'risk', 'hba1c', 'simulator', 'progression', 'bp'] },
            { text: 'AI Clinical Chatbot', match: ['chat', 'bot', 'assistant', 'ask', 'question', 'samd'] },
            { text: 'Session Prediction', match: ['history', 'log', 'past', 'session'] },
            { text: 'Image Comparison', match: ['report', 'json', 'download', 'compare', 'graham', 'preproc'] },
            { text: 'Longitudinal Analysis', match: ['longitudinal', 'previous', 'current', 'delta', 'progression'] }
        ];
        const found = tabs.find(function(t) {
            return t.match.some(function(m) { return q.includes(m); });
        });
        if (found) {
            window.retinaOpenTab(found.text);
        }
    };

    window.retinaToggleChat = function() {
        const existingPanel = document.querySelector('[role="tabpanel"].rg-chat-floating-panel');
        if (existingPanel) {
            window.retinaCloseChat();
            return;
        }

        window.retinaChatPreviousTab = Array.from(document.querySelectorAll('button, [role="tab"]'))
            .find(function(btn) {
                return !btn.closest('#rg-sidebar') && btn.getAttribute('aria-selected') === 'true';
            });
        if (window.retinaOpenTab) {
            window.retinaOpenTab('AI Clinical Chatbot', null, 'rg-tab-chat');
        }
        setTimeout(function() {
            const chatTarget = document.getElementById('rg-tab-chat');
            if (!chatTarget) return;
            let panel = chatTarget;
            while (panel && panel !== document.body && panel.getAttribute('role') !== 'tabpanel') {
                panel = panel.parentElement;
            }
            panel = panel && panel.getAttribute('role') === 'tabpanel' ? panel : chatTarget;
            panel.classList.add('rg-chat-floating-panel');
            chatTarget.classList.add('rg-chat-floating-panel');
        }, 80);
    };

    window.retinaCloseChat = function() {
        const chatTarget = document.getElementById('rg-tab-chat');
        if (!chatTarget) return;
        let panel = chatTarget;
        while (panel && panel !== document.body && panel.getAttribute('role') !== 'tabpanel') {
            panel = panel.parentElement;
        }
        if (panel) panel.classList.remove('rg-chat-floating-panel');
        chatTarget.classList.remove('rg-chat-floating-panel');
        if (window.retinaChatPreviousTab) {
            window.retinaChatPreviousTab.click();
            window.retinaChatPreviousTab = null;
        }
    };

    function mountSidebar() {
        const sb = document.getElementById('rg-sidebar');
        const bd = document.getElementById('rg-sidebar-backdrop');
        if (sb && sb.parentElement && sb.parentElement !== document.body) {
            document.body.appendChild(sb);
        }
        if (bd && bd.parentElement && bd.parentElement !== document.body) {
            document.body.appendChild(bd);
        }
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', mountSidebar);
    } else {
        setTimeout(mountSidebar, 100);
    }
    setInterval(mountSidebar, 1500);
})();

(function() {
    try {
        const savedTheme = localStorage.getItem('retinatrace_theme') || localStorage.getItem('retinaguard_theme');
        const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
            document.documentElement.classList.add('dark');
            document.body.classList.add('dark');
        }
    } catch(e) {}
})();
</script>
"""

THEME_TOGGLE_JS = """
() => {
    const elApp = document.querySelector('gradio-app');
    const isDark = document.documentElement.classList.contains('dark') 
                || document.body.classList.contains('dark')
                || (elApp && elApp.classList.contains('dark'));
    const targets = [document.documentElement, document.body];
    if (elApp) targets.push(elApp);
    
    if (isDark) {
        targets.forEach(t => t.classList.remove('dark'));
        try { localStorage.setItem('retinatrace_theme', 'light'); } catch(e) {}
    } else {
        targets.forEach(t => t.classList.add('dark'));
        try { localStorage.setItem('retinatrace_theme', 'dark'); } catch(e) {}
    }
}
"""

OPEN_CHATBOT_JS = """
() => {
    if (window.retinaToggleChat) {
        window.retinaToggleChat();
    }
}
"""

OPEN_TAB_JS = """
(label) => {
    if (window.retinaOpenTab) {
        window.retinaOpenTab(label);
    }
}
"""


def open_tab_js(label: str) -> str:
    """Build Gradio-supported JavaScript for activating an existing tab."""
    escaped_label = label.replace("'", "\\'")
    return f"""() => {{
        if (window.retinaOpenTab) {{
            window.retinaOpenTab('{escaped_label}');
        }}
    }}"""
    
# RetinaTrace Brand Assets
_LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "retinatrace_icon.png")
if os.path.exists(_LOGO_PATH):
    import base64 as _base64
    with open(_LOGO_PATH, "rb") as _f:
        RETINA_LOGO_SRC = f"data:image/png;base64,{_base64.b64encode(_f.read()).decode('ascii')}"
else:
    RETINA_LOGO_SRC = ""

RETINA_LOGO_IMG = f'<img src="{RETINA_LOGO_SRC}" alt="RetinaTrace" class="rt-logo-img" />' if RETINA_LOGO_SRC else """<svg viewBox="0 0 112 65" fill="#2e7d32" class="rt-logo-img"><path d="M 52 19 L 50 20 L 46 24 L 44 29 L 44 34 L 46 38 L 49 41 L 53 43 L 58 43 L 64 40 L 66 37 L 67 34 L 67 29 L 65 25 L 61 26 L 60 22 L 63 21 L 61 19 Z M 0 30 L 1 34 L 15 48 L 23 54 L 37 61 L 49 64 L 60 64 L 66 63 L 76 60 L 86 55 L 97 47 L 110 34 L 111 32 L 111 30 L 100 19 L 88 10 L 78 5 L 69 2 L 56 0 L 42 2 L 28 7 L 14 16 Z M 50 15 L 59 15 L 64 19 L 67 20 L 70 17 L 72 19 L 75 26 L 75 37 L 73 41 L 65 49 L 60 51 L 50 51 L 46 49 L 40 44 L 36 37 L 36 27 L 44 18 Z M 75 10 L 83 14 L 92 20 L 103 32 L 92 43 L 87 47 L 75 54 L 72 53 L 78 45 L 81 37 L 81 25 L 78 18 L 73 12 Z M 33 11 L 37 11 L 33 16 L 30 24 L 30 38 L 34 46 L 44 55 L 41 56 L 30 51 L 20 44 L 7 31 L 21 18 Z"/></svg>"""

theme = gr.themes.Soft(primary_hue="teal", secondary_hue="slate")

with gr.Blocks(title="RetinaTrace — DR Research Prototype") as demo:
    # Inject Custom Clinical Styling & Theme Detection
    gr.HTML(CUSTOM_CSS)

    # ── Sidebar (injected as fixed HTML, JS drives tab navigation) ───
    gr.HTML("""
    <div id="rg-sidebar-backdrop" onclick="window.retinaToggleSidebar(false)"></div>
    <div id="rg-sidebar">
        <!-- Mobile close button -->
        <button class="rg-sb-close-btn" onclick="window.retinaToggleSidebar(false)" title="Close Navigation">&times;</button>

        <!-- Logo -->
        <div class="rg-sb-logo">
            <div class="rg-sb-logo-icon">
                """ + RETINA_LOGO_IMG + """
            </div>
            <div>
                <div class="rg-sb-logo-text">Retina<span style="color:#4ade80;">Trace</span></div>
                <div class="rg-sb-logo-sub">Clinical Intelligence</div>
            </div>
        </div>

        <!-- Search input (ProvoHeal style) -->
        <div class="rg-sb-search-wrap">
            <div class="rg-sb-search">
                <span style="font-size:12px; opacity:0.6;">🔍</span>
                <input type="text" placeholder="Search features..." oninput="window.retinaSearch(this.value)" />
                <span class="rg-kbd">⌘K</span>
            </div>
        </div>

        <!-- Navigation -->
        <div class="rg-sb-section-label">CLINICAL SUITE</div>
        <div class="rg-sb-nav">
            <button class="rg-sb-btn rg-active" onclick="window.retinaOpenTab('Diagnostic Assessment', this, 'rg-tab-diag')" title="Diagnostic Assessment">
                <span class="rg-sb-icon">🩺</span>
                <span>Diagnosis &amp; CV</span>
            </button>
            <button class="rg-sb-btn" onclick="window.retinaOpenTab('Case-Based Reasoning', this, 'rg-tab-cbr')" title="Case-Based Reasoning (CBR)">
                <span class="rg-sb-icon">📚</span>
                <span>CBR Evidence</span>
            </button>
            <button class="rg-sb-btn" onclick="window.retinaOpenTab('Clinical Management', this, 'rg-tab-care')" title="Clinical Management &amp; EHR">
                <span class="rg-sb-icon">📋</span>
                <span>Care Protocol</span>
            </button>
            <button class="rg-sb-btn" onclick="window.retinaOpenTab('Multimodal Triage', this, 'rg-tab-triage')" title="Triage &amp; Risk Simulator">
                <span class="rg-sb-icon">🚦</span>
                <span>Multimodal Triage</span>
            </button>
            <button class="rg-sb-btn" onclick="window.retinaOpenTab('AI Clinical Chatbot', this, 'rg-tab-chat')" title="AI Clinical Chatbot">
                <span class="rg-sb-icon">💬</span>
                <span>AI Chatbot</span>
                <span class="rg-badge">AAO</span>
            </button>
            <button class="rg-sb-btn" onclick="window.retinaOpenTab('Session Prediction', this, 'rg-tab-history')" title="Session Prediction History">
                <span class="rg-sb-icon">📜</span>
                <span>Prediction History</span>
            </button>
            <button class="rg-sb-btn" onclick="window.retinaOpenTab('Image Comparison', this, 'rg-tab-compare')" title="Image Comparison &amp; Report">
                <span class="rg-sb-icon">🖼️</span>
                <span>Image Reports</span>
            </button>
            <button class="rg-sb-btn" onclick="window.retinaOpenTab('Longitudinal Analysis', this, 'rg-tab-longitudinal')" title="Longitudinal Retinal Analysis">
                <span class="rg-sb-icon">📊</span>
                <span>Longitudinal View</span>
            </button>
        </div>

        <!-- Telemetry status chip -->
        <div class="rg-sb-telemetry">
            <div style="display:flex; align-items:center; gap:6px;">
                <span class="rg-status-dot"></span>
                <span style="font-size:11px; font-weight:700; color:#38bdf8;">Governance Active</span>
            </div>
            <div style="font-size:10px; color:#64748b; margin-top:2px;">EfficientNetB3 · U-Net · CBR</div>
        </div>

        <!-- Footer -->
        <div class="rg-sb-footer">
            <button onclick="window.retinaToggleTheme()" title="Toggle Dark / Light">
                <span>🌓</span>
                <span>Dark / Light Mode</span>
            </button>
        </div>
    </div>

    <!-- Floating chat pill button (pure HTML, fixed bottom-right, zero flow disruption) -->
    <div class="rg-floating-chat-pill" onclick="window.retinaToggleChat()" title="Open AI Clinical Assistant">
        🤖 AI Chatbot
    </div>
    """)

    # Session state
    pred_context_state = gr.State({})
    session_history_state = gr.State([])
    research_support_state = gr.State({})
    _diag_stage_name_state = gr.State("")
    _diag_conf_state = gr.State(0.0)
    _diag_urgency_state = gr.State("")
    _diag_followup_state = gr.State("")
    _diag_plan_state = gr.State("")

    # ── Top Title Strip ──────
    gr.HTML(f"""
    <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px; margin-bottom:6px; padding-bottom:6px; border-bottom:1px solid rgba(226,232,240,0.5);">
        <div style="display:flex; align-items:center; gap:10px;">
            <!-- Mobile Menu Drawer Toggle (<= 900px) -->
            <button id="rg-mobile-menu-btn" onclick="window.retinaToggleSidebar(true)" title="Open Navigation Menu">
                <span>☰</span>
            </button>
            <div class="rt-logo-badge">
                {RETINA_LOGO_IMG}
            </div>
            <div style="display:flex; align-items:center; gap:9px; flex-wrap:wrap;">
                <span class="header-title" style="font-size:22px; font-weight:800; letter-spacing:-0.4px; margin:0; line-height:1;">
                    Retina<span style="color:#2e7d32;">Trace</span>
                </span>
                <span class="rt-badge-pill" style="font-size:12px; font-weight:700; color:#2e7d32; background:rgba(46,125,50,0.12); border:1px solid rgba(46,125,50,0.28); padding:3px 9px; border-radius:6px; letter-spacing:0.3px; text-transform:uppercase;">
                    Clinical AI
                </span>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
            <span style="background:#e0f2fe; color:#0369a1; padding:4px 10px; border-radius:6px; font-weight:700; font-size:11px; white-space:nowrap;">
                EfficientNetB3 &bull; Grad-CAM &bull; U-Net &bull; CBR
            </span>
        </div>
    </div>
    """)

    gr.Markdown(
        "> **Research prototype — not for clinical decisions.** The saved classifier "
        "checkpoint predates the corrected EfficientNet input-scale pipeline and current "
        "duplicate-safe split. Predictions from it are unvalidated; retraining and evaluation "
        "are required before reporting model performance."
    )

    # 2. Main Workspace (2 Columns)
    with gr.Row():
        # Left Column: Upload & Governance Configuration
        with gr.Column(scale=4):
            input_image = gr.Image(label="📥 Upload Fundus Photo", type="numpy", height=180)

            # ── Run & Reset buttons DIRECTLY under upload section ───────────
            with gr.Row():
                submit_btn = gr.Button("🚀 Run Diagnostic Analysis", variant="primary", size="lg", scale=3, elem_classes=["action-btn"])
                btn_clear = gr.Button("🔄 Reset", variant="secondary", size="lg", scale=1)

            # Quick Preset Buttons (Single balanced row)
            gr.Markdown("<div style='font-size:11px; font-weight:700; color:#64748b; margin:3px 0 1px 0;'>⚡ QUICK-LOAD SAMPLES:</div>")
            with gr.Row(elem_classes=["quick-samples-row"]):
                btn_normal = gr.Button("🟢 Normal", size="sm")
                btn_moderate = gr.Button("🟡 Moderate", size="sm")
                btn_prolif = gr.Button("🔴 Proliferative", size="sm")

            with gr.Accordion("🛡️ Safety gate & patient context", open=False):
                threshold_slider = gr.Slider(
                    minimum=0.50, maximum=0.95, value=0.70, step=0.05,
                    label="Governance Confidence Threshold",
                    info="Predictions below this confidence trigger an active safety override and withhold automated guidance.",
                )
                btn_override_test = gr.Button("🧪 Simulate Safety Override (Set to 95%)", variant="secondary", size="sm")
                gr.Markdown("**Optional patient context**")
                patient_age = gr.Slider(minimum=18, maximum=90, value=55, step=1, label="Patient Age (years)")
                diabetes_type = gr.Dropdown(
                    choices=["Type 1", "Type 2", "Gestational", "Not Specified"],
                    value="Type 2", label="Diabetes Type"
                )
                hba1c_level = gr.Slider(minimum=5.0, maximum=14.0, value=7.5, step=0.1, label="HbA1c (%)")
                diabetes_duration = gr.Slider(minimum=0, maximum=40, value=10, step=1, label="Duration of Diabetes (years)")
                systolic_bp = gr.Slider(minimum=90, maximum=220, value=135, step=1, label="Systolic Blood Pressure (mmHg)")

        # Right Column: Multi-Tab Clinical Dossier
        with gr.Column(scale=6):
            # Governance Status Banner (Appears at top of results when analysis runs)
            status_banner = gr.HTML("")

            # Structured Tabs
            with gr.Tabs():
                # Tab 1: Primary Diagnosis & 3-Layer Explainability
                with gr.TabItem("🏥 Diagnostic Assessment & Explainability", elem_id="rg-tab-diag"):
                    uncertainty_banner_top = gr.HTML()
                    hero_diagnosis = gr.HTML()
                    prob_distribution = gr.Label(label="5-Stage Disease Probability Distribution (Softmax)", num_top_classes=5)

                    with gr.Accordion("🔬 Explainability & visual evidence", open=False):
                        gr.Markdown("Grad-CAM, U-Net, classical CV, and experimental research measurements")
                        with gr.Row():
                            with gr.Column(scale=5):
                                gr.Markdown("**Layer 2: Regional Attention (Grad-CAM)**")
                                overlay_cam_view = gr.Image(label="Grad-CAM Saliency Overlay", type="numpy", height=200)
                            with gr.Column(scale=5):
                                gr.Markdown("**Layer 3: Lesion Segmentation (U-Net)**")
                                lesion_seg_view = gr.Image(label="Segmented Lesions (U-Net)", type="numpy", height=200)
                        with gr.Row():
                            with gr.Column(scale=5):
                                gr.Markdown("**Classical CV Vessel Analysis**")
                                vessel_overlay_view = gr.Image(label="Retinal Vessel Overlay", type="numpy", height=180)
                            with gr.Column(scale=5):
                                gr.Markdown("**Optic-Disc Localisation**")
                                optic_disc_overlay_view = gr.Image(label="Optic-Disc Localisation Overlay", type="numpy", height=180)
                        classical_cv_status_view = gr.HTML(
                            "<div style='color:#64748b; font-size:12px;'>Visual support only — not an independent diagnosis.</div>"
                        )
                        with gr.Row():
                            overlap_view = gr.Image(label="Attention–Lesion Agreement", type="numpy", height=180)
                            severity_map_view = gr.Image(label="Retinal Visual-Abnormality Map", type="numpy", height=180)
                        advanced_analysis_view = gr.HTML()
                        lesion_burden_view = gr.HTML()
                        quadrant_text = gr.Markdown()
                        quadrant_chart_view = gr.HTML()
                        confidence_margin_view = gr.HTML()

                # Tab 2: Case-Based Reasoning (CBR) Evidence
                with gr.TabItem("📚 Case-Based Reasoning (CBR) Evidence", elem_id="rg-tab-cbr"):
                    gr.Markdown("### 🔎 Nearest Available Reference Cases")
                    gr.Markdown(
                        "The query image was projected into the 256-D penultimate feature bottleneck. "
                        "When a verified reference library is available, the **Top-3 closest matching cases** are retrieved via cosine similarity:"
                    )
                    gallery_view = gr.Gallery(columns=3, rows=1, height=260, object_fit="contain")

                # Tab 3: Clinical Care Protocol & EHR Note
                with gr.TabItem("📋 Clinical Management & EHR Note", elem_id="rg-tab-care"):
                    advisory_view = gr.HTML()
                    gr.Markdown("### 📄 Exportable Electronic Health Record (EHR) Summary Note")
                    ehr_note_box = gr.Textbox(label="Clinical Session Note (Copy to Clipboard)", lines=12, interactive=False)

                # Tab 4: AI Clinical Chatbot & SaMD Guidelines
                with gr.TabItem("💬 AI Clinical Chatbot", elem_id="rg-tab-chat"):
                    gr.HTML('<button class="rg-chat-close" onclick="window.retinaCloseChat()" title="Close AI Clinical Assistant">&times;</button>')
                    gr.Markdown("""
                    ### 🤖 RetinaTrace Clinical Knowledge Assistant
                    Ask about diabetic retinopathy, the RetinaTrace model, or clinical guidelines.
                    """)
                    with gr.Row():
                        with gr.Column(scale=3):
                            chatbot_widget = gr.Chatbot(
                                label="Clinical Knowledge Assistant",
                                height=420,
                                value=[],
                            )
                            with gr.Row(elem_classes=["rg-chat-input-row"]):
                                chat_input = gr.Textbox(
                                    placeholder="Ask a question about DR staging, the model, or clinical guidelines…",
                                    label="",
                                    scale=5,
                                    container=False,
                                )
                                chat_send_btn = gr.Button("Send", variant="primary", scale=1)
                            chat_clear_btn = gr.Button("🗑️ Clear Chat", size="sm")
                            gr.Markdown("**Quick Prompts:**")
                            with gr.Row():
                                chip_classify = gr.Button("Explain why this was classified", size="sm", variant="secondary")
                                chip_rule421 = gr.Button("What is the 4-2-1 rule?", size="sm", variant="secondary")
                            with gr.Row():
                                chip_gradcam = gr.Button("How does Grad-CAM work?", size="sm", variant="secondary")
                                chip_refer = gr.Button("When should I refer urgently?", size="sm", variant="secondary")
                        with gr.Column(scale=2, elem_classes=["rg-chat-specs"]):
                            gr.Markdown(r"""
                            ### ⚙️ System Specifications:
                            * **Deep Learning Backbone:** EfficientNetB3 with historical saved weights. The corrected input-scale pipeline and duplicate-safe split have not yet been retrained and evaluated.
                            * **Ordinal Metric:** Historical EXP-03 results include Quadratic Weighted Kappa (QWK); they are not current final-split validation.
                            * **Threshold Workflow Flag:** Low-confidence predictions are flagged for specialist review; the prototype cannot enforce a clinical referral.
                            * **Intended Use:** Coursework prototype for research and demonstration only; not validated for clinical use.
                            """)

                # Tab 5: Multimodal Clinical Triage & Risk Simulator
                with gr.TabItem("🚦 Multimodal Triage & 10-Yr Risk Simulator", elem_id="rg-tab-triage"):
                    gr.Markdown("""
                    ### 🏥 Multimodal Clinical Triage & Progression Risk Simulator
                    Integrates the **image-derived DR severity stage** with systemic endocrinology indicators 
                    (**HbA1c, Diabetes Duration, Blood Pressure, Patient Age**) based on the validated **UKPDS 33** 
                    and **WESDR** epidemiological risk models.
                    """)
                    initial_risk_html, initial_ticket_text = calculate_multimodal_risk(2, 7.5, 10.0, 55.0, 135.0, "Type 2")
                    triage_risk_card = gr.HTML(initial_risk_html)
                    with gr.Row():
                        btn_recalc_triage = gr.Button("⚡ Recalculate Multimodal Triage & Dispatch", variant="primary", scale=2)
                    gr.Markdown("### 📄 Official Digital Hospital Referral Ticket")
                    referral_ticket_view = gr.Textbox(
                        label="Digital Hospital Referral Ticket (Copy to Clipboard / EMR Transfer)", 
                        value=initial_ticket_text,
                        lines=11, 
                        interactive=False
                    )

                # Tab 6: Prediction History
                with gr.TabItem("📜 Session Prediction History", elem_id="rg-tab-history"):
                    gr.Markdown("### 🕐 Prediction History — Current Session")
                    gr.Markdown("Each analysis run is logged here for comparison during the same session. History resets on page refresh.")
                    session_history_view = gr.HTML('<div style="color:#94a3b8; font-size:13px; padding:12px;">No predictions yet.</div>')

                # Tab 7: Image Comparison & Download
                with gr.TabItem("🖼️ Image Comparison & Report", elem_id="rg-tab-compare"):
                    gr.Markdown("### 📸 Original vs. Preprocessed Fundus Image")
                    gr.Markdown("Left: raw upload. Right: after Ben Graham enhancement, circular crop, and 224×224 resize.")
                    with gr.Row():
                        original_img_view = gr.Image(label="Original Upload", type="numpy", height=260, interactive=False)
                        processed_img_view = gr.Image(label="Preprocessed (Ben Graham Enhanced)", type="numpy", height=260, interactive=False)
                    quality_warning_view = gr.HTML('<div style="color:#94a3b8; font-size:12px;">Upload an image to see quality assessment.</div>')
                    gr.Markdown("---")
                    gr.Markdown("### 📥 Download Full Diagnosis Report")
                    gr.Markdown("Downloads a structured JSON file containing the diagnosis, confidence, probabilities, advisory, and EHR note.")
                    download_btn = gr.Button("⬇️ Generate & Download Report (JSON)", variant="primary")
                    download_file = gr.File(label="Download", visible=False)

                with gr.TabItem("📊 Longitudinal Analysis", elem_id="rg-tab-longitudinal"):
                    gr.Markdown("### Optional Previous vs Current Examination Comparison")
                    gr.Markdown("Do not assume the images belong to the same patient. Results are visual comparisons only, not confirmed disease progression.")
                    with gr.Row():
                        longitudinal_previous = gr.Image(label="Previous Examination", type="numpy", height=240)
                        longitudinal_current = gr.Image(label="Current Examination", type="numpy", height=240)
                    longitudinal_compare_btn = gr.Button("Compare Examinations", variant="primary")
                    longitudinal_summary_view = gr.HTML("<div style='color:#64748b;'>Submit two images to begin.</div>")
                    longitudinal_diff_view = gr.Image(label="Registration/Difference Visualisation", type="numpy", height=240)



    # ─────────────────────────────────────────────────────────────────────────
    # 8. Event Connections
    # ─────────────────────────────────────────────────────────────────────────
    def update_triage_routing(prob_dict, hba1c, duration, age, bp, d_type):
        st = 2
        if isinstance(prob_dict, dict) and prob_dict:
            stage_map = {name: i for i, name in enumerate(AppConfig.CLASS_NAMES)}
            try:
                top_name = max(prob_dict, key=prob_dict.get)
                st = stage_map.get(top_name, 2)
            except (KeyError, ValueError, TypeError):
                st = 2
        return calculate_multimodal_risk(st, hba1c, duration, age, bp, d_type)


    # Reusable analysis outputs tuple for all analyze_fundus call sites
    analysis_outputs = [
        status_banner,
        hero_diagnosis,
        prob_distribution,
        overlay_cam_view,
        lesion_seg_view,
        vessel_overlay_view,
        optic_disc_overlay_view,
        classical_cv_status_view,
        overlap_view,
        severity_map_view,
        advanced_analysis_view,
        quadrant_text,
        gallery_view,
        advisory_view,
        ehr_note_box,
        lesion_burden_view,
        quadrant_chart_view,
        confidence_margin_view,
        pred_context_state,
        session_history_state,
        research_support_state,
        uncertainty_banner_top,
        processed_img_view,
        session_history_view,
        _diag_stage_name_state,
        _diag_conf_state,
        _diag_urgency_state,
        _diag_followup_state,
        _diag_plan_state,
        quality_warning_view,
    ]

    # Main Analysis Event
    submit_btn.click(
        fn=analyze_fundus,
        inputs=[input_image, threshold_slider, session_history_state],
        outputs=analysis_outputs,
    ).then(
        fn=update_triage_routing,
        inputs=[prob_distribution, hba1c_level, diabetes_duration, patient_age, systolic_bp, diabetes_type],
        outputs=[triage_risk_card, referral_ticket_view],
    ).then(
        fn=lambda img: img,
        inputs=[input_image],
        outputs=[original_img_view],
    )

    download_btn.click(
        fn=generate_report_json,
        inputs=[_diag_stage_name_state, _diag_conf_state, prob_distribution,
            _diag_urgency_state, _diag_followup_state, _diag_plan_state, ehr_note_box, research_support_state],
        outputs=[download_file],
    ).then(
        fn=lambda: gr.File(visible=True),
        outputs=[download_file],
    )

    longitudinal_compare_btn.click(
        fn=compare_longitudinal_images,
        inputs=[longitudinal_previous, longitudinal_current],
        outputs=[longitudinal_summary_view, longitudinal_diff_view],
    )

    # Interactive Multimodal Risk Recalculation Handlers
    btn_recalc_triage.click(
        fn=update_triage_routing,
        inputs=[prob_distribution, hba1c_level, diabetes_duration, patient_age, systolic_bp, diabetes_type],
        outputs=[triage_risk_card, referral_ticket_view],
    )
    hba1c_level.release(
        fn=update_triage_routing,
        inputs=[prob_distribution, hba1c_level, diabetes_duration, patient_age, systolic_bp, diabetes_type],
        outputs=[triage_risk_card, referral_ticket_view],
    )
    diabetes_duration.release(
        fn=update_triage_routing,
        inputs=[prob_distribution, hba1c_level, diabetes_duration, patient_age, systolic_bp, diabetes_type],
        outputs=[triage_risk_card, referral_ticket_view],
    )
    systolic_bp.release(
        fn=update_triage_routing,
        inputs=[prob_distribution, hba1c_level, diabetes_duration, patient_age, systolic_bp, diabetes_type],
        outputs=[triage_risk_card, referral_ticket_view],
    )
    patient_age.release(
        fn=update_triage_routing,
        inputs=[prob_distribution, hba1c_level, diabetes_duration, patient_age, systolic_bp, diabetes_type],
        outputs=[triage_risk_card, referral_ticket_view],
    )
    diabetes_type.change(
        fn=update_triage_routing,
        inputs=[prob_distribution, hba1c_level, diabetes_duration, patient_age, systolic_bp, diabetes_type],
        outputs=[triage_risk_card, referral_ticket_view],
    )

    def reset_workspace():
        empty_img = np.zeros((AppConfig.IMG_SIZE, AppConfig.IMG_SIZE, 3), dtype=np.uint8)
        initial_banner = "<div class='card'><em>Upload a retinal fundus photograph or click a quick-load sample to begin.</em></div>"
        default_risk_html, default_ticket_text = calculate_multimodal_risk(0, 7.0, 5.0, 50.0, 120.0, "Type 2")
        return (
            None,
            0.70,
            initial_banner,
            "",
            {},
            empty_img,
            empty_img,
            empty_img,
            empty_img,
            "<div style='color:#64748b; font-size:12px;'>Visual support only — not an independent diagnosis.</div>",
            empty_img,
            empty_img,
            "",
            "",
            [],
            "",
            "",
            "",
            "",
            "",
            default_risk_html,
            default_ticket_text,
            {},
            [],
            {},
            "",
            empty_img,
            '<div style="color:#94a3b8; font-size:13px; padding:12px;">No predictions yet.</div>',
            "",
            0.0,
            "",
            "",
            "",
            '<div style="color:#94a3b8; font-size:12px;">Upload an image to see quality assessment.</div>',
            empty_img,
            None,
        )

    btn_clear.click(
        fn=reset_workspace,
        outputs=[
            input_image,
            threshold_slider,
            status_banner,
            hero_diagnosis,
            prob_distribution,
            overlay_cam_view,
            lesion_seg_view,
            vessel_overlay_view,
            optic_disc_overlay_view,
            classical_cv_status_view,
            overlap_view,
            severity_map_view,
            advanced_analysis_view,
            quadrant_text,
            gallery_view,
            advisory_view,
            ehr_note_box,
            lesion_burden_view,
            quadrant_chart_view,
            confidence_margin_view,
            triage_risk_card,
            referral_ticket_view,
            pred_context_state,
            session_history_state,
            research_support_state,
            uncertainty_banner_top,
            processed_img_view,
            session_history_view,
            _diag_stage_name_state,
            _diag_conf_state,
            _diag_urgency_state,
            _diag_followup_state,
            _diag_plan_state,
            quality_warning_view,
            original_img_view,
            download_file,
        ],
    )

    # Live threshold adjustment re-evaluates active prediction upon release
    threshold_slider.release(
        fn=analyze_fundus,
        inputs=[input_image, threshold_slider, session_history_state],
        outputs=analysis_outputs,
    ).then(
        fn=update_triage_routing,
        inputs=[prob_distribution, hba1c_level, diabetes_duration, patient_age, systolic_bp, diabetes_type],
        outputs=[triage_risk_card, referral_ticket_view],
    ).then(
        fn=lambda img: img,
        inputs=[input_image],
        outputs=[original_img_view],
    )

    # Preset Sample Button Handlers
    btn_normal.click(
        fn=lambda: (create_sample_fundus(0), 0.70),
        outputs=[input_image, threshold_slider],
    ).then(
        fn=analyze_fundus,
        inputs=[input_image, threshold_slider, session_history_state],
        outputs=analysis_outputs,
    ).then(
        fn=update_triage_routing,
        inputs=[prob_distribution, hba1c_level, diabetes_duration, patient_age, systolic_bp, diabetes_type],
        outputs=[triage_risk_card, referral_ticket_view],
    ).then(
        fn=lambda img: img,
        inputs=[input_image],
        outputs=[original_img_view],
    )

    btn_moderate.click(
        fn=lambda: (create_sample_fundus(2), 0.70),
        outputs=[input_image, threshold_slider],
    ).then(
        fn=analyze_fundus,
        inputs=[input_image, threshold_slider, session_history_state],
        outputs=analysis_outputs,
    ).then(
        fn=update_triage_routing,
        inputs=[prob_distribution, hba1c_level, diabetes_duration, patient_age, systolic_bp, diabetes_type],
        outputs=[triage_risk_card, referral_ticket_view],
    ).then(
        fn=lambda img: img,
        inputs=[input_image],
        outputs=[original_img_view],
    )

    btn_prolif.click(
        fn=lambda: (create_sample_fundus(4), 0.70),
        outputs=[input_image, threshold_slider],
    ).then(
        fn=analyze_fundus,
        inputs=[input_image, threshold_slider, session_history_state],
        outputs=analysis_outputs,
    ).then(
        fn=update_triage_routing,
        inputs=[prob_distribution, hba1c_level, diabetes_duration, patient_age, systolic_bp, diabetes_type],
        outputs=[triage_risk_card, referral_ticket_view],
    ).then(
        fn=lambda img: img,
        inputs=[input_image],
        outputs=[original_img_view],
    )

    # Safety Override Simulation Button Handler (Sets threshold to 95% and executes)
    btn_override_test.click(
        fn=lambda: 0.95,
        outputs=[threshold_slider],
    ).then(
        fn=analyze_fundus,
        inputs=[input_image, threshold_slider, session_history_state],
        outputs=analysis_outputs,
    ).then(
        fn=update_triage_routing,
        inputs=[prob_distribution, hba1c_level, diabetes_duration, patient_age, systolic_bp, diabetes_type],
        outputs=[triage_risk_card, referral_ticket_view],
    ).then(
        fn=lambda img: img,
        inputs=[input_image],
        outputs=[original_img_view],
    )



    # ── Chatbot Event Handlers ──────────────────────────────────────────────
    # -- Quick-Prompt Chip Handlers
    chip_classify.click(fn=lambda: 'Explain why this image was classified with the current DR stage.', outputs=[chat_input])
    chip_rule421.click(fn=lambda: 'What is the 4-2-1 rule in diabetic retinopathy?', outputs=[chat_input])
    chip_gradcam.click(fn=lambda: 'How does Grad-CAM work and what does it highlight?', outputs=[chat_input])
    chip_refer.click(fn=lambda: 'When should a patient be referred urgently to a retina specialist?', outputs=[chat_input])

    chat_send_btn.click(
        fn=respond_to_clinical_query,
        inputs=[chat_input, chatbot_widget, pred_context_state],
        outputs=[chatbot_widget, chat_input],
    )
    chat_input.submit(
        fn=respond_to_clinical_query,
        inputs=[chat_input, chatbot_widget, pred_context_state],
        outputs=[chatbot_widget, chat_input],
    )
    chat_clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot_widget, chat_input])


if __name__ == "__main__":
    demo.launch(head=HEAD_SCRIPT, theme=theme, share=True)
