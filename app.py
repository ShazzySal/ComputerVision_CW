"""
app.py — RetinaGuard AI: Multi-Agent Clinical Decision Support System
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
from typing import Tuple, List, Dict, Any, Union, Optional
import numpy as np
import cv2
import gradio as gr

from core.config import AppConfig
from core.preprocessing import preprocess_image

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
            "RetinaGuard uses **EfficientNetB3** pre-trained on ImageNet as its convolutional "
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
            "A symmetrical U-Net architecture provides pixel-level lesion contours "
            "(microaneurysms, exudates highlighted in fluorescent green).\n\n"
            "Because the 38,034-image dataset lacks manual pixel-level annotations, "
            "masks are synthesized via a **semi-supervised self-distillation pipeline**:\n"
            "1. Green-channel optical extraction (best hemoglobin contrast).\n"
            "2. Morphological Top-Hat (bright exudates) + Black-Hat (dark microaneurysms).\n"
            "3. Grad-CAM saliency gating (threshold > 0.35) to discard non-pathological edges.\n\n"
            "The U-Net is trained with a **Hybrid Soft Dice + BCE Loss** to handle "
            "the extreme class imbalance (<2% lesion pixels)."
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
            "After diagnosis, RetinaGuard extracts a **256-dimensional feature vector** "
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
            "   This eliminates global illumination gradients and reveals micro-vascular detail.\n"
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
            "RetinaGuard achieves a QWK of ~0.842, indicating substantial clinical agreement."
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
            "**Dataset — 38,034-Image Multi-Source Fundus Cohort**\n\n"
            "RetinaGuard is trained on the **Combined DR Dataset** (Harsha, 2020) from Kaggle, "
            "pooling four internationally recognized ophthalmic cohorts:\n\n"
            "1. **APTOS 2019:** Rural Indian clinic screening, multi-camera variability.\n"
            "2. **IDRiD:** Gold-standard Indian clinical staging with sub-lesion verification.\n"
            "3. **Messidor-2:** European multi-hospital 3-CCD camera study.\n"
            "4. **EyePACS Subset:** US population-scale real-world screening repository.\n\n"
            "Total: **38,034 annotations** split 70/15/15 (Train/Val/Test) using "
            "`StratifiedGroupKFold` with patient-level grouping to prevent bilateral eye leakage."
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


def respond_to_clinical_query(message: str, history: List, pred_context: dict = None) -> tuple:
    """Rule-based clinical knowledge chatbot for DR staging and model architecture queries.
    If pred_context is provided (from the current analysis), context-aware questions are answered."""
    if not message or not message.strip():
        return history, ""
    query = message.lower().strip()
    reply = None

    # Context-aware answers about the current prediction
    if pred_context and isinstance(pred_context, dict) and pred_context.get("stage_name"):
        ctx_triggers = ["why", "classified", "this image", "current", "result", "prediction", "explain this",
                        "why moderate", "why severe", "why mild", "why proliferative", "why no dr",
                        "confidence", "how confident", "what was found", "lesion", "quadrant", "peak"]
        if any(t in query for t in ctx_triggers):
            stage_name = pred_context.get("stage_name", "Unknown")
            conf = pred_context.get("confidence", 0.0)
            urgency = pred_context.get("urgency", "N/A")
            followup = pred_context.get("followup", "N/A")
            plan = pred_context.get("plan", "N/A")
            peak_q = pred_context.get("peak_quadrant", "-")
            lesion_pct = pred_context.get("lesion_pct", 0.0)
            reply = (
                f"**Current Prediction Context: {stage_name}**\n\n"
                f"The model classified this fundus image as **{stage_name}** with a confidence of **{conf*100:.1f}%**.\n\n"
                f"**Why this classification?**\n"
                f"- The Grad-CAM attention map highlighted the **{peak_q}** quadrant as the primary region of pathological activation.\n"
                f"- The U-Net lesion segmentation identified **{lesion_pct:.1f}%** of the retinal parenchyma as containing lesion candidates (microaneurysms / exudates).\n"
                f"- These visual features are consistent with the morphological criteria for **{stage_name}** on the ICDR severity scale.\n\n"
                f"**Clinical Advisory:**\n"
                f"- Urgency: {urgency}\n"
                f"- Recommended follow-up: {followup}\n"
                f"- Action: {plan}\n\n"
                f"*Note: This is an AI-assisted analysis. Always confirm with a qualified ophthalmologist.*"
            )

    if reply is None:
        for entry in _CLINICAL_KB:
            if any(kw in query for kw in entry["keys"]):
                reply = entry["reply"]
                break

    if reply is None:
        reply = _CHATBOT_FALLBACK

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    return history, ""
import json as _json
import datetime as _datetime
import tempfile as _tempfile


def generate_report_json(stage_name: str, confidence: float, probabilities: dict,
                         advisory_urgency: str, advisory_followup: str,
                         advisory_plan: str, ehr_text: str) -> str:
    """Serializes the current diagnosis to a timestamped JSON file for download."""
    if not stage_name or not probabilities or not ehr_text:
        return None
    report = {
        "retinaguard_report": {
            "generated_at": _datetime.datetime.now().isoformat(),
            "model": "EfficientNetB3 + U-Net + CBR (RetinaGuard AI)",
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
            "disclaimer": "This report is generated by an AI assistive tool and must be reviewed by a qualified ophthalmologist before any clinical decision.",
        }
    }
    tmp = _tempfile.NamedTemporaryFile(delete=False, suffix="_retinaguard_report.json", mode="w", encoding="utf-8")
    _json.dump(report, tmp, indent=2)
    tmp.close()
    return tmp.name


def calculate_multimodal_risk(stage: int = 2, hba1c: float = 7.5, duration_years: float = 10.0, age: float = 55.0, systolic_bp: float = 135.0, diabetes_type: str = "Type 2") -> Tuple[str, str]:
    """Computes evidence-based 10-year vision loss progression risk and NHS hospital triage dispatch routing (UKPDS/WESDR)."""
    try:
        stage = int(stage) if stage is not None else 2
    except Exception:
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
    <div style="padding:16px; background:#f8fafc; border-radius:12px; border:1px solid #e2e8f0; margin-bottom:12px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; border-bottom:1px solid #e2e8f0; padding-bottom:8px;">
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
            <div style="display:flex; gap:8px;">
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
        return notice, "", {}, empty_img, empty_img, empty_img, empty_img, "", [], "", "", "", "", None, [], "", empty_img, "", "", 0.0, "", "", "", ""

    try:
        preproc = preprocess_image(img)
    except ValueError as exc:
        notice = (
            "<div class='card warning-card'>⚠️ <strong>Invalid fundus image input</strong> — "
            f"{exc}. Please upload a valid retina image or choose a sample fundus from the quick-load buttons.</div>"
        )
        return notice, "", {}, empty_img, empty_img, empty_img, empty_img, "", [], "", "", "", "", None, [], "", empty_img, "", "", 0.0, "", "", "", ""

    # Image Quality Assessment (before inference)
    qc = assess_image_quality(img)
    quality_html = build_quality_warning_html(qc)

    result = run_pipeline(preproc, threshold=threshold)


    diag = result["diagnosis"]
    expl = result["explanation"]
    adv = result["advisory"]
    flagged = result["flagged"]
    stage = diag["stage"]

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
        "           RETINAGUARD AI CLINICAL TRIAGE NOTE             \n"
        "===========================================================\n"
        f"ASSESSMENT DATE/TIME   : Diagnostic Session Active\n"
        f"PREDICTED DR STAGE     : Stage {stage} — {diag['stage_name']}\n"
        f"MODEL CONFIDENCE       : {diag['confidence']*100:.2f}%\n"
        f"GOVERNANCE GATE STATUS : {gate_label}\n"
        f"SAFETY THRESHOLD ENF.  : {threshold*100:.0f}%\n"
        f"PEAK ANATOMICAL REGION : {clean_quad}\n"
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
        expl["quadrant_desc"],          # 8: quadrant_text
        gallery_items,                  # 9: gallery_view
        advisory_html,                  # 10: advisory_view
        ehr_text,                       # 11: ehr_note_box
        lesion_burden_html,             # 12: lesion_burden_view
        quadrant_chart_html,             # 13: quadrant_chart_view
        confidence_margin_html,         # 14: confidence_margin_view
        pred_context,                   # 15: pred_context_state
        session_history,                # 16: session_history_state
        quality_html + uncertainty_banner_html,  # 17: uncertainty_banner_top
        processed_display,              # 18: processed_img_view
        session_history_html,           # 19: session_history_view
        diag["stage_name"],             # 20: _diag_stage_name_state
        conf_val,                       # 21: _diag_conf_state
        adv["urgency"],                 # 22: _diag_urgency_state
        adv["followup"],                # 23: _diag_followup_state
        adv["plan"],                    # 24: _diag_plan_state
        quality_html,                   # 25: quality_warning_view
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Modern UI Assembly (Gradio Blocks Layout)
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* Medical Dashboard Base Styling */
.gradio-container {
    max-width: 1280px !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    transition: background-color 0.3s ease, color 0.3s ease;
}

/* Header Telemetry Styling */
.telemetry-bar {
    display: flex;
    gap: 12px;
    background: #0f172a;
    color: #f8fafc;
    padding: 10px 16px;
    border-radius: 8px;
    font-size: 12px;
    margin-bottom: 16px;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    border: 1px solid #1e293b;
}
.dark .telemetry-bar {
    background: #020617;
    border: 1px solid #1e293b;
}
.telemetry-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1e293b;
    padding: 4px 10px;
    border-radius: 6px;
    border: 1px solid #334155;
    font-weight: 500;
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
    background: #1e293b !important;
    color: #f8fafc !important;
    border: 1px solid #475569 !important;
}
.theme-toggle-btn:hover {
    background: #334155 !important;
    transform: translateY(-1px);
}
.dark .theme-toggle-btn {
    background: #f8fafc !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
}
.dark .theme-toggle-btn:hover {
    background: #e2e8f0 !important;
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
}

/* Stepper Component (Medios / Mobile Workflow) */
.stepper-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #042f2e;
    border: 1px solid #115e59;
    border-radius: 10px;
    padding: 10px 18px;
    margin-bottom: 16px;
    color: #f0fdfa;
    flex-wrap: wrap;
    gap: 8px;
}
.dark .stepper-container {
    background: #022c22;
    border-color: #134e4a;
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
    color: #ccfbf1;
}
.stepper-sub {
    font-size: 10px;
    color: #99f6e4;
    opacity: 0.85;
}
.stepper-arrow {
    color: #2dd4bf;
    font-weight: 800;
    font-size: 14px;
}

/* Medios Patient EHR Card */
.patient-id-card {
    background: #ffffff;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
    padding: 12px 14px;
    margin-bottom: 12px;
}
.dark .patient-id-card {
    background: #0f172a;
    border-color: #334155;
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
}
.dark .patient-chip {
    background: #1e293b;
    color: #cbd5e1;
}

/* Medios SaMD Regulatory Card */
.medios-disclaimer-card {
    background: #042f2e;
    border: 1px solid #115e59;
    border-radius: 10px;
    padding: 12px 16px;
    margin-top: 14px;
    color: #ccfbf1;
}
.dark .medios-disclaimer-card {
    background: #022c22;
    border-color: #134e4a;
}
</style>
"""

HEAD_SCRIPT = """
<script>
(function() {
    try {
        const savedTheme = localStorage.getItem('retinaguard_theme');
        const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
            document.documentElement.classList.add('dark');
            document.body.classList.add('dark');
            const elApp = document.querySelector('gradio-app');
            if (elApp) elApp.classList.add('dark');
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
        try { localStorage.setItem('retinaguard_theme', 'light'); } catch(e) {}
    } else {
        targets.forEach(t => t.classList.add('dark'));
        try { localStorage.setItem('retinaguard_theme', 'dark'); } catch(e) {}
    }
}
"""

theme = gr.themes.Soft(primary_hue="teal", secondary_hue="slate")

with gr.Blocks(title="RetinaGuard AI — Diabetic Retinopathy CDS") as demo:
    # Inject Custom Clinical Styling & Theme Detection
    gr.HTML(CUSTOM_CSS)

    # Session state
    pred_context_state = gr.State({})
    session_history_state = gr.State([])
    _diag_stage_name_state = gr.State("")
    _diag_conf_state = gr.State(0.0)
    _diag_urgency_state = gr.State("")
    _diag_followup_state = gr.State("")
    _diag_plan_state = gr.State("")

    # 1. Main Header & Telemetry Bar
    with gr.Row():
        with gr.Column(scale=9):
            gr.HTML("""
            <div style="text-align:left;">
                <h1 class="header-title" style="margin:0; font-size:26px; font-weight:800; letter-spacing:-0.5px;">
                    👁️ RetinaGuard AI: Clinical Decision Support & Governance System
                </h1>
                <p class="header-subtitle" style="margin:4px 0 0 0; font-size:14px;">
                    Multi-Stage Diabetic Retinopathy Diagnostic Pipeline • BSc (Hons) Computer Science Coursework
                </p>
            </div>
            """)
        with gr.Column(scale=3, min_width=220):
            theme_toggle_btn = gr.Button("🌓 Toggle Dark / Light", size="sm", elem_classes=["theme-toggle-btn"])
            gr.HTML("""
            <div style="text-align:right; margin-top:4px;">
                <span style="background:#e0f2fe; color:#0369a1; padding:4px 10px; border-radius:6px; font-weight:700; font-size:12px;">
                    EfficientNetB3 • Grad-CAM • U-Net • CBR
                </span>
            </div>
            """)

    gr.HTML("""
    <div class="telemetry-bar">
        <div style="display:flex; align-items:center; gap:8px;">
            <div class="pulse-dot"></div>
            <strong style="color:#f8fafc; font-size:13px;">ACTIVE MULTI-AGENT TELEMETRY</strong>
        </div>
        <div style="display:flex; gap:8px; flex-wrap:wrap;">
            <div class="telemetry-badge">🩺 DiagnosisAgent: <strong>Online</strong></div>
            <div class="telemetry-badge">🔬 ExplainabilityAgent: <strong>Online</strong></div>
            <div class="telemetry-badge">📋 AdvisoryAgent: <strong>Online</strong></div>
            <div class="telemetry-badge" style="border-color:#38bdf8;">🛡️ GovernanceGate: <strong>Active</strong></div>
        </div>
    </div>
    """)

    # Clinical Workflow Stepper Bar (Medios / Mobile Screening Workflow)
    gr.HTML("""
    <div class="stepper-container">
        <div class="stepper-item">
            <span class="stepper-circle">01</span>
            <div><div class="stepper-title">CLINICAL INTAKE</div><div class="stepper-sub">Patient Profile &amp; HbA1c</div></div>
        </div>
        <div class="stepper-arrow">➔</div>
        <div class="stepper-item">
            <span class="stepper-circle">02</span>
            <div><div class="stepper-title">IMAGE ACQUISITION</div><div class="stepper-sub">Optical Quality Check</div></div>
        </div>
        <div class="stepper-arrow">➔</div>
        <div class="stepper-item">
            <span class="stepper-circle">03</span>
            <div><div class="stepper-title">AI SAFETY GATE</div><div class="stepper-sub">70% Threshold &amp; Override</div></div>
        </div>
        <div class="stepper-arrow">➔</div>
        <div class="stepper-item">
            <span class="stepper-circle">04</span>
            <div><div class="stepper-title">CLINICAL DOSSIER</div><div class="stepper-sub">Triage, Grad-CAM &amp; 10-Yr Risk</div></div>
        </div>
    </div>
    """)

    # 2. Main Workspace (2 Columns)
    with gr.Row():
        # Left Column: Upload & Governance Configuration
        with gr.Column(scale=4):
            gr.HTML("""
            <div class="patient-id-card">
                <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #e2e8f0; padding-bottom:8px; margin-bottom:8px;">
                    <div>
                        <div style="font-size:10px; color:#64748b; font-weight:800; text-transform:uppercase;">EHR Active Screening Record</div>
                        <div style="font-size:15px; font-weight:800; color:#0f172a;">MRN: RG-2026-9812 &bull; Test Patient</div>
                    </div>
                    <span class="badge-quota">⚡ 99 SCANS LEFT</span>
                </div>
                <div style="display:flex; flex-wrap:wrap; gap:6px;">
                    <span class="patient-chip">👤 Age: 55y</span>
                    <span class="patient-chip">🩺 Type 2 DM</span>
                    <span class="patient-chip">⏳ 10 Years</span>
                    <span class="patient-chip">🩸 HbA1c: 7.5%</span>
                    <span class="patient-chip">💓 BP: 135 mmHg</span>
                </div>
            </div>
            """)
            gr.Markdown("### 📥 Retinal Photography Input")
            input_image = gr.Image(label="Upload Fundus Photo", type="numpy", height=280)

            # Quick Preset Buttons
            gr.Markdown("**⚡ Quick-Load Test Samples (No download needed):**")
            with gr.Row():
                btn_normal = gr.Button("🟢 Normal (Stage 0)", size="sm")
                btn_moderate = gr.Button("🟡 Moderate (Stage 2)", size="sm")
                btn_prolif = gr.Button("🔴 Proliferative (Stage 4)", size="sm")

            gr.Markdown("---")
            gr.Markdown("### 🛡️ Clinical Safety Gate Config")
            threshold_slider = gr.Slider(
                minimum=0.50, maximum=0.95, value=0.70, step=0.05,
                label="Governance Confidence Threshold",
                info="Predictions below this confidence trigger an active safety override and withhold automated guidance.",
            )

            # Safety Gate Override Test Trigger
            btn_override_test = gr.Button("🧪 Simulate Safety Override (Set to 95%)", variant="secondary", size="sm")

            gr.Markdown("---")
            gr.Markdown("### Patient Intake Panel *(Optional Context)*")
            patient_age = gr.Slider(minimum=18, maximum=90, value=55, step=1, label="Patient Age (years)")
            diabetes_type = gr.Dropdown(
                choices=["Type 1", "Type 2", "Gestational", "Not Specified"],
                value="Type 2", label="Diabetes Type"
            )
            hba1c_level = gr.Slider(minimum=5.0, maximum=14.0, value=7.5, step=0.1, label="HbA1c (%)")
            diabetes_duration = gr.Slider(minimum=0, maximum=40, value=10, step=1, label="Duration of Diabetes (years)")
            systolic_bp = gr.Slider(minimum=90, maximum=220, value=135, step=1, label="Systolic Blood Pressure (mmHg)")
            gr.Markdown("---")
            with gr.Row():
                submit_btn = gr.Button("🚀 Run Diagnostic Analysis", variant="primary", size="lg", scale=3, elem_classes=["action-btn"])
                btn_clear = gr.Button("🔄 Reset", variant="secondary", size="lg", scale=1)

        # Right Column: Multi-Tab Clinical Dossier
        with gr.Column(scale=6):
            # Governance Status Banner (Appears at the very top of results)
            status_banner = gr.HTML("<div class='card'><em>Upload a retinal fundus photograph or click a quick-load sample to begin.</em></div>")

            # Structured Tabs
            with gr.Tabs():
                # Tab 1: Primary Diagnosis & 3-Layer Explainability
                with gr.TabItem("🏥 Diagnostic Assessment & Explainability"):
                    uncertainty_banner_top = gr.HTML()
                    hero_diagnosis = gr.HTML()
                    prob_distribution = gr.Label(label="5-Stage Disease Probability Distribution (Softmax)", num_top_classes=5)

                    gr.Markdown("---")
                    gr.Markdown("### 🔬 Multi-Layer Morphological Explainability")
                    with gr.Row():
                        with gr.Column(scale=5):
                            gr.Markdown("**Layer 2: Regional Attention (Grad-CAM)**")
                            overlay_cam_view = gr.Image(label="Grad-CAM Saliency Overlay", type="numpy", height=240)
                        with gr.Column(scale=5):
                            gr.Markdown("**Layer 3: Lesion Segmentation (U-Net)**")
                            lesion_seg_view = gr.Image(label="Segmented Lesions (Fluorescent Green)", type="numpy", height=240)
                    gr.Markdown("### Classical CV Visual Support")
                    with gr.Row():
                        with gr.Column(scale=5):
                            gr.Markdown("**Classical CV Vessel Analysis**")
                            vessel_overlay_view = gr.Image(label="Retinal Vessel Overlay", type="numpy", height=220)
                        with gr.Column(scale=5):
                            gr.Markdown("**Optic-Disc Localisation**")
                            optic_disc_overlay_view = gr.Image(label="Optic-Disc Localisation Overlay", type="numpy", height=220)
                    classical_cv_status_view = gr.HTML(
                        "<div style='color:#64748b; font-size:12px;'>Visual support only — not an independent diagnosis.</div>"
                    )

                    lesion_burden_view = gr.HTML()
                    quadrant_text = gr.Markdown()
                    quadrant_chart_view = gr.HTML()
                    confidence_margin_view = gr.HTML()

                # Tab 2: Case-Based Reasoning (CBR) Evidence
                with gr.TabItem("📚 Case-Based Reasoning (CBR) Evidence"):
                    gr.Markdown("### 🔎 Nearest Available Reference Cases")
                    gr.Markdown(
                        "The query image was projected into the 256-D penultimate feature bottleneck. "
                        "When a verified reference library is available, the **Top-3 closest matching cases** are retrieved via cosine similarity:"
                    )
                    gallery_view = gr.Gallery(columns=3, rows=1, height=260, object_fit="contain")

                # Tab 3: Clinical Care Protocol & EHR Note
                with gr.TabItem("📋 Clinical Management & EHR Note"):
                    advisory_view = gr.HTML()
                    gr.Markdown("### 📄 Exportable Electronic Health Record (EHR) Summary Note")
                    ehr_note_box = gr.Textbox(label="Clinical Session Note (Copy to Clipboard)", lines=12, interactive=False)

                # Tab 4: AI Clinical Chatbot & SaMD Guidelines
                with gr.TabItem("💬 AI Clinical Chatbot"):
                    gr.Markdown("""
                    ### 🤖 RetinaGuard Clinical Knowledge Assistant
                    Ask any question about diabetic retinopathy stages, the model architecture, preprocessing pipeline,
                    or clinical management guidelines. The assistant references the AAO Preferred Practice Patterns.

                    *Example questions: "What does Stage 3 mean?", "When should I refer a proliferative patient?",
                    "How does Grad-CAM work?", "What is the Governance Agent?"*
                    """)
                    with gr.Row():
                        with gr.Column(scale=3):
                            chatbot_widget = gr.Chatbot(
                                label="Clinical Knowledge Assistant",
                                height=420,
                            )
                            with gr.Row():
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
                        with gr.Column(scale=2):
                            gr.Markdown(r"""
                            ### ⚙️ System Specifications:
                            * **Deep Learning Backbone:** EfficientNetB3 fine-tuned on **38,034** multi-source fundus images (APTOS + IDRiD + Messidor-2 + EyePACS).
                            * **Ordinal Metric:** Evaluated via **Quadratic Weighted Kappa (QWK)** ($\kappa$) to quadratically penalize clinically dangerous multi-stage misclassifications.
                            * **Active Safety Gate:** Autonomous `GovernanceAgent` intercepts low-confidence predictions and routes to mandatory human specialist triage.
                            * **Regulatory Category:** SaMD (Software as a Medical Device) — Assistive Clinical Decision Support (FDA 21 CFR 860 / EU AI Act Class IIa).
                            """)

                # Tab 5: Multimodal Clinical Triage & Risk Simulator
                with gr.TabItem("🚦 Multimodal Triage & 10-Yr Risk Simulator"):
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
                with gr.TabItem("📜 Session Prediction History"):
                    gr.Markdown("### 🕐 Prediction History — Current Session")
                    gr.Markdown("Each analysis run is logged here for comparison during the same session. History resets on page refresh.")
                    session_history_view = gr.HTML('<div style="color:#94a3b8; font-size:13px; padding:12px;">No predictions yet.</div>')

                # Tab 7: Image Comparison & Download
                with gr.TabItem("🖼️ Image Comparison & Report"):
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

    # Medios-Standard SaMD Regulatory Notice Card (Image 1 Inspiration)
    gr.HTML("""
    <div class="medios-disclaimer-card">
        <div style="display:flex; align-items:flex-start; gap:12px;">
            <span style="font-size:24px;">🛡️</span>
            <div>
                <div style="font-weight:800; font-size:12.5px; color:#2dd4bf; letter-spacing:0.5px; text-transform:uppercase;">
                    Medios-Standard SaMD Clinical Decision Support Notice &bull; FDA 21 CFR 860 / EU AI Act Class IIa
                </div>
                <div style="font-size:11.5px; color:#ccfbf1; margin-top:3px; line-height:1.5;">
                    RetinaGuard AI is an assistive physician-support system developed for diabetic eye screening augmentation. It is not an autonomous diagnostic replacement for a definitive stereoscopic slit-lamp fundus biomicroscopic examination by a certified ophthalmologist. All autonomous therapeutic guidance is withheld whenever the model confidence is intercepted below the active Governance Safety Gate threshold.
                </div>
            </div>
        </div>
    </div>
    """)


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
            except Exception:
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
        quadrant_text,
        gallery_view,
        advisory_view,
        ehr_note_box,
        lesion_burden_view,
        quadrant_chart_view,
        confidence_margin_view,
        pred_context_state,
        session_history_state,
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
                _diag_urgency_state, _diag_followup_state, _diag_plan_state, ehr_note_box],
        outputs=[download_file],
    ).then(
        fn=lambda: gr.File(visible=True),
        outputs=[download_file],
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

    # Theme Toggle Event Handler (Client-Side JavaScript)
    theme_toggle_btn.click(
        fn=None,
        inputs=None,
        outputs=None,
        js=THEME_TOGGLE_JS,
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
