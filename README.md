---
title: RetinaTrace AI - Diabetic Retinopathy Research Prototype
emoji: 👁️
colorFrom: teal
colorTo: blue
sdk: gradio
sdk_version: "4.20"
app_file: app.py
pinned: false
license: other
short_description: Coursework prototype for diabetic-retinopathy image analysis
---

# Diabetic Retinopathy Stage Detection: Coursework Research Prototype

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.20%2B-green.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-Academic%20Coursework-lightgrey.svg)]()

> **BSc (Hons) in Computing (Batch 2024.2) — Computer Vision Coursework**  
> **Coventry University (UK) & National Institute of Business Management (NIBM)**  
> **Student Registered Name:** M.S.F. Shazna  
> **Coventry Index:** 16115859 | **NIBM Index:** COBSCCOMP242P-019  
> **Assessment Weighting:** 100 Marks (Individual Project Report with Video Demonstration)

---

## 📋 Executive Summary & Project Purpose

Diabetic Retinopathy (DR) is the leading cause of preventable blindness among working-age adults worldwide. Early detection and precise severity staging are critical: while early stages (Mild/Moderate NPDR) require monitoring and primary care glycemic optimization, advanced stages (Severe NPDR and Proliferative DR) demand urgent specialist laser photocoagulation or anti-VEGF pharmacotherapy to avert permanent visual loss.

This coursework project implements a research prototype for **five-class diabetic-retinopathy image analysis** (ICDR Stages 0 to 4), using an **EfficientNetB3** classifier and experimental explainability and retrieval components. It has not been clinically validated and is not intended for diagnosis or treatment decisions.
1. **Embedding-Based Similar-Case Retrieval (Case-Based Reasoning):** Prototype retrieval using image embeddings.
2. **Multi-Agent Decision Workflow:** Separate components for classification, visual explanations, advisory text, and a threshold-based low-confidence flag; this workflow does not enforce clinical review.

---

## 📊 Dataset Specification

* **Dataset:** [Combined DR Dataset (APTOS + IDRiD + Messidor-2 + EyePACS subset)](https://www.kaggle.com/datasets/harsha1289/combined-dr-dataset-aptosidridmessidoreyepacs)
* **Scale:** **38,034 downloaded image records** (21,000 in `train/`, 8,349 in `val/`, and 8,685 in `test/`). The duplicate-safe model partition is separate: 26,623 train / 5,706 validation / 5,705 test. The Kaggle description says approximately 21,000 images; see the report manifest for the measured folder counts and limitations.
* **Disease Staging Hierarchy (International Clinical Diabetic Retinopathy Scale):**
  * `Stage 0`: No DR (12,996 image records)
  * `Stage 1`: Mild Non-Proliferative DR (NPDR) (5,825)
  * `Stage 2`: Moderate NPDR (8,262)
  * `Stage 3`: Severe NPDR (5,373)
  * `Stage 4`: Proliferative DR (PDR) (5,578)
* **Class Imbalance Handling:** The locally counted maximum-to-minimum class ratio is approximately 2.4:1. The notebook computes inverse-frequency class weights from training labels; no controlled with/without-weight performance comparison is currently reported.
* **Label Scope:** The local `labels.csv` provides one record per file with a five-class diagnosis. It does not identify each record's source dataset, so per-source counts for the 38,034 local records cannot be verified from this copy.

---

## 🏗️ Repository Architecture & File Structure

```
Computer-Vision-CW/
├── diabetic_retinopathy_detection.ipynb  # Primary end-to-end research notebook (Sections 1-11)
├── app.py                                # Gradio UI, callbacks, and deployment entry point
├── core/                                  # Reusable backend modules
│   ├── __init__.py
│   ├── config.py                          # Shared application settings and checkpoint paths
│   ├── preprocessing.py                   # Fundus cropping, resizing, and Ben Graham enhancement
│   ├── models.py                          # EfficientNetB3, U-Net, Grad-CAM, and embeddings
│   ├── explainability.py                  # Grad-CAM, retrieval, segmentation, and classical CV
│   ├── advanced_cv.py                      # Research overlap, biomarkers, consistency, and longitudinal analysis
│   ├── research_evidence.py                # Ablation, ordinal error analysis, and reproducibility artifacts
│   └── agents.py                           # Diagnosis, advisory, explainability, and governance
├── requirements.txt                      # Complete pinned Python environment dependencies
├── README.md                             # Comprehensive technical documentation & reproduction guide
├── checkpoints/                          # Saved model weight checkpoints
│   ├── best_phase1.weights.h5            # Historical frozen-base checkpoint
│   └── best_phase2.weights.h5            # Historical fine-tuned checkpoint
└── report_images/                        # Saved project figures, metrics, and logs
    ├── class_distribution.png            # Imbalance breakdown visualization
    ├── preprocessing_class*.png          # Ben Graham & border-crop comparison grids
    ├── augmentation_examples.png         # Stochastic transform validation panels
    ├── curves_Phase_1_Frozen_Base.png    # Phase 1 loss/accuracy learning curves
    ├── curves_Phase_2_Fine_Tuning.png    # Phase 2 fine-tuning loss/accuracy curves
    ├── classification_report.csv         # Per-stage precision, recall, F1-scores
    ├── confusion_matrix.png              # Raw count & normalized recall heatmaps
    ├── gradcam_multiclass_overlays.png   # 5-stage Grad-CAM overlays & quadrant analysis
    └── similar_cases_demo.png            # Query image vs top-3 retrieved historical cases
```

  Research evidence utilities are documented in `research_evidence.md`. They run
  outside the deployed application and do not overwrite saved checkpoints. The
  ablation suite must be run
  in the GPU notebook environment because the local workspace does not contain
  the full training dataset or an executed notebook kernel.

> **Checkpoint status:** The saved checkpoints and reported EXP-03 metrics predate
> the corrected EfficientNet input-scale adapter and do not use the current
> duplicate-safe split. Retrain and evaluate before treating predictions from the
> updated application as validated model results.

---

## 🔬 Technical Innovation Features (Grounded in Module Lecture Materials)

All core innovations in this project are directly grounded in and adapted from prior coursework and laboratory materials:
* **Innovation 1 (Multi-Agent Safety Pipeline):** Derived from `6. Multi-Agent_AI_Blueprint.pdf` and `7/8. Defense_Multi_Agent_LLM.ipynb` (military ISR/cyber/governance decision pattern adapted to clinical safety).
* **Innovation 2 (Embedding-Based Case Retrieval):** Derived from `11. siamese_network_tutorial.ipynb` (AT&T Faces similarity learning adapted to retinal pathology Case-Based Reasoning).
* **Innovation 3 (Lesion-Level Segmentation):** Derived from `9. U-Net_Brain_Tumor_Segmentation.pdf` and `8. Brain_MRI_Segmentation_Kaggle_UNet_Dice_Report.pdf` (Brain tumor U-Net with Soft Dice loss adapted to microvascular fundus lesions).
* **Metric Formulation (Quadratic Weighted Kappa):** Derived from official APTOS/Kaggle competition evaluation standards for ordinal clinical disease grading.

### Innovation Feature A: Embedding-Based Similar-Case Retrieval (CBR Engine)

* **Theoretical Inspiration:** Inspired by **Siamese networks** and deep metric learning (traditionally deployed in facial verification and one-shot matching), this feature repurposes deep latent representations for clinical case comparison.
* **Implementation:** We tap the penultimate dense representation layer ($D = 256$) immediately prior to the 5-class softmax output layer. When a verified reference image library is available, 256-dimensional feature vectors are extracted and $L_2$-normalized such that Euclidean distance is strictly monotonic with cosine distance:
  $$\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} = u \cdot v \quad (\text{for } \|u\|_2 = \|v\|_2 = 1)$$
* **Clinical Rationale:** Medical practitioners rarely rely on an isolated probabilistic number. In ophthalmic practice, clinicians reason by **analogy to definitive historical cases** (Case-Based Reasoning). When the reference library is verified, returning the top-3 nearest cases with their labels offers **inter-case comparative explainability**, complementing the **intra-image spatial explainability** provided by Grad-CAM.

### Innovation Feature B: Multi-Agent Clinical Decision Pipeline & Safety Governance

* **Theoretical Rationale:** Single-model "monolithic" architectures and unstructured LLM chatbot wrappers are dangerous in clinical healthcare because they conflate perception with risk governance. A single high-probability hallucination can be presented as medical guidance without safety checks.
* **Architecture:** We architected a 4-agent decoupled clinical decision support pipeline:
  1. **`DiagnosisAgent`**: Dedicated to perceptual classification through the fine-tuned CNN, outputting discrete ICDR stages and softmax confidence scores.
  2. **`ExplainabilityAgent`**: Formulates evidence dossiers: computes spatial Grad-CAM saliency heatmaps, derives anatomical quadrant lesion descriptions (e.g., Superior-Temporal microaneurysm concentration), and executes similar-case retrieval.
  3. **`AdvisoryAgent`**: Aligns predicted stages with international clinical protocols (American Academy of Ophthalmology Preferred Practice Patterns & NHS Diabetic Eye Screening protocols), formulating concrete referral timeframes and management steps with an unambiguous legal disclaimer.
  4. **`GovernanceAgent` (Active Safety Gate)**: An autonomous safety officer. If `DiagnosisAgent` confidence falls below a configurable threshold (default: $70\%$), the `GovernanceAgent` **actively intercepts and overrides** the pipeline output. Rather than merely logging a warning, it withholds automated treatment advice, sets `flagged_for_review: True`, and issues an urgent clinical triage alert requiring manual ophthalmologist review.

---

### Innovation Feature C (Stretch Goal): Lesion-Level U-Net Segmentation (3-Layer Explainability Hierarchy)

* **Theoretical Translation:** Adapts the classical **U-Net** architecture (Ronneberger et al., 2015), originally designed for biomedical microscopy and brain tumor segmentation, to the domain of retinal microvascular lesions.
* **The 3-Layer Explainability Hierarchy:**
  1. **Layer 1 (Global Classification):** EfficientNetB3 outputs the 5-stage ICDR disease grade ($0-4$) and confidence score.
  2. **Layer 2 (Regional Attention):** Grad-CAM visualizes class-discriminative heatmap activations and maps peak pathology quadrants (*Superior-Temporal, Inferior-Nasal*, etc.).
  3. **Layer 3 (Pixel Segmentation):** An auxiliary U-Net with skip connections delineates microaneurysms, dot-and-blot hemorrhages, and hard exudates at the individual pixel level.
* **Hybrid Soft Dice Loss:** Because retinal lesions occupy $< 1-3\%$ of total pixels, standard binary cross-entropy collapses to predicting background. The network optimizes a **hybrid Soft Dice + BCE loss**:
  $$\mathcal{L} = 0.5\,\mathcal{L}_{\text{BCE}} + 0.5\left(1 - \frac{2\sum y_i\hat{y}_i + \epsilon}{\sum y_i + \sum \hat{y}_i + \epsilon}\right)$$
  ensuring stable gradient propagation while penalizing boundary overlap errors on tiny microvascular lesions.
* **Mask Synthesis Methodology (Semi-Supervised Self-Distillation):** Manually-annotated pixel-level lesion segmentation masks do not exist for the APTOS/EyePACS datasets at the 38,034-image scale used in this project. Rather than abandoning pixel-level explainability entirely, we employ a legitimate **semi-supervised self-distillation** technique: the trained EfficientNetB3 classifier's Grad-CAM attention maps are thresholded and combined with green-channel morphological analysis (top-hat transform + adaptive Otsu) to synthesize pseudo-masks that approximate the spatial extent of DR lesions. These synthesized masks are used exclusively to train the auxiliary U-Net for qualitative visual explainability -- they are not presented as clinical-grade annotations, and the U-Net's role is to generate interpretable overlays for the clinician, not to produce quantitative lesion measurements for diagnostic decisions.

## 🌟 Bonus Features Implemented

* **Bonus C: Interactive Gradio UI**: Complete clinical web dashboard featuring image drag-and-drop, adjustable governance threshold sliders, live Grad-CAM heatmaps, case retrieval galleries, and a public `share=True` link for video recording.
* **Bonus D: Cloud Hosting & Hugging Face Spaces Readiness**: Standalone `app.py` and `requirements.txt` structured specifically for zero-configuration deployment to Hugging Face Spaces using the Gradio SDK.
* **Bonus E: Multi-Stage Classification Verification**: The network strictly performs 5-class ordinal disease staging across all ICDR grades (`No DR`, `Mild`, `Moderate`, `Severe`, `Proliferative DR`), rejecting binary (DR present/absent) simplification.

---

## 🚀 Step-by-Step Reproduction Guide

### Option 1: Running in Google Colab (Recommended for Training)

1. Open [Google Colab](https://colab.research.google.com/) and upload `diabetic_retinopathy_detection.ipynb`.
2. Enable GPU acceleration: **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ **T4 GPU** (or A100).
3. Upload your Kaggle API token (`kaggle.json`) when prompted in **Section 1.5** to download and extract the 38,034 labelled fundus images.
4. Select **Runtime** $\rightarrow$ **Run all** to execute the pipeline end-to-end:
   - Data verification & integrity audit
   - Ben Graham contrast enhancement & border cropping
   - Stratified 70/15/15 split
   - Two-phase EfficientNetB3 training (frozen feature extraction $\rightarrow$ top-30 layer fine-tuning)
   - Evaluation (Accuracy, Quadratic Weighted Kappa, Confusion Matrix)
   - Grad-CAM heatmap generation & automated quadrant descriptions
   - Embedding extraction & Similar-Case Retrieval
   - Multi-agent decision pipeline execution
   - Launching the public Gradio web UI (`share=True`)

### Option 2: Local Execution

```bash
# 1. Clone the repository
git clone https://github.com/ShazzySal/ComputerVision_CW.git
cd ComputerVision_CW

# 2. Create and activate a clean virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the interactive Gradio clinical application
python app.py
```

### Option 3: Deploying to Hugging Face Spaces

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces) selecting the **Gradio SDK**.
2. Push `app.py`, `requirements.txt`, `README.md`, and your trained `checkpoints/best_phase2.weights.h5` to the Space repository.
3. Hugging Face Spaces will automatically build the environment and host your clinical AI app at a permanent public URL.

---

## 📈 Clinical Evaluation Methodology: Why QWK?

Standard classification accuracy treats a Mild vs. Severe error identically to a Mild vs. Moderate error. In ophthalmology, adjacent-stage errors (Mild vs Moderate) represent minor monitoring adjustments, whereas distant errors (classifying Severe NPDR or Proliferative DR as No DR) risk catastrophic vision loss due to omitted treatment.

The model is evaluated using **Quadratic Weighted Kappa (QWK)** via `cohen_kappa_score(weights='quadratic')`:
$$\kappa = 1 - \frac{\sum_{i,j} w_{ij} O_{ij}}{\sum_{i,j} w_{ij} E_{ij}}, \quad w_{ij} = \frac{(i - j)^2}{(N - 1)^2}$$
Quadratic penalties ($|i - j|^2$) heavily penalize distant staging mistakes, reflecting true clinical safety requirements.

---

## ⚖️ Ethical, Regulatory, & Safety Statement

This system is an investigational computer science coursework project and clinical decision-support research prototype. It is **not** certified as Software as a Medical Device (SaMD) by the FDA, EMA, or MHRA. It does not provide medical diagnoses or prescribe treatment regimens. In clinical deployment, all automated outputs must be verified by a board-certified ophthalmologist or licensed optometrist.
