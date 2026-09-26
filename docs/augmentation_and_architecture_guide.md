# Section 3 & Section 4 Report Guide: Data Augmentation, Dataset Balancing & CNN Transfer Learning Architecture

This document provides complete, publication-grade academic text, mathematical formulations, and structured parameter tables tailored for **Criterion 3 (Data Augmentation & Dataset Balancing - 10 Marks)** and **Criterion 4 (CNN Architecture & Transfer Learning Implementation - 20 Marks)** of the Computer Vision Coursework.

---

## 1. Data Augmentation & Dataset Balancing (Criterion 3 - 10 Marks)

### 1.1 Clinical & Physiological Justification of Augmentation Transforms

In medical computer vision—and particularly in retinal fundus photography—applying arbitrary, generic data augmentation can inadvertently alter or destroy critical diagnostic features. For example, aggressive elastic distortions can artificially introduce or distort vessel tortuosity, while severe color jitter can obliterate the subtle red hue of microaneurysms. 

In RetinaTrace, all augmentation transforms are implemented in the modular codebase ([`core/augmentation.py`](file:///d:/Computer%20Vision%20CW/core/augmentation.py)) and are strictly constrained by physiological and optical invariants of ophthalmic imaging:

| Transformation | Parameter Range | Physical & Clinical Justification |
|:---|:---:|:---|
| **Arbitrary Planar Rotation** | $[-180^\circ, +180^\circ]$ (Full $360^\circ$) | **Circular Optical Aperture Invariance:** The human retina and the fundus camera aperture exhibit radial circular symmetry. A patient's head tilt or camera rotation changes the apparent angle of the vascular arcade but does not alter the underlying diagnostic DR stage. Full $360^\circ$ planar rotation teaches convolutional kernels rotation-invariant lesion detection. |
| **Horizontal & Vertical Reflection** | $p = 0.50$ (each axis) | **Morphological Directional Invariance:** Unlike thoracic radiographs (where left/right anatomical asymmetry is clinically vital for organ positioning), retinal microaneurysms, blot hemorrhages, and lipid exudates possess no intrinsic directional polarity. Reflecting the image across horizontal and vertical axes quadruples diagnostic variety without introducing non-biological artifacts. |
| **Scale & Working-Distance Zoom** | Scale factor $[0.90, 1.10]$ | **Camera Working-Distance Variability:** In busy screening clinics, working distances between the patient's cornea and the objective lens naturally fluctuate between 30 mm and 50 mm, yielding slight optical magnification differences. Bounded zooming ($\pm 10\%$) simulates this optical variation. |
| **Subtle Photometric Jitter** | Contrast: $[0.85, 1.15]$<br>Brightness: $\pm 15\%$ | **Sensor Calibration & Mydriasis Invariance:** Simulates clinical variations in Xenon flash strobe voltage, camera sensor exposure curves, and differences in pharmacologically induced pupil dilation quality across different screening sites. |
| **Coarse Dropout / Cutout** | $p = 0.30$, 4 holes,<br>size $4\text{--}16$ pixels | **Occlusion & Distributed Representation:** Simulates transient lens dust particles or intermittent eyelash shadowing in the optical path. Forces deep convolutional feature maps to learn distributed spatial patterns of microaneurysms across the entire retinal surface rather than over-relying on a single dominant lesion cluster. |

---

### 1.2 Quantitative Dataset Balancing & Loss Weighting Formulations

As demonstrated in Section 2, the dataset presents an intrinsic epidemiological imbalance: **No DR (Class 0: 34.17%)** and **Moderate NPDR (Class 2: 21.72%)** outnumber **Severe NPDR (Class 3: 14.13%)** and **Proliferative DR (Class 4: 14.67%)**.

To prevent the gradient descent optimization from collapsing toward majority-class predictions, two complementary balancing strategies were formulated and implemented in [`core/augmentation.py`](file:///d:/Computer%20Vision%20CW/core/augmentation.py):

#### Strategy A: Inverse-Frequency Class Weighting
Under this formulation, loss penalties for class $c$ are scaled inversely proportional to their empirical frequencies:

$$w_c = \frac{N}{K \cdot N_c}$$

Where $N = 38,034$ is the total dataset volume, $K = 5$ is the number of diagnostic classes, and $N_c$ is the sample count for class $c$.

*Empirical Class Weights Computed for Training Cohort ($N=26,623$):*
- **Class 0 (No DR, $N_0 = 9,096$):** $w_0 = \frac{26623}{5 \times 9096} = \mathbf{0.585}$
- **Class 1 (Mild NPDR, $N_1 = 4,078$):** $w_1 = \frac{26623}{5 \times 4078} = \mathbf{1.306}$
- **Class 2 (Moderate NPDR, $N_2 = 5,783$):** $w_2 = \frac{26623}{5 \times 5783} = \mathbf{0.921}$
- **Class 3 (Severe NPDR, $N_3 = 3,762$):** $w_3 = \frac{26623}{5 \times 3762} = \mathbf{1.415}$
- **Class 4 (Proliferative DR, $N_4 = 3,904$):** $w_4 = \frac{26623}{5 \times 3904} = \mathbf{1.364}$

#### Strategy B: Effective Number of Samples Formulation (Cui et al., CVPR 2019)
Because highly correlated images (e.g., bilateral captures from the same patient) contribute redundant information, inverse frequency can over-penalize large classes. The Effective Number of Samples formulation scales weights based on the volume of unique feature space occupied:

$$E_{N_c} = \frac{1 - \beta^{N_c}}{1 - \beta}, \quad w_c^{\text{eff}} = \frac{1}{E_{N_c}}$$

Where $\beta = \frac{N - 1}{N} \approx 0.9999$. This mathematical formulation is available in `compute_balanced_class_weights(..., method='effective_samples')`.

---

## 2. CNN Architecture & Transfer Learning Implementation (Criterion 4 - 20 Marks)

### 2.1 Backbone Selection: EfficientNetB3 vs. Traditional Architectures

The selection of **EfficientNetB3** as the primary diagnostic backbone was driven by a principled analysis of accuracy, parameter efficiency, and architectural scaling in clinical settings:

| Metric / Property | VGG16 | ResNet-50 | **EfficientNetB3 (Ours)** | Clinical & Engineering Rationale |
|:---|:---:|:---:|:---:|:---|
| **Parameters (M)** | 138.4 M | 25.6 M | **12.3 M** | **$11.2\times$ smaller than VGG16, $2.1\times$ smaller than ResNet-50.** Prevents severe overfitting on subtle retinal microaneurysms while enabling real-time edge execution in point-of-care clinics. |
| **Top-1 ImageNet Acc** | 71.3% | 76.0% | **81.6%** | Superior feature representation transfer from pre-training. |
| **Scaling Principle** | Single (Depth) | Single (Depth) | **Compound Scaling** | Simultaneously balances network depth ($d$), network width ($w$), and input resolution ($r$) via compound coefficient $\phi$: $d = \alpha^\phi, w = \beta^\phi, r = \gamma^\phi$ subject to $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$. |
| **Attention Mechanism** | None | None | **Squeeze-and-Excitation (SE)** | Every MBConv block incorporates an internal SE channel-attention module that adaptively recalibrates channel-wise feature responses, emphasizing microvascular anomalies over homogeneous background. |
| **Activation Function** | ReLU | ReLU | **Swish ($x \cdot \sigma(x)$)** | Smooth, non-monotonic activation prevents dead neurons during fine-tuning of early layers. |

```
Architecture Flowchart:
[Input 224x224x3] 
       │
       ▼
[EfficientNetB3 Backbone (Pretrained ImageNet, 26 MBConv Blocks)] ──► [Top Activation: 7x7x1536]
       │                                                                      │
       ▼                                                                      ▼
[Global Average Pooling 2D (1536-D)]                                    [Branch 1: Grad-CAM XAI Engine]
       │                                                                (Gradients w.r.t Top Activation)
       ▼                                                                      │
[Batch Normalization]                                                         ▼
       │                                                                [Saliency Attention Heatmap]
       ▼
[Dense Bottleneck (256-D, ReLU)] ──► [Branch 2: CBR Similarity Retrieval]
       │                             (256-D Cosine Embedding vs Gallery)
       ▼
[Dropout (0.30)]
       │
       ▼
[Dense (5, Softmax)] ──────────────► [Primary Classification: Stage 0–4]
                                              │
                                              ▼
                                     [Clinical Governance Agent]
                                     (Confidence Threshold τ = 0.70)
```

---

### 2.2 Comprehensive Hyperparameter Tuning & Two-Phase Training Schedule

To protect the generalized optical feature representations learned on ImageNet while adapting high-level convolutional kernels to subtle microvascular pathologies, a **staged two-phase transfer learning regimen** was executed (centralized in [`core/config.py`](file:///d:/Computer%20Vision%20CW/core/config.py)):

| Hyperparameter / Setting | Phase 1: Feature Extraction (Warmup) | Phase 2: End-to-End Fine-Tuning | Clinical & Algorithmic Rationale |
|:---|:---:|:---:|:---|
| **Backbone State** | **Frozen** (384 layers locked) | **Partially Unfrozen** (Top 65 layers active: `block7`, `block6`, `top_conv`) | Prevents catastrophic forgetting of ImageNet low-level edge filters during initial head stabilization; unfreezing top blocks allows specialized adaptation to retinal lesions. |
| **Learning Rate ($\eta$)** | $\eta_1 = 1.0 \times 10^{-3}$ | $\eta_2 = 1.0 \times 10^{-4} \to 1.0 \times 10^{-6}$ | High initial rate trains randomly initialized dense head; lower rate with cosine annealing gently updates pre-trained weights without gradient explosions. |
| **Optimizer** | Adam ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-7}$) | Adam with Cosine Learning Rate Decay | Adaptive moment estimation ensures stable directional convergence across sparse retinal lesion gradients. |
| **Batch Size** | 32 | 16 | Reduced batch size in Phase 2 accommodates expanded gradient storage for unfrozen convolutional stages. |
| **Epochs** | 10 | 25 | Total 35 epochs; convergence validated via held-out validation loss. |
| **Dropout Rate** | $p = 0.30$ (after 256-D Bottleneck) | $p = 0.30$ | Enforces co-adaptation resistance among dense feature units. |
| **Weight Decay ($L_2$)** | $1.0 \times 10^{-4}$ | $1.0 \times 10^{-4}$ | Constrains $L_2$ weight norms to prevent overfitting on minority class features. |
| **Label Smoothing** | $\epsilon_{\text{smooth}} = 0.05$ | $\epsilon_{\text{smooth}} = 0.05$ | Replaces one-hot targets $y \in \{0, 1\}$ with $y_{\text{smooth}} = y(1 - \epsilon) + \frac{\epsilon}{K}$, penalizing over-confident predictions on ambiguous borderline fundus images. |
| **Early Stopping** | Patience = 5 epochs (monitoring `val_loss`) | Patience = 5 epochs (monitoring `val_loss`) | Restores best checkpoint weights automatically upon validation degradation. |
| **Learning Rate Reduction** | Factor = 0.50, Patience = 2 epochs | Factor = 0.50, Patience = 2 epochs | Reduces learning rate when plateaus are encountered. |

---

### 2.3 Four-Level Deep Clinical U-Net Architecture (`build_deep_clinical_unet`)

To provide genuine multi-scale lesion segmentation rather than simple morphological heuristics, the architecture incorporates a **4-level deep encoder-decoder U-Net** implemented in [`core/models.py`](file:///d:/Computer%20Vision%20CW/core/models.py):

1. **4-Stage Hierarchical Encoder:**
   - **Level 1 ($224 \times 224$):** Double Conv ($3 \times 3$, 32 filters) + BatchNorm + ReLU $\to$ MaxPool ($2 \times 2$) $\to 112 \times 112$.
   - **Level 2 ($112 \times 112$):** Double Conv ($3 \times 3$, 64 filters) + BatchNorm + ReLU $\to$ MaxPool ($2 \times 2$) $\to 56 \times 56$.
   - **Level 3 ($56 \times 56$):** Double Conv ($3 \times 3$, 128 filters) + BatchNorm + ReLU $\to$ MaxPool ($2 \times 2$) $\to 28 \times 28$.
   - **Level 4 ($28 \times 28$):** Double Conv ($3 \times 3$, 256 filters) + BatchNorm + ReLU $\to$ MaxPool ($2 \times 2$) $\to 14 \times 14$.
2. **Latent Bottleneck ($14 \times 14$):**
   - Double Conv ($3 \times 3$, 512 filters) + Spatial Dropout ($p = 0.40$). Captures contextual multi-quadrant lesion relationships at the coarsest spatial resolution.
3. **4-Stage Transposed Convolution Decoder with Multi-Scale Skip Connections:**
   - At each decoder level, feature maps are upsampled via $2 \times 2$ Transposed Convolutions (strides=2) and **concatenated directly with the corresponding high-resolution encoder feature map** along the channel dimension.
   - Preserves sub-millimeter lesion boundary localization (vital for detecting single microaneurysms) that would otherwise be permanently lost through encoder pooling operations.
4. **Output Head ($224 \times 224 \times 1$):**
   - $1 \times 1$ Convolution with Sigmoid activation producing a calibrated pixel-level probability map $\hat{M}(x, y) \in [0.0, 1.0]$ representing active retinal lesion area.
   - Total model parameters: **7,771,873** trainable parameters.
