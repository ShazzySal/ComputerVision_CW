"""Application configuration for RetinaTrace AI."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AppConfig:
    """Centralized configuration parameters for RetinaTrace AI."""

    IMG_SIZE: int = 224
    NUM_CLASSES: int = 5
    CLASS_NAMES: list = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

    BEN_GRAHAM_SIGMA: int = 10
    BEN_GRAHAM_ALPHA: float = 4.0
    BEN_GRAHAM_BETA: float = -4.0
    BEN_GRAHAM_GAMMA: float = 128.0
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.70
    ENABLE_OPTIC_DISC_REMOVAL: bool = False

    # Advanced Preprocessing Configuration
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_GRID_SIZE: tuple = (8, 8)
    EDGE_SHARPEN_STRENGTH: float = 1.2
    EDGE_SHARPEN_SIGMA: float = 3.0

    # ── Centralized Training & Hyperparameter Configuration ─────────
    # Phase 1: Transfer Learning Feature Extraction (Frozen Backbone)
    PHASE1_EPOCHS: int = 15
    PHASE1_LR: float = 1e-3
    PHASE1_BATCH_SIZE: int = 32
    PHASE1_OPTIMIZER: str = "Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7)"
    PHASE1_FROZEN_LAYERS: int = 384  # Full EfficientNetB3 backbone (verified layer count)

    # Phase 2: End-to-End Fine-Tuning (Top Stages Unfrozen)
    PHASE2_EPOCHS: int = 25
    PHASE2_LR: float = 1e-5
    PHASE2_MIN_LR: float = 1e-7
    PHASE2_BATCH_SIZE: int = 32
    PHASE2_OPTIMIZER: str = "Adam(learning_rate=1e-5)"
    PHASE2_UNFROZEN_LAYERS: int = 30  # Matches the notebook's current fine-tuning policy

    # Regularization & Optimization Guardrails
    DROPOUT_RATE: float = 0.30
    # Reserved; the notebook training path currently does not apply weight decay.
    L2_WEIGHT_DECAY: float = 1e-4
    LABEL_SMOOTHING: float = 0.1
    EARLY_STOPPING_PATIENCE: int = 5
    REDUCE_LR_PATIENCE: int = 3
    REDUCE_LR_FACTOR: float = 0.50

    WEIGHTS_PATH: str = str(PROJECT_ROOT / "checkpoints" / "best_phase2.weights.h5")
    UNET_WEIGHTS_PATH: str = str(PROJECT_ROOT / "checkpoints" / "unet_lesion_best.weights.h5")
    EMBEDDINGS_PATH: str = str(PROJECT_ROOT / "embeddings.npz")
