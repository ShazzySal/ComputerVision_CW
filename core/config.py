"""Application configuration for RetinaGuard AI."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AppConfig:
    """Centralized configuration parameters for RetinaGuard AI."""

    IMG_SIZE: int = 224
    NUM_CLASSES: int = 5
    CLASS_NAMES: list = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

    BEN_GRAHAM_SIGMA: int = 10
    BEN_GRAHAM_ALPHA: float = 4.0
    BEN_GRAHAM_BETA: float = -4.0
    BEN_GRAHAM_GAMMA: float = 128.0
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.70

    WEIGHTS_PATH: str = str(PROJECT_ROOT / "checkpoints" / "best_phase2.weights.h5")
    UNET_WEIGHTS_PATH: str = str(PROJECT_ROOT / "checkpoints" / "unet_lesion_best.weights.h5")
    EMBEDDINGS_PATH: str = str(PROJECT_ROOT / "embeddings.npz")
