"""Training strategy, optimization schedules, and clinical evaluation callbacks.

Provides reproducible training routines for two-phase transfer learning on EfficientNetB3,
including custom callbacks for Quadratic Weighted Kappa (QWK) tracking and overfitting prevention.

Two-Phase Training Protocol (matches diabetic_retinopathy_detection.ipynb):
  Phase 1: Frozen backbone — train head only at lr=1e-3 for 15 epochs.
  Phase 2: Top-30 layers unfrozen — fine-tune at lr=1e-5 for 25 epochs.
  Both phases use label_smoothing=0.1 (AppConfig.LABEL_SMOOTHING) and class_weight dict.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks

from core.config import AppConfig


def compile_for_phase1(model: keras.Model) -> None:
    """Compile model for Phase 1 (frozen backbone, head-only training).

    Uses Adam at AppConfig.PHASE1_LR (1e-3) with CategoricalCrossentropy
    and label_smoothing=AppConfig.LABEL_SMOOTHING (0.1). Matches notebook
    compile_model_phase1() exactly.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=AppConfig.PHASE1_LR),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=AppConfig.LABEL_SMOOTHING
        ),
        metrics=["accuracy"],
    )


def compile_for_phase2(model: keras.Model) -> None:
    """Compile model for Phase 2 (top-30 layers unfrozen, domain fine-tuning).

    Uses Adam at AppConfig.PHASE2_LR (1e-5) with the same label smoothing.
    Matches notebook Phase 2 recompile step exactly.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=AppConfig.PHASE2_LR,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
        ),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=AppConfig.LABEL_SMOOTHING
        ),
        metrics=["accuracy"],
    )


def unfreeze_top_n_layers(
    base_model: keras.Model,
    n: int = AppConfig.PHASE2_UNFROZEN_LAYERS,
) -> None:
    """Unfreeze the top-N layers of the EfficientNetB3 backbone for Phase 2.

    Freezes all layers except the final ``n`` layers of the base model.
    Default n=AppConfig.PHASE2_UNFROZEN_LAYERS=30, matching the notebook's
    ``Config.UNFREEZE_TOP_N = 30`` setting.
    """
    base_model.trainable = True
    freeze_until = len(base_model.layers) - n
    for i, layer in enumerate(base_model.layers):
        layer.trainable = i >= freeze_until



def compute_qwk(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5) -> float:
    """Compute Quadratic Weighted Kappa (QWK) between ground truth and predicted ordinal stages."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    
    # Compute confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1.0
            
    total = np.sum(cm)
    if total == 0:
        return 0.0
        
    # Quadratic weight matrix: w_ij = (i - j)^2 / (K - 1)^2
    weights = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i in range(num_classes):
        for j in range(num_classes):
            weights[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)
            
    # Expected matrix under independent marginal chance
    row_sums = np.sum(cm, axis=1)
    col_sums = np.sum(cm, axis=0)
    expected = np.outer(row_sums, col_sums) / total
    
    # Kappa calculation
    numerator = np.sum(weights * cm)
    denominator = np.sum(weights * expected)
    
    if denominator == 0:
        return 1.0
        
    return float(1.0 - (numerator / denominator))


class QWKEvaluationCallback(callbacks.Callback):
    """Custom Keras callback monitoring Quadratic Weighted Kappa on validation data.
    
    Computes true multi-class QWK after each epoch and automatically archives
    the checkpoint with the highest validation QWK score.
    """

    def __init__(
        self,
        val_data: Tuple[np.ndarray, np.ndarray],
        checkpoint_path: Optional[str] = None,
        num_classes: int = 5,
        verbose: int = 1,
    ):
        super().__init__()
        self.val_x, self.val_y = val_data
        self.checkpoint_path = checkpoint_path
        self.num_classes = num_classes
        self.verbose = verbose
        self.best_qwk = -1.0

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        preds = self.model.predict(self.val_x, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        
        # If one-hot encoded, convert to integer labels
        y_true = self.val_y if self.val_y.ndim == 1 else np.argmax(self.val_y, axis=1)
        
        current_qwk = compute_qwk(y_true, y_pred, num_classes=self.num_classes)
        
        if logs is not None:
            logs["val_qwk"] = current_qwk
            
        if current_qwk > self.best_qwk:
            if self.verbose:
                print(f"\n[QWK] Epoch {epoch + 1}: val_qwk improved from {self.best_qwk:.4f} to {current_qwk:.4f}.")
            self.best_qwk = current_qwk
            if self.checkpoint_path:
                self.model.save_weights(self.checkpoint_path)
                if self.verbose:
                    print(f"[QWK] Model weights saved to {self.checkpoint_path}")
        else:
            if self.verbose:
                print(f"\n[QWK] Epoch {epoch + 1}: val_qwk = {current_qwk:.4f} (best: {self.best_qwk:.4f})")


def create_clinical_callbacks(
    checkpoint_dir: str = "checkpoints",
    model_name: str = "retinatrace_model",
    val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    patience_early_stopping: int = AppConfig.EARLY_STOPPING_PATIENCE,
    patience_reduce_lr: int = AppConfig.REDUCE_LR_PATIENCE,
    reduce_lr_factor: float = AppConfig.REDUCE_LR_FACTOR,
) -> List[callbacks.Callback]:
    """Assemble a standard clinical training callback suite.
    
    Includes:
    1. EarlyStopping (monitors val_loss, restores best weights)
    2. ReduceLROnPlateau (halves learning rate on stalled progress)
    3. ModelCheckpoint (preserves best model on validation loss)
    4. CSVLogger (records per-epoch training metrics for auditable provenance)
    5. QWKEvaluationCallback (optional, tracks quadratic weighted kappa)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_weights_path = os.path.join(checkpoint_dir, f"{model_name}_best.weights.h5")
    log_csv_path = os.path.join(checkpoint_dir, f"{model_name}_training_log.csv")

    callback_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_early_stopping,
            restore_best_weights=True,
            mode="min",
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=reduce_lr_factor,
            patience=patience_reduce_lr,
            min_lr=AppConfig.PHASE2_MIN_LR,
            mode="min",
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=best_weights_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            mode="min",
            verbose=1,
        ),
        callbacks.CSVLogger(
            filename=log_csv_path,
            separator=",",
            append=False,
        ),
    ]

    if val_data is not None:
        qwk_checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_best_qwk.weights.h5")
        callback_list.append(
            QWKEvaluationCallback(
                val_data=val_data,
                checkpoint_path=qwk_checkpoint_path,
                num_classes=AppConfig.NUM_CLASSES,
            )
        )

    return callback_list
