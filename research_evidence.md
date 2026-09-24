# Research Evidence Protocol

This protocol is separate from the deployed `app.py` path. It does not replace
the production EfficientNetB3 checkpoint.

## Ablation experiments

Run the five variants from the notebook using the same patient-level split,
random seed, test set, batch size, and evaluation code:

```python
from core.research_evidence import ABLATION_VARIANTS, run_ablation_suite

def train_and_predict(variant):
    # Build a fresh EfficientNetB3 model for this variant in the notebook.
    # Return y_true and y_pred from the unchanged held-out test set.
    ...

ablation_results = run_ablation_suite(
    train_and_predict,
    "report_images/ablation_results.csv",
)
```

The callback must create a new experimental checkpoint for each variant. It
must never overwrite `checkpoints/best_phase2.weights.h5`.

The variants are:

- `full_system`
- `without_augmentation`
- `without_class_weights`
- `without_ben_graham`
- `frozen_efficientnet`

The output reports accuracy, macro F1, and quadratic weighted kappa. Do not
fill the result template with invented numbers; populate it only after the
experiments have executed.

## Error analysis

After the final model has generated `y_true`, `y_pred`, and test
probabilities, run:

```python
from core.research_evidence import build_error_analysis, save_error_analysis

analysis = build_error_analysis(y_true, y_pred, probabilities=test_probabilities)
save_error_analysis(analysis, "report_images/error_analysis")
```

This produces:

- `error_cases.csv`
- `error_analysis_summary.json`

The summary includes Mild-to-Moderate errors, Moderate-to-Severe errors,
adjacent versus distant-stage error rates, high-confidence errors, and
low-confidence cases suitable for Governance Gate screenshots. Add quality and
explainability records by passing aligned lists through `quality_records` and
`explainability_records`.

## Reproducibility manifest

Create the manifest after training:

```python
from core.research_evidence import build_reproducibility_manifest

build_reproducibility_manifest(
    config={
        "random_seed": 42,
        "image_size": [224, 224],
        "batch_size": 32,
        "phase_1_learning_rate": 1e-3,
        "phase_2_learning_rate": 1e-5,
        "phase_1_max_epochs": 15,
        "phase_2_max_epochs": 25,
        "confidence_threshold": 0.70,
    },
    output_path="report_images/reproducibility_manifest.json",
    dataset_version="Combined DR Dataset Kaggle release used for this run",
    split_files=[
        "report_images/train_split.csv",
        "report_images/validation_split.csv",
        "report_images/test_split.csv",
    ],
    checkpoint_files=[
        "checkpoints/best_phase1.weights.h5",
        "checkpoints/best_phase2.weights.h5",
        "checkpoints/unet_lesion_best.weights.h5",
    ],
    training_duration_seconds=None,
)
```

The manifest records Python, TensorFlow, platform, GPU devices, Git commit,
dataset version, split files, checkpoints, seed, hyperparameters, and training
duration. Replace `None` with a measured duration when available.