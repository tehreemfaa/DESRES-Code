# Classifier Evaluation: Calibration, PCA, and t-SNE

`analysis.py` is the post-training evaluation module from the classification training repository. After the best checkpoint is evaluated, one forward pass over the test set collects probabilities and backbone embeddings. Every figure and table is derived from that single pass.

## Problem

Accuracy and a confusion matrix do not show whether a model's confidence can be trusted, which classes it finds ambiguous, or whether its feature space separates the classes. These questions matter when predicted probabilities are used to filter or sort data downstream, as in [../image-classification](../image-classification/README.md).

## Methods

### Feature capture

`collect_test_predictions` registers a forward hook to capture backbone features without changing the model. If the classifier head is an `nn.Sequential` that contains an `nn.Flatten`, the hook reads the output of that layer. This covers ConvNeXt and EfficientNet heads. Otherwise a forward pre-hook reads the input to the classifier, which covers ResNet, DenseNet, and Swin. The features are discarded if their row count does not match the number of predictions.

### PCA (`_pca_reduce`)

The data are centred, and the economy SVD `X_c = U S V^T` is computed. The data are then projected onto the first k rows of `V^T`. This avoids forming the covariance matrix `X_c^T X_c`, which would square the condition number. t-SNE uses this step to reduce the inputs to 50 dimensions.

### t-SNE (`_compute_joint_probabilities`, `_tsne_numpy`)

This is an exact O(N²) implementation in NumPy. It follows van der Maaten and Hinton (2008).

1. After PCA, each of the 50 components is standardised to unit variance. Pairwise squared distances are computed with the expansion `|x|² + |y|² − 2x·y`, clamped at zero.
2. **Perplexity calibration.** Each point i needs a precision β_i such that the entropy of P(·|i) equals log₂(perplexity). The bisection runs on all N rows at once as array operations. A row's β doubles or halves until a bracket is found, then bisects. The loop runs at most 50 steps and stops early when every row is within 1e-5 bits. Self-affinities are excluded by setting the diagonal distance to infinity. Exponents are clipped to [−500, 0] so the exponentials stay finite and nonzero, and row sums are floored at 1e-12.
3. P is symmetrised as `(P + Pᵀ)/2`, normalised, and multiplied by 4 for early exaggeration.
4. **Optimisation.** The gradient is `4 Σ_j (p_ij − q_ij)(1 + |y_i − y_j|²)⁻¹ (y_i − y_j)`, evaluated with `einsum`. Updates use momentum 0.5 for the first 20 iterations and 0.8 afterwards, with per-parameter adaptive gains. A gain grows by 0.2 when the gradient and the previous update have opposite signs, so the step keeps its direction. Otherwise it shrinks by a factor of 0.8. Gains are floored at 0.01. Exaggeration is removed at iteration 100. The embedding is re-centred after every step.

When there are more than 1500 test images, a seeded random subset of 1500 is embedded.

### Calibration (`_compute_ece`, `plot_calibration_curve`)

Predictions are split into 10 equal-width confidence bins, with the right edge included only in the last bin. Expected calibration error is `ECE = Σ_b (|B_b|/N) · |acc(B_b) − conf(B_b)|`. The reliability diagram draws each bin at its mean confidence and marks its gap from the diagonal.

### Other outputs

| Output | Content |
| --- | --- |
| `test_predictions.csv` | One row per image with the full probability vector |
| `per_class_precision_recall_f1.png` | Precision, recall, and F1 from TP, FP, and FN counts, with zero-division guards |
| `class_probability_heatmap.png` | Mean probability vector for each true class, which shows confusion structure that the argmax hides |
| `confidence_by_class.png`, `confidence_boxplots.png`, `confidence_correct_vs_incorrect.png` | Confidence distributions |
| `top_confident_mistakes.png`, `most_uncertain_predictions.png` | Image grids of the 25 most confident errors and the 25 least confident predictions |
| `feature_space_tsne.png` | t-SNE embedding coloured by true class |
| `error_analysis.json` | Confusion percentages, per-class accuracy, the most confused pair, ECE |

For losses whose name marks them as sigmoid, `bce_sigmoid` and `focal_loss_sigmoid`, probabilities come from an element-wise sigmoid. Confidence is then the largest sigmoid output, and the values do not sum to 1.

## Dependencies

- numpy, matplotlib, torch, pillow.
- `collect_test_predictions` runs `from .model_utils import _get_classifier_attr`. The module therefore has to sit inside the training repository's `utils` package, next to `model_utils.py`, which is not included here.
- `run_analysis_pipeline` expects a trained model, an `ImageFolder` test `DataLoader` with `shuffle=False`, and a CUDA device for `autocast("cuda")`.

The numerical routines use only NumPy and can be called directly. Importing the module still loads torch, matplotlib, and Pillow, because they are imported at module level.

```python
import numpy as np
from analysis import _tsne_numpy, _compute_ece

Y = _tsne_numpy(np.random.default_rng(0).standard_normal((500, 128)), verbose=False)
ece, bins = _compute_ece(np.array([0.9, 0.6, 0.8]), np.array([1.0, 0.0, 1.0]))
```

## Tests

This module has no automated tests. The review for this repository ran the following checks against the unmodified code:

- `import analysis` succeeds.
- The symmetrised P matrix sums to 4, as expected after exaggeration.
- Bandwidth bisection converges for a synthetic outlier 2000 squared-distance units from every other point.
- `plot_confidence_boxplots` raises `TypeError` on Matplotlib 3.11, because `Axes.boxplot` no longer accepts `labels=`. Use Matplotlib 3.10 or earlier.

The t-SNE output has not been compared numerically with a reference implementation such as scikit-learn.

## Limitations

- Exact t-SNE needs O(N²) memory and time. The `(N, N, 2)` difference tensor is the largest allocation. This is why the input is capped at 1500 samples.
- The 300-iteration default leaves 200 iterations after exaggeration ends, which is shorter than common defaults.
- Standardising PCA components to unit variance gives low-variance directions the same weight as dominant ones. This is a modelling choice that affects the embedding.
- The colour palette has six entries. `zip` with the palette truncates silently, so with more than six classes the extra classes are missing from the t-SNE scatter.
