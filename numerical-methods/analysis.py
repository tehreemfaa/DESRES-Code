"""
utils/analysis.py

Deep evaluation analysis pipeline for image classification experiments.
[Research programme name redacted for publication.]

After the best model is evaluated on the test set, this module generates:
  - test_predictions.csv          one row per test image with full probability vector
  - confidence_by_class.png       confidence histograms per true class
  - confidence_correct_vs_incorrect.png  overlay: correct vs wrong predictions
  - class_probability_heatmap.png mean softmax output matrix (true × predicted)
  - per_class_precision_recall_f1.png   grouped bar chart
  - confidence_boxplots.png       boxplots of confidence grouped by true class
  - calibration_curve.png         reliability diagram + ECE
  - top_confident_mistakes.png    5×5 grid of highest-confidence wrong predictions
  - most_uncertain_predictions.png 5×5 grid of lowest-confidence predictions
  - feature_space_tsne.png        t-SNE of backbone embeddings (pure numpy)
  - error_analysis.json           structured summary statistics

Called from utils/trainer.train_model() after confusion matrix generation,
while test_loader is still alive.
"""

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from PIL import Image


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

# Colorblind-friendly IBM palette used throughout the project
COLORS = ["#648FFF", "#DC267F", "#FFB000", "#FE6100", "#785EF0", "#009E73"]

PLOT_STYLE = {
    "font.family":       "serif",
    "font.size":         10,
    "axes.labelsize":    11,
    "axes.titlesize":    12,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "figure.dpi":        100,
    "savefig.dpi":       150,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save(fig, path: Path):
    """Save figure to path and print confirmation."""
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Analysis] Saved: {path}")


def _load_image_for_display(path: str, size: int = 112) -> np.ndarray:
    """Load an image from disk and return as (H, W, 3) uint8 numpy array."""
    try:
        img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
        return np.array(img)
    except Exception:
        # Return a grey placeholder if the file cannot be read
        placeholder = np.full((size, size, 3), 180, dtype=np.uint8)
        return placeholder


# ---------------------------------------------------------------------------
# Data Collection  
# ---------------------------------------------------------------------------

def collect_test_predictions(model, test_loader, class_names, device,
                             loss_fn="cross_entropy", backbone_name=None):
    """
    Run one forward pass on the test set and collect per-image data.

    Also extracts backbone features for t-SNE via a forward hook:
      - ConvNeXt / architectures with a Flatten inside the classifier Sequential:
        hook is placed on that Flatten module → 1024-dim flattened backbone vector.
      - ResNet / DenseNet / Swin / architectures with a direct Linear classifier:
        forward_pre_hook on the classifier captures the backbone output directly.

    Args:
        model:          Best-checkpoint model (already in eval mode).
        test_loader:    DataLoader with shuffle=False over an ImageFolder dataset.
        class_names:    List[str] of class names in dataset-index order.
        device:         torch.device.
        loss_fn:        Determines softmax vs sigmoid conversion.
        backbone_name:  Architecture key used to locate the classifier attribute.

    Returns:
        records  (list[dict]): One dict per image (see keys below).
        features (np.ndarray | None): [N, D] backbone feature matrix, or None.

    Record keys:
        image_path, true_label, predicted_label, correct (bool),
        confidence (float), true_idx (int), pred_idx (int),
        probs (np.ndarray shape [C]),
        <cls>_probability × n_classes
    """
    from .model_utils import _get_classifier_attr

    model.eval()
    dataset  = test_loader.dataset
    # ImageFolder.samples = [(abs_path, class_idx), ...]
    samples  = getattr(dataset, "samples", None)

    # ── Feature extraction hook ──────────────────────────────────────────────
    feat_buffer  = []
    hook_handle  = None

    if backbone_name is not None:
        try:
            classifier_attr = _get_classifier_attr(backbone_name)
            classifier      = getattr(model, classifier_attr)

            # Prefer hooking the Flatten layer if it exists inside the classifier
            # Sequential (ConvNeXt, EfficientNet).  This gives the flattened
            # backbone representation after global average pooling.
            flatten_module = None
            if isinstance(classifier, nn.Sequential):
                for submod in classifier:
                    if isinstance(submod, nn.Flatten):
                        flatten_module = submod
                        break

            if flatten_module is not None:
                # Hook output of Flatten → [B, 1024] for ConvNeXt Base
                def _hook_flat(mod, inp, out):
                    feat_buffer.append(out.detach().cpu().float())
                hook_handle = flatten_module.register_forward_hook(_hook_flat)
            else:
                # Hook input of the whole classifier (ResNet, Swin, DenseNet…)
                # inp[0] is already the flattened backbone vector [B, D]
                def _pre_hook(mod, inp):
                    if inp and inp[0] is not None:
                        x = inp[0].detach().cpu().float()
                        feat_buffer.append(x.view(x.size(0), -1))
                hook_handle = classifier.register_forward_pre_hook(_pre_hook)

        except Exception as exc:
            print(f"  [Analysis] Feature hook failed ({exc}); t-SNE will be skipped.")

    # ── Inference loop ───────────────────────────────────────────────────────
    all_probs      = []
    all_true_idxs  = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            with autocast("cuda"):
                logits = model(images)

            if loss_fn in ("bce_sigmoid", "focal_loss_sigmoid"):
                probs = torch.sigmoid(logits)
            else:
                probs = torch.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_true_idxs.extend(labels.tolist())

    if hook_handle is not None:
        hook_handle.remove()

    all_probs = np.concatenate(all_probs, axis=0)   # [N, C]
    n_samples = all_probs.shape[0]

    # ── Consolidate feature buffer ───────────────────────────────────────────
    features = None
    if feat_buffer:
        try:
            features = torch.cat(feat_buffer, dim=0).numpy()  # [N, D]
            if features.shape[0] != n_samples:
                features = None   # mismatch — discard
        except Exception:
            features = None

    # ── Build per-image records ──────────────────────────────────────────────
    class_names_lower = [c.lower() for c in class_names]
    records = []

    for i in range(n_samples):
        true_idx = all_true_idxs[i]
        pred_idx = int(all_probs[i].argmax())

        rec = {
            "image_path":      samples[i][0] if samples else f"sample_{i}",
            "true_label":      class_names[true_idx],
            "predicted_label": class_names[pred_idx],
            "correct":         (true_idx == pred_idx),
            "confidence":      float(all_probs[i].max()),
            "true_idx":        true_idx,
            "pred_idx":        pred_idx,
            "probs":           all_probs[i],   # kept in memory for heatmap / t-SNE
        }
        for j, cls in enumerate(class_names_lower):
            rec[f"{cls}_probability"] = float(all_probs[i, j])

        records.append(rec)

    return records, features


# ---------------------------------------------------------------------------
# 0. test_predictions.csv
# ---------------------------------------------------------------------------

def save_test_predictions_csv(records, class_names, results_dir: Path):
    """
    Write test_predictions.csv — one row per test image.

    Columns:
        image_path, true_label, predicted_label, correct, confidence,
        <cls>_probability × n_classes
    """
    out_path = results_dir / "test_predictions.csv"
    class_names_lower = [c.lower() for c in class_names]

    fieldnames = (
        ["image_path", "true_label", "predicted_label", "correct", "confidence"]
        + [f"{c}_probability" for c in class_names_lower]
    )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    print(f"  [Analysis] Saved: {out_path}  ({len(records):,} rows)")


# ---------------------------------------------------------------------------
# 1. Confidence Distribution by True Class
# ---------------------------------------------------------------------------

def plot_confidence_by_class(records, class_names, results_dir: Path):
    """
    One histogram per true class showing the distribution of max-softmax
    confidence for all images of that class.

    Low-confidence distributions indicate classes that are inherently harder
    to classify or that overlap with other classes in feature space.
    """
    n_classes = len(class_names)
    ncols = min(n_classes, 4)
    nrows = (n_classes + ncols - 1) // ncols

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5 * ncols, 4 * nrows),
                                 squeeze=False)
        axes_flat = axes.flatten()

        for i, cls in enumerate(class_names):
            ax     = axes_flat[i]
            color  = COLORS[i % len(COLORS)]
            confs  = [r["confidence"] for r in records if r["true_label"] == cls]

            if confs:
                ax.hist(confs, bins=20, color=color,
                        edgecolor="white", linewidth=0.5, alpha=0.85)
                mean_c = float(np.mean(confs))
                ax.axvline(mean_c, color="black", linestyle="--",
                           linewidth=1.3, label=f"mean = {mean_c:.3f}")
                ax.legend(fontsize=8)

            ax.set_title(f"{cls.capitalize()}  (n = {len(confs):,})",
                         fontweight="bold")
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Count")
            ax.set_xlim(0, 1)

        for j in range(n_classes, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle("Prediction Confidence by True Class",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])

    _save(fig, results_dir / "confidence_by_class.png")


# ---------------------------------------------------------------------------
# 2. Confidence: Correct vs Incorrect Predictions
# ---------------------------------------------------------------------------

def plot_correct_vs_incorrect_confidence(records, results_dir: Path):
    """
    Overlaid histograms for correct and incorrect predictions.

    High-confidence errors (right side of the incorrect histogram) indicate
    a calibration problem — the model is confidently wrong.
    Dashed vertical lines mark the mean confidence for each group.
    """
    correct_conf   = [r["confidence"] for r in records if     r["correct"]]
    incorrect_conf = [r["confidence"] for r in records if not r["correct"]]

    bins = np.linspace(0, 1, 26)

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.hist(correct_conf, bins=bins, alpha=0.65, color="#2CA02C",
                edgecolor="white", linewidth=0.5,
                label=f"Correct   (n = {len(correct_conf):,})")
        ax.hist(incorrect_conf, bins=bins, alpha=0.65, color="#D62728",
                edgecolor="white", linewidth=0.5,
                label=f"Incorrect (n = {len(incorrect_conf):,})")

        if correct_conf:
            ax.axvline(np.mean(correct_conf), color="#2CA02C",
                       linestyle="--", linewidth=1.4, alpha=0.9)
        if incorrect_conf:
            ax.axvline(np.mean(incorrect_conf), color="#D62728",
                       linestyle="--", linewidth=1.4, alpha=0.9)

        ax.set_xlabel("Prediction Confidence (max softmax probability)")
        ax.set_ylabel("Count")
        ax.set_title("Confidence Distribution: Correct vs Incorrect Predictions",
                     fontweight="bold")
        ax.legend()
        plt.tight_layout()

    _save(fig, results_dir / "confidence_correct_vs_incorrect.png")


# ---------------------------------------------------------------------------
# 3. Class Probability Heatmap
# ---------------------------------------------------------------------------

def plot_class_probability_heatmap(records, class_names, results_dir: Path):
    """
    For each true class, compute the average predicted softmax probability
    assigned to every class and display as a labelled heatmap.

    Rows  = true class label
    Cols  = target class receiving probability mass

    Reveals confusion structure beyond argmax:  a row that spreads probability
    across two columns indicates a genuinely ambiguous class boundary, even when
    the argmax is still correct.
    """
    n = len(class_names)
    matrix = np.zeros((n, n), dtype=np.float64)

    for cls_idx, cls in enumerate(class_names):
        cls_recs = [r for r in records if r["true_label"] == cls]
        if not cls_recs:
            continue
        prob_rows = np.stack([r["probs"] for r in cls_recs], axis=0)  # [K, C]
        matrix[cls_idx] = prob_rows.mean(axis=0)

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(7, 6))

        im   = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Mean Predicted Probability", fontsize=10)

        tick_labels = [c.capitalize() for c in class_names]
        ax.set_xticks(range(n))
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels(tick_labels)
        ax.set_xlabel("Probability Assigned To")
        ax.set_ylabel("True Class")
        ax.set_title("Class Probability Heatmap\n"
                     "(average softmax output per true class)",
                     fontweight="bold")

        for row in range(n):
            for col in range(n):
                val   = matrix[row, col]
                tcolor = "white" if val > 0.55 else "black"
                ax.text(col, row, f"{val:.3f}",
                        ha="center", va="center",
                        fontsize=9, color=tcolor, fontweight="bold")

        plt.tight_layout()

    _save(fig, results_dir / "class_probability_heatmap.png")


# ---------------------------------------------------------------------------
# 4. Per-Class Precision / Recall / F1 Bar Chart
# ---------------------------------------------------------------------------

def plot_per_class_metrics_bar(records, class_names, results_dir: Path):
    """
    Compute TP / FP / FN from per-image records and draw a grouped bar chart
    of precision, recall, and F1 for each class.

    Values are computed directly from the prediction records and therefore
    match what would be obtained from a confusion matrix.
    """
    n  = len(class_names)
    tp = np.zeros(n, dtype=float)
    fp = np.zeros(n, dtype=float)
    fn = np.zeros(n, dtype=float)

    for r in records:
        ti = r["true_idx"]
        pi = r["pred_idx"]
        if ti == pi:
            tp[ti] += 1
        else:
            fp[pi] += 1
            fn[ti] += 1

    prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    rec  = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    f1   = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)

    x = np.arange(n)
    w = 0.26   # bar width

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))

        b_p = ax.bar(x - w, prec, width=w, label="Precision",
                     color="#1f77b4", alpha=0.85, edgecolor="white")
        b_r = ax.bar(x,     rec,  width=w, label="Recall",
                     color="#ff7f0e", alpha=0.85, edgecolor="white")
        b_f = ax.bar(x + w, f1,   width=w, label="F1 Score",
                     color="#2ca02c", alpha=0.85, edgecolor="white")

        for bars in (b_p, b_r, b_f):
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2,
                        h + 0.012, f"{h:.3f}",
                        ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in class_names])
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.14)
        ax.set_title("Per-Class Precision / Recall / F1", fontweight="bold")
        ax.legend(loc="lower right")
        plt.tight_layout()

    _save(fig, results_dir / "per_class_precision_recall_f1.png")


# ---------------------------------------------------------------------------
# 5. Confidence Boxplots by True Class
# ---------------------------------------------------------------------------

def plot_confidence_boxplots(records, class_names, results_dir: Path):
    """
    Boxplot of prediction confidence grouped by true class.

    Shows median, IQR, whiskers, and outliers.  Individual data points are
    jitter-overlaid for classes with ≤ 200 images.
    """
    data_by_class = [
        [r["confidence"] for r in records if r["true_label"] == cls]
        for cls in class_names
    ]

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))

        bp = ax.boxplot(
            data_by_class,
            labels=[c.capitalize() for c in class_names],
            patch_artist=True,
            notch=False,
            medianprops=dict(color="black", linewidth=2),
            flierprops=dict(marker="o", markersize=3, alpha=0.4),
        )

        for patch, color in zip(bp["boxes"], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)

        # Jitter-overlay for small classes
        rng = np.random.default_rng(42)
        for i, (cls_data, color) in enumerate(zip(data_by_class, COLORS)):
            if 0 < len(cls_data) <= 200:
                jitter = rng.uniform(-0.12, 0.12, len(cls_data))
                ax.scatter(
                    np.full(len(cls_data), i + 1) + jitter, cls_data,
                    alpha=0.30, s=8, color=color, zorder=3,
                )

        ax.set_ylabel("Prediction Confidence")
        ax.set_title("Confidence Boxplots by True Class", fontweight="bold")
        ax.set_ylim(-0.02, 1.05)
        plt.tight_layout()

    _save(fig, results_dir / "confidence_boxplots.png")


# ---------------------------------------------------------------------------
# 6. Calibration Curve + ECE
# ---------------------------------------------------------------------------

def _compute_ece(confidences: np.ndarray, correctness: np.ndarray,
                 n_bins: int = 10):
    """
    Expected Calibration Error (ECE).

    Splits all predictions into n_bins equally-spaced confidence bins.
    ECE = sum_b  (|B_b| / N) * |accuracy(B_b) - confidence(B_b)|

    Returns:
        ece      (float)
        bin_data (list): list of (mean_conf, accuracy, count) or None per bin.
    """
    edges    = np.linspace(0.0, 1.0, n_bins + 1)
    ece      = 0.0
    N        = len(confidences)
    bin_data = []

    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        # Include hi boundary only in the last bin
        mask = (confidences >= lo) & (
            (confidences < hi) if b < n_bins - 1 else (confidences <= hi)
        )
        count = int(mask.sum())
        if count == 0:
            bin_data.append(None)
            continue

        acc  = float(correctness[mask].mean())
        conf = float(confidences[mask].mean())
        ece += (count / N) * abs(acc - conf)
        bin_data.append((conf, acc, count))

    return ece, bin_data


def plot_calibration_curve(records, results_dir: Path, n_bins: int = 10):
    """
    Reliability diagram: mean predicted confidence vs actual accuracy per bin.

    A well-calibrated model lies on the diagonal.  Bars above the diagonal
    indicate underconfidence; bars below indicate overconfidence.

    Computes and reports the Expected Calibration Error (ECE).

    Returns:
        ece      (float)
        bin_data (list)
    """
    confidences = np.array([r["confidence"]    for r in records])
    correctness = np.array([float(r["correct"]) for r in records])

    ece, bin_data = _compute_ece(confidences, correctness, n_bins=n_bins)
    print(f"  [Analysis] Expected Calibration Error (ECE): {ece:.4f}")

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6, 6))

        # Perfect calibration reference
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.3, label="Perfect calibration",
                zorder=5)

        # Bin bars
        valid_bins = [(c, a, cnt) for entry in bin_data
                      if entry is not None
                      for c, a, cnt in [entry]]
        if valid_bins:
            bin_confs, bin_accs, _ = zip(*valid_bins)
            bar_width = 1.0 / n_bins * 0.88

            ax.bar(bin_confs, bin_accs, width=bar_width, alpha=0.60,
                   color="#1f77b4", edgecolor="white", label="Observed accuracy")

            # Red gap markers showing deviation from perfect calibration
            for c, a in zip(bin_confs, bin_accs):
                ax.plot([c, c], [a, c], color="#D62728", linewidth=1.8, alpha=0.75)

            ax.plot(bin_confs, bin_accs, "o", color="#1f77b4", markersize=6, zorder=6)

        ax.set_xlabel("Mean Predicted Confidence")
        ax.set_ylabel("Actual Accuracy")
        ax.set_title(f"Calibration Curve\nECE = {ece:.4f}", fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper left")
        plt.tight_layout()

    _save(fig, results_dir / "calibration_curve.png")
    return ece, bin_data


# ---------------------------------------------------------------------------
# 7. Top Confident Mistakes
# ---------------------------------------------------------------------------

def plot_top_confident_mistakes(records, results_dir: Path,
                                n: int = 25, img_size: int = 112):
    """
    5×5 grid of the n highest-confidence incorrect predictions.

    Each cell shows the image, true label, predicted label, and confidence.
    High-confidence errors are the most actionable failure mode.
    """
    mistakes = sorted(
        [r for r in records if not r["correct"]],
        key=lambda r: r["confidence"],
        reverse=True,
    )[:n]

    if not mistakes:
        print("  [Analysis] No mistakes to plot; skipping top_confident_mistakes.png")
        return

    ncols = 5
    nrows = (len(mistakes) + ncols - 1) // ncols

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 2.6, nrows * 3.2))
        axes_flat = np.array(axes).flatten()

        for ax in axes_flat:
            ax.axis("off")

        for i, r in enumerate(mistakes):
            ax  = axes_flat[i]
            img = _load_image_for_display(r["image_path"], size=img_size)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(
                f"True: {r['true_label'].capitalize()}\n"
                f"Pred: {r['predicted_label'].capitalize()}\n"
                f"Conf: {r['confidence']:.3f}",
                fontsize=7.5,
                color="#AA0000",
                pad=3,
            )

        fig.suptitle(
            f"Top {len(mistakes)} Highest-Confidence Mistakes",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])

    _save(fig, results_dir / "top_confident_mistakes.png")


# ---------------------------------------------------------------------------
# 8. Most Uncertain Predictions
# ---------------------------------------------------------------------------

def plot_most_uncertain(records, results_dir: Path,
                        n: int = 25, img_size: int = 112):
    """
    5×5 grid of the n lowest-confidence predictions.

    Shows the hardest cases for the model, regardless of correctness.
    A green ✓ or red ✗ indicates whether the low-confidence prediction
    happened to be correct.
    """
    bottom_n = sorted(records, key=lambda r: r["confidence"])[:n]

    ncols = 5
    nrows = (len(bottom_n) + ncols - 1) // ncols

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 2.6, nrows * 3.2))
        axes_flat = np.array(axes).flatten()

        for ax in axes_flat:
            ax.axis("off")

        for i, r in enumerate(bottom_n):
            ax  = axes_flat[i]
            img = _load_image_for_display(r["image_path"], size=img_size)
            ax.imshow(img)
            ax.axis("off")
            marker = "✓" if r["correct"] else "✗"   # ✓ / ✗
            color  = "#005500" if r["correct"] else "#AA0000"
            ax.set_title(
                f"{marker} True: {r['true_label'].capitalize()}\n"
                f"Pred: {r['predicted_label'].capitalize()}\n"
                f"Conf: {r['confidence']:.3f}",
                fontsize=7.5,
                color=color,
                pad=3,
            )

        fig.suptitle(
            f"Top {len(bottom_n)} Most Uncertain Predictions",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])

    _save(fig, results_dir / "most_uncertain_predictions.png")


# ---------------------------------------------------------------------------
# 9. t-SNE Feature Visualization 
# ---------------------------------------------------------------------------

def _pca_reduce(X: np.ndarray, n_components: int) -> np.ndarray:
    """
    Project X onto its top n_components principal components using SVD.

    Used as a pre-processing step before t-SNE to reduce input dimensionality
    to a manageable size (typically 50).
    """
    X_c = X - X.mean(axis=0)
    # Economy SVD: Vt rows are the principal components
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    return X_c @ Vt[:n_components].T


def _compute_joint_probabilities(D_sq: np.ndarray,
                                  perplexity: float = 30.0) -> np.ndarray:
    """
    Compute symmetrized joint probability matrix P for t-SNE.

    For each sample i, binary-search for a bandwidth beta_i such that the
    perplexity of the conditional distribution P(·|i) matches the target.
    The search is fully vectorized across all samples simultaneously.

    Early exaggeration (factor 4) is applied before returning — the caller
    should divide by 4 at iteration 100.

    Args:
        D_sq:       [N, N] pairwise squared Euclidean distance matrix.
        perplexity: Target perplexity (typically 5–50).

    Returns:
        P: [N, N] symmetrized, row-sum-normalised joint probability matrix.
    """
    n          = D_sq.shape[0]
    log_target = np.log2(float(perplexity))

    betas    = np.ones(n, dtype=np.float64)
    beta_min = np.full(n, -np.inf)
    beta_max = np.full(n, np.inf)

    # Set diagonal to inf so exp(-inf * beta) = 0 (no self-affinity)
    D = D_sq.astype(np.float64).copy()
    np.fill_diagonal(D, np.inf)

    for _ in range(50):   # 50 binary-search steps is always sufficient
        exponents = np.clip(-D * betas[:, None], -500.0, 0.0)   # [N, N]
        Pi        = np.exp(exponents)                             # [N, N]
        row_sums  = np.maximum(Pi.sum(axis=1, keepdims=True), 1e-12)
        Pi_norm   = Pi / row_sums                                 # [N, N]

        # Shannon entropy in bits
        H      = -np.sum(Pi_norm * np.log2(Pi_norm + 1e-12), axis=1)  # [N]
        H_diff = H - log_target                                         # [N]

        converged = np.abs(H_diff) < 1e-5
        if converged.all():
            break

        # H > target  →  distribution too broad  →  increase beta (sharpen)
        go_up = (H_diff > 0) & ~converged
        # H < target  →  distribution too sharp  →  decrease beta (broaden)
        go_dn = (H_diff < 0) & ~converged

        beta_min = np.where(go_up, betas, beta_min)
        beta_max = np.where(go_dn, betas, beta_max)

        new_betas = betas.copy()
        new_betas = np.where(
            go_up,
            np.where(beta_max == np.inf, betas * 2.0, (betas + beta_max) / 2.0),
            new_betas,
        )
        new_betas = np.where(
            go_dn,
            np.where(beta_min == -np.inf, betas / 2.0, (betas + beta_min) / 2.0),
            new_betas,
        )
        betas = new_betas

    # Final conditional probabilities
    exponents = np.clip(-D * betas[:, None], -500.0, 0.0)
    Pi        = np.exp(exponents)
    row_sums  = np.maximum(Pi.sum(axis=1, keepdims=True), 1e-12)
    P_cond    = Pi / row_sums

    # Symmetrize, normalize, apply early exaggeration (×4)
    P = (P_cond + P_cond.T) / 2.0
    P = P / np.maximum(P.sum(), 1e-12)
    P = np.maximum(P * 4.0, 1e-12)   # early exaggeration removed at iter 100
    return P


def _tsne_numpy(X: np.ndarray,
                n_components: int  = 2,
                perplexity:    float = 30.0,
                n_iter:        int   = 300,
                lr:            float = 200.0,
                random_state:  int   = 42,
                verbose:       bool  = True) -> np.ndarray:
    """
    Exact t-SNE implemented in pure numpy (no sklearn / scipy required).

    Algorithm:
      1. PCA-preprocess to 50 dims if input is wider.
      2. Compute pairwise squared Euclidean distances.
      3. Vectorized binary search to calibrate per-sample bandwidth.
      4. Gradient descent with momentum and adaptive gains.
         Early exaggeration is applied for the first 100 iterations.

    Args:
        X:            [N, D] float array of backbone feature vectors.
        n_components: Output dimensionality (2 for scatter plots).
        perplexity:   t-SNE perplexity; good range is 5–50.
        n_iter:       Gradient descent steps (300 is sufficient for N ≤ 1500).
        lr:           Learning rate (step size multiplier).
        random_state: Reproducibility seed.
        verbose:      Print progress lines.

    Returns:
        np.ndarray [N, 2]: 2-D t-SNE embedding.
    """
    rng = np.random.default_rng(random_state)
    n   = X.shape[0]

    # Step 1 — PCA preprocessing
    if X.shape[1] > 50:
        if verbose:
            print(f"  [t-SNE] PCA: {X.shape[1]}D → 50D  (N={n})")
        X = _pca_reduce(X, 50)

    # Standardise each feature column
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # Step 2 — Pairwise squared distances
    sq  = (X ** 2).sum(axis=1)
    D_sq = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (X @ X.T), 0.0)

    # Step 3 — Joint probabilities
    if verbose:
        print(f"  [t-SNE] Computing P  (perplexity={perplexity}) ...")
    P = _compute_joint_probabilities(D_sq, perplexity=perplexity)

    # Step 4 — Gradient descent
    Y      = rng.standard_normal((n, n_components)).astype(np.float64) * 1e-4
    iY     = np.zeros_like(Y)    # momentum term
    gains  = np.ones_like(Y)     # per-parameter adaptive gains

    if verbose:
        print(f"  [t-SNE] Gradient descent: {n_iter} iterations ...")

    for it in range(n_iter):
        # Remove early exaggeration at iteration 100
        if it == 100:
            P = np.maximum(P / 4.0, 1e-12)

        # Student-t kernel Q (low-dimensional affinities)
        sq_Y = (Y ** 2).sum(axis=1)
        D_Y  = np.maximum(sq_Y[:, None] + sq_Y[None, :] - 2.0 * (Y @ Y.T), 0.0)
        Q_raw = 1.0 / (1.0 + D_Y)
        np.fill_diagonal(Q_raw, 0.0)
        Q = np.maximum(Q_raw / np.maximum(Q_raw.sum(), 1e-12), 1e-12)

        # Gradient: dC/dy_i = 4 * Σ_j (P_ij - Q_ij) * q_ij * (y_i - y_j)
        PQ   = (P - Q) * Q_raw                                  # [N, N]
        diff = Y[:, None, :] - Y[None, :, :]                    # [N, N, d]
        dY   = 4.0 * np.einsum("ij,ijk->ik", PQ, diff)          # [N, d]

        # Momentum + adaptive gains (Jacobs 1988)
        momentum = 0.5 if it < 20 else 0.8
        same_dir = (dY > 0.0) == (iY > 0.0)
        gains    = np.where(same_dir, gains * 0.8, gains + 0.2)
        gains    = np.maximum(gains, 0.01)

        iY  = momentum * iY - lr * gains * dY
        Y   = Y + iY
        Y  -= Y.mean(axis=0)    # keep embedding centred

        if verbose and (it + 1) % 100 == 0:
            print(f"  [t-SNE]   iter {it + 1}/{n_iter}")

    return Y


def plot_feature_space(features, records, class_names, results_dir: Path,
                       perplexity: float = 30.0, n_iter: int = 300,
                       max_samples: int = 1500):
    """
    Scatter plot of backbone embeddings reduced to 2D via t-SNE (pure numpy).

    Points are coloured by true class.  If N > max_samples, a random subset
    is drawn to keep runtime acceptable (< 2 min for 1500 points typically).

    Falls back gracefully if features is None (hook registration failed).
    Output: feature_space_tsne.png
    """
    if features is None:
        print("  [Analysis] Features unavailable; skipping feature_space_tsne.png")
        return

    n            = features.shape[0]
    true_indices = np.array([r["true_idx"] for r in records])

    # Subsample for performance
    if n > max_samples:
        rng          = np.random.default_rng(42)
        idx          = rng.choice(n, max_samples, replace=False)
        features     = features[idx]
        true_indices = true_indices[idx]
        print(f"  [t-SNE] Subsampled {max_samples} / {n} images for performance.")

    try:
        embedding = _tsne_numpy(features, perplexity=perplexity,
                                n_iter=n_iter, verbose=True)
    except Exception as exc:
        print(f"  [Analysis] t-SNE failed ({exc}); skipping feature_space_tsne.png")
        return

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 7))

        for cls_idx, (cls, color) in enumerate(zip(class_names, COLORS)):
            mask = true_indices == cls_idx
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=color, label=cls.capitalize(),
                s=20, alpha=0.70, linewidths=0, rasterized=True,
            )

        ax.set_title(
            "Feature Space — t-SNE  (backbone embeddings, true labels)",
            fontweight="bold",
        )
        ax.set_xlabel("t-SNE dimension 1")
        ax.set_ylabel("t-SNE dimension 2")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(markerscale=2.5, fontsize=9)
        plt.tight_layout()

    _save(fig, results_dir / "feature_space_tsne.png")


# ---------------------------------------------------------------------------
# 10. Error Analysis JSON
# ---------------------------------------------------------------------------

def save_error_analysis_json(records, class_names, ece, results_dir: Path):
    """
    Write error_analysis.json with structured summary statistics:

      most_confused_pair       — (true, predicted) pair with highest off-diagonal count
      avg_confidence_per_class — mean confidence when the image's true class is X
      avg_confidence_correct   — mean confidence over all correct predictions
      avg_confidence_incorrect — mean confidence over all incorrect predictions
      calibration              — ECE value and number of bins used
      top_confusion_percentages— per true-class breakdown of where predictions go
      per_class_accuracy       — per-class recall / accuracy
      overall_accuracy, n_total, n_correct, n_incorrect
    """
    n = len(class_names)

    # Confusion matrix [true, predicted]
    confusion = np.zeros((n, n), dtype=int)
    for r in records:
        confusion[r["true_idx"], r["pred_idx"]] += 1

    # Most confused off-diagonal pair
    conf_off = confusion.astype(float)
    np.fill_diagonal(conf_off, 0.0)
    flat_max      = int(conf_off.argmax())
    confused_true = flat_max // n
    confused_pred = flat_max % n

    # Average confidence per true class
    avg_conf = {}
    for i, cls in enumerate(class_names):
        vals = [r["confidence"] for r in records if r["true_idx"] == i]
        avg_conf[cls] = float(np.mean(vals)) if vals else 0.0

    correct_c   = [r["confidence"] for r in records if     r["correct"]]
    incorrect_c = [r["confidence"] for r in records if not r["correct"]]

    # Per-class confusion percentages
    confusion_pct = {}
    for i, cls in enumerate(class_names):
        row_sum = int(confusion[i].sum())
        if row_sum == 0:
            confusion_pct[cls] = {c: 0.0 for c in class_names}
        else:
            confusion_pct[cls] = {
                class_names[j]: round(float(confusion[i, j]) / row_sum * 100, 2)
                for j in range(n)
            }

    # Per-class accuracy
    per_class_acc = {}
    for i, cls in enumerate(class_names):
        row_sum = int(confusion[i].sum())
        per_class_acc[cls] = (float(confusion[i, i]) / row_sum
                              if row_sum > 0 else 0.0)

    analysis = {
        "most_confused_pair": {
            "true_class":    class_names[confused_true],
            "predicted_as":  class_names[confused_pred],
            "count":         int(conf_off[confused_true, confused_pred]),
        },
        "avg_confidence_per_class":  avg_conf,
        "avg_confidence_correct":    float(np.mean(correct_c))   if correct_c   else 0.0,
        "avg_confidence_incorrect":  float(np.mean(incorrect_c)) if incorrect_c else 0.0,
        "calibration": {
            "ece":    float(ece) if ece is not None else None,
            "n_bins": 10,
        },
        "top_confusion_percentages": confusion_pct,
        "per_class_accuracy":        per_class_acc,
        "overall_accuracy":   float(np.mean([r["correct"] for r in records])),
        "n_total":    len(records),
        "n_correct":  int(sum(r["correct"]     for r in records)),
        "n_incorrect":int(sum(not r["correct"] for r in records)),
    }

    out_path = results_dir / "error_analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"  [Analysis] Saved: {out_path}")
    return analysis


# ---------------------------------------------------------------------------
# Pipeline entry point [called from utils/trainer.train_model()]
# ---------------------------------------------------------------------------

def run_analysis_pipeline(model, test_loader, class_names, results_dir,
                          device, loss_fn="cross_entropy", backbone_name=None):
    """
    Orchestrate the full deep evaluation analysis pipeline.

    Called from utils/trainer.train_model() after the confusion matrix is saved,
    while test_loader is still alive (before the DataLoader del statement).

    Single forward pass collects all per-image data; all plots and the CSV
    are derived from that one pass.

    Outputs written to results_dir:
        test_predictions.csv
        confidence_by_class.png
        confidence_correct_vs_incorrect.png
        class_probability_heatmap.png
        per_class_precision_recall_f1.png
        confidence_boxplots.png
        calibration_curve.png
        top_confident_mistakes.png
        most_uncertain_predictions.png
        feature_space_tsne.png
        error_analysis.json

    Args:
        model:         Best-checkpoint model (already loaded from .pth).
        test_loader:   Test DataLoader (shuffle=False, ImageFolder).
        class_names:   List[str] of class names in dataset-index order.
        results_dir:   pathlib.Path of the experiment results directory.
        device:        torch.device.
        loss_fn:       Loss function identifier string.
        backbone_name: Architecture key (e.g. "convnext_base", "resnet152").
    """
    print("\n[Analysis] Running deep evaluation pipeline ...")
    results_dir = Path(results_dir)

    # ──── Single forward pass: collect all per-image data ──────────────────────
    records, features = collect_test_predictions(
        model, test_loader, class_names, device, loss_fn, backbone_name,
    )
    print(f"  [Analysis] Collected {len(records):,} predictions.")

    # ──── Per-image prediction CSV ──────────────────────────────────────────────
    save_test_predictions_csv(records, class_names, results_dir)

    # ──── Confidence figures ────────────────────────────────────────────────────
    # Figure 1: histogram per true class
    plot_confidence_by_class(records, class_names, results_dir)

    # Figure 2: correct vs incorrect overlay
    plot_correct_vs_incorrect_confidence(records, results_dir)

    # ──── Structure figures ─────────────────────────────────────────────────────
    # Figure 3: avg probability heatmap (true class × predicted class)
    plot_class_probability_heatmap(records, class_names, results_dir)

    # Figure 4: per-class Precision / Recall / F1 bars
    plot_per_class_metrics_bar(records, class_names, results_dir)

    # ──── Distribution figures ──────────────────────────────────────────────────
    # Figure 5: boxplots of confidence by true class
    plot_confidence_boxplots(records, class_names, results_dir)

    # Figure 6: calibration curve + ECE
    ece, _ = plot_calibration_curve(records, results_dir)

    # ──── Image grids ───────────────────────────────────────────────────────────
    # Figure 7: 25 highest-confidence wrong predictions
    plot_top_confident_mistakes(records, results_dir)

    # Figure 8: 25 lowest-confidence predictions
    plot_most_uncertain(records, results_dir)

    # ──── Feature space ─────────────────────────────────────────────────────────
    # Figure 9: t-SNE of backbone embeddings
    plot_feature_space(features, records, class_names, results_dir)

    # ──── Summary JSON ──────────────────────────────────────────────────────────
    save_error_analysis_json(records, class_names, ece, results_dir)

    print("[Analysis] Pipeline complete.\n")
