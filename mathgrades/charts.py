"""Every chart saved to reports/figures, drawn in the app's colours."""

import re
import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write image files; never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from mathgrades.settings import SETTINGS

THEME = tomllib.loads(SETTINGS.theme_path.read_text())["theme"]
INK_BLUE = THEME["primaryColor"]
PAPER = THEME["backgroundColor"]
INK = THEME["textColor"]
GREY = "#a7a99f"
RED_PEN = "#c8312b"  # second series colour: the red of a marking pen

plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.prop_cycle": plt.cycler(color=[INK_BLUE, RED_PEN, GREY]),
})


def file_slug(name: str) -> str:
    """'SVM (RBF)' -> 'svm_rbf'."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def save(fig: plt.Figure, name: str) -> Path:
    SETTINGS.figures_dir.mkdir(parents=True, exist_ok=True)
    path = SETTINGS.figures_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def confusion_file_name(model_name: str) -> str:
    return f"confusion_{file_slug(model_name)}.png"


def confusion_chart(y_true, y_pred, model_name: str) -> Path:
    """Counts of (true band, predicted band). Correct answers sit on the diagonal."""
    counts = confusion_matrix(y_true, y_pred, labels=range(SETTINGS.n_classes))
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(counts, cmap="Blues")
    for row in range(SETTINGS.n_classes):
        for col in range(SETTINGS.n_classes):
            dark_cell = counts[row, col] > counts.max() / 2
            ax.text(col, row, counts[row, col], ha="center", va="center",
                    color="white" if dark_cell else INK)
    ax.set_xticks(range(SETTINGS.n_classes), SETTINGS.band_names, rotation=20)
    ax.set_yticks(range(SETTINGS.n_classes), SETTINGS.band_names)
    ax.set(xlabel="Predicted band", ylabel="True band", title=model_name)
    ax.spines[:].set_visible(False)
    return save(fig, confusion_file_name(model_name))


def training_history_chart(history: dict) -> Path:
    """Training loss vs stop-set loss per epoch, and the learning rate schedule."""
    fig, (loss_ax, lr_ax) = plt.subplots(1, 2, figsize=(10, 3.8))
    loss_ax.plot(history["train_loss"], label="training batches")
    loss_ax.plot(history["stop_loss"], label="stop set")
    loss_ax.set(title="Loss per epoch", xlabel="Epoch", ylabel="Loss")
    loss_ax.legend(frameon=False)
    lr_ax.plot(history["learning_rate"], color=INK_BLUE)
    lr_ax.set(title="Learning rate (halved when the stop-set loss stalls)",
              xlabel="Epoch", yscale="log")
    return save(fig, "training_history.png")


def feature_tests_chart(tests: pd.DataFrame, level: float) -> Path:
    """-log10(p) per column: the further right, the stronger the evidence.
    The dashed line is the chosen significance level; kept columns are blue."""
    fig, ax = plt.subplots(figsize=(7, 8))
    strength = -np.log10(tests["p_value"].clip(lower=1e-300))
    colors = [INK_BLUE if kept else GREY for kept in tests["kept"]]
    ax.barh(tests["column"][::-1], strength[::-1], color=colors[::-1])
    ax.axvline(-np.log10(level), color=RED_PEN, linestyle="--")
    ax.set(xlabel="-log10(p-value)",
           title=f"Evidence of a link with the band (kept if p < {level})")
    return save(fig, "feature_tests.png")


def search_history_chart(trials: pd.DataFrame, model_name: str) -> Path:
    """Kappa of each Optuna trial, with the best score so far as a line."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.scatter(trials["number"], trials["value"], color=GREY, label="trial")
    ax.plot(trials["number"], trials["value"].cummax(), color=INK_BLUE, label="best so far")
    ax.set(xlabel="Trial", ylabel="Mean CV weighted kappa", title=f"Optuna search: {model_name}")
    ax.legend(frameon=False)
    return save(fig, f"optuna_{file_slug(model_name)}.png")


def importance_chart(importance: pd.DataFrame) -> Path:
    """Permutation importance: drop in test kappa when a column is shuffled."""
    shown = importance[importance["kappa_drop"].abs() > 1e-9].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(shown) + 1.2))
    ax.barh(shown["column"], shown["kappa_drop"], xerr=shown["kappa_drop_std"],
            color=INK_BLUE, ecolor=GREY)
    ax.set(xlabel="Drop in test kappa when the column is shuffled",
           title="Permutation importance (GradeNet)")
    return save(fig, "permutation_importance.png")


def reliability_chart(curves: dict[str, tuple[np.ndarray, np.ndarray]]) -> Path:
    """Average confidence vs share correct per confidence bin. On the diagonal
    = calibrated; below it = over-confident; above it = under-confident."""
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], color=GREY, linestyle="--", label="perfect calibration")
    for (label, (confidence, accuracy)), color in zip(curves.items(), [RED_PEN, INK_BLUE]):
        ax.plot(confidence, accuracy, marker="o", color=color, label=label)
    ax.set(xlabel="Model confidence in its top band", ylabel="Share actually correct",
           title="Reliability diagram (test set)", xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False)
    return save(fig, "reliability.png")


def learning_curve_chart(sizes: np.ndarray, curves: dict[str, np.ndarray]) -> Path:
    """Cross-validated kappa (mean and spread) against the number of training students."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, scores in curves.items():
        mean, spread = scores.mean(axis=1), scores.std(axis=1)
        ax.plot(sizes, mean, marker="o", label=name)
        ax.fill_between(sizes, mean - spread, mean + spread, alpha=0.15)
    ax.set(xlabel="Training students", ylabel="CV weighted kappa",
           title="Learning curve")
    ax.legend(frameon=False)
    return save(fig, "learning_curve.png")


def ablation_chart(table: pd.DataFrame) -> Path:
    """Change in CV kappa versus the full model, one bar per removed choice."""
    rows = table.iloc[1:][::-1]  # row 0 is the full model itself
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(rows) + 1.2))
    ax.barh(rows["variant"], rows["change"],
            color=[RED_PEN if change < 0 else INK_BLUE for change in rows["change"]])
    ax.axvline(0, color=GREY, linewidth=1)
    ax.set(xlabel="Change in CV kappa versus the full model", title="Ablation study")
    return save(fig, "ablation.png")
