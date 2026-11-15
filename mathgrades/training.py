"""GradeNet wrapped as a scikit-learn classifier, with a hand-written training loop.

Being a scikit-learn estimator (fit / predict / predict_proba) lets the
network sit in the same Pipeline as the SVM and gradient boosting, so
cross-validation, Optuna, permutation importance and learning curves treat
every model in exactly the same way.
"""

import copy

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from mathgrades.network import GradeNet
from mathgrades.settings import SETTINGS


def make_deterministic() -> None:
    """Deterministic kernels; every fit also reseeds itself (see fit)."""
    torch.manual_seed(SETTINGS.seed)
    torch.use_deterministic_algorithms(True)


def class_weights(bands: np.ndarray) -> torch.Tensor:
    """weight = students / (bands x students in this band).

    Excellent has the fewest students, so its mistakes cost the most in the
    loss; without this the network would mostly ignore it.
    """
    counts = np.bincount(bands, minlength=SETTINGS.n_classes)
    if (counts == 0).any():
        raise ValueError(f"Every band needs at least one training student; counts were {counts}")
    return torch.tensor(len(bands) / (SETTINGS.n_classes * counts), dtype=torch.float32)


class OrdinalLoss(nn.Module):
    """Binary cross-entropy over the K-1 threshold questions.

    A Good student (band 2) has targets [1, 1, 0]: above At risk yes, above
    Average yes, above Good no. Label smoothing moves each target towards 0.5
    by `smoothing` (1 -> 0.95, 0 -> 0.05 for smoothing 0.1). With class
    weights, each student's loss counts in proportion to their band's weight.
    """

    def __init__(self, weight: torch.Tensor | None, smoothing: float):
        super().__init__()
        self.weight = weight
        self.smoothing = smoothing

    def forward(self, threshold_logits: torch.Tensor, bands: torch.Tensor) -> torch.Tensor:
        levels = torch.arange(threshold_logits.shape[1])
        targets = (bands.unsqueeze(1) > levels).float()
        targets = targets * (1 - self.smoothing) + self.smoothing / 2
        per_student = binary_cross_entropy_with_logits(
            threshold_logits, targets, reduction="none").mean(dim=1)
        if self.weight is None:
            return per_student.mean()
        student_weights = self.weight[bands]
        return (per_student * student_weights).sum() / student_weights.sum()


def as_tensors(X: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


class GradeNetClassifier(ClassifierMixin, BaseEstimator):
    """Hyperparameters go in __init__ (scikit-learn rule: store them unchanged);
    everything learnt gets a trailing underscore and is set in fit()."""

    def __init__(self, hidden_sizes: tuple[int, ...] = (32, 16, 8), dropout: float = 0.3,
                 learning_rate: float = 1e-3, weight_decay: float = 1e-3, batch_size: int = 32,
                 head: str = "ordinal", linear_skip: bool = True, batch_norm: bool = True,
                 class_weighting: bool = True, label_smoothing: float = SETTINGS.label_smoothing,
                 n_networks: int = 1, seed: int = SETTINGS.seed,
                 max_epochs: int = SETTINGS.max_epochs):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.head = head
        self.linear_skip = linear_skip
        self.batch_norm = batch_norm
        self.class_weighting = class_weighting
        self.label_smoothing = label_smoothing
        self.n_networks = n_networks
        self.seed = seed
        self.max_epochs = max_epochs

    def fit(self, X, y):
        """Train n_networks fresh GradeNets on (X, y), one per seed.

        A stratified 20 % of (X, y) is kept aside as the "stop set": after every
        epoch we measure the loss on it. The learning rate halves when that loss
        stalls for lr_patience epochs, training ends when it stalls for
        stop_patience epochs, and the weights from the best epoch are restored.
        Finally a temperature is fitted on the stop set (temperature scaling).
        With n_networks > 1 (a seed ensemble) the networks differ only in their
        starting weights and batch order, and their logits are averaged.
        """
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y)
        self.classes_ = np.arange(SETTINGS.n_classes)
        X_fit, X_stop, y_fit, y_stop = train_test_split(
            X, y, test_size=SETTINGS.stop_share, stratify=y, random_state=SETTINGS.seed)

        self.nets_, histories = [], []
        for offset in range(self.n_networks):
            seed = self.seed + offset
            torch.manual_seed(seed)  # same starting weights for every fit with this seed
            net = GradeNet(X.shape[1], tuple(self.hidden_sizes), self.dropout,
                           SETTINGS.n_classes, self.batch_norm, self.head, self.linear_skip)
            histories.append(self._train(net, seed, X_fit, y_fit, X_stop, y_stop))
            self.nets_.append(net)
        self.history_ = histories[0]
        self.temperature_ = 1.0
        self.temperature_ = fit_temperature(self, X_stop, y_stop)
        return self

    def _loss_function(self, y_fit: np.ndarray) -> nn.Module:
        weight = class_weights(y_fit) if self.class_weighting else None
        if self.head == "ordinal":
            return OrdinalLoss(weight, self.label_smoothing)
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=self.label_smoothing)

    def _train(self, net: GradeNet, seed: int, X_fit, y_fit, X_stop, y_stop) -> dict:
        # drop_last avoids a final batch of one student, which BatchNorm cannot normalise.
        batches = DataLoader(TensorDataset(*as_tensors(X_fit, y_fit)), batch_size=self.batch_size,
                             shuffle=True, drop_last=True,
                             generator=torch.Generator().manual_seed(seed))
        loss_fn = self._loss_function(y_fit)
        # AdamW = Adam with weight decay applied directly to the weights
        # (shrinks them a little every step, a form of L2 regularisation).
        optimiser = torch.optim.AdamW(net.parameters(), lr=self.learning_rate,
                                      weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimiser, mode="min", factor=0.5,
                                      patience=SETTINGS.lr_patience)
        X_stop_t, y_stop_t = as_tensors(X_stop, y_stop)

        history = {"train_loss": [], "stop_loss": [], "learning_rate": []}
        best_loss, best_weights, stale_epochs = float("inf"), None, 0
        for _ in range(self.max_epochs):
            history["train_loss"].append(run_one_epoch(net, batches, loss_fn, optimiser))
            net.eval()
            with torch.no_grad():
                stop_loss = loss_fn(net(X_stop_t), y_stop_t).item()
            history["stop_loss"].append(stop_loss)
            history["learning_rate"].append(optimiser.param_groups[0]["lr"])
            scheduler.step(stop_loss)

            if stop_loss < best_loss:
                best_loss, best_weights, stale_epochs = stop_loss, copy.deepcopy(net.state_dict()), 0
            else:
                stale_epochs += 1
                if stale_epochs >= SETTINGS.stop_patience:
                    break

        net.load_state_dict(best_weights)
        net.eval()
        return history

    def logits(self, X) -> np.ndarray:
        """Raw outputs, averaged over the networks: K-1 threshold logits
        (ordinal) or K band logits (softmax)."""
        inputs = torch.tensor(np.asarray(X, dtype=np.float32))
        outputs = []
        with torch.no_grad():
            for net in self.nets_:
                net.eval()  # dropout off, BatchNorm uses its stored averages
                outputs.append(net(inputs).numpy())
        return np.mean(outputs, axis=0)

    def probabilities_from_logits(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        scaled = logits / temperature
        if self.head == "ordinal":
            return ordinal_probabilities(scaled)
        return softmax(scaled)

    def predict_proba(self, X) -> np.ndarray:
        """Calibrated probability of every band (temperature-scaled); rows sum to 1."""
        return self.probabilities_from_logits(self.logits(X), self.temperature_)

    def predict_proba_uncalibrated(self, X) -> np.ndarray:
        """The same before temperature scaling (for the calibration study)."""
        return self.probabilities_from_logits(self.logits(X), 1.0)

    def predict(self, X) -> np.ndarray:
        """Ordinal: the number of thresholds passed (P(band > k) > 0.5), which is
        the median band of the predicted distribution. Softmax: the most likely band.
        A positive temperature never changes either answer."""
        logits = self.logits(X)
        if self.head == "ordinal":
            return (logits > 0).sum(axis=1)
        return logits.argmax(axis=1)


def run_one_epoch(net: GradeNet, batches: DataLoader, loss_fn, optimiser) -> float:
    """One pass over the training batches; returns the average training loss."""
    net.train()  # dropout on, BatchNorm uses batch statistics
    total, seen = 0.0, 0
    for X_batch, y_batch in batches:
        optimiser.zero_grad()                  # forget the previous batch's gradients
        loss = loss_fn(net(X_batch), y_batch)  # forward pass
        loss.backward()                        # backpropagation: gradient of loss w.r.t. every weight
        optimiser.step()                       # move every weight a small step downhill
        total += loss.item() * len(y_batch)
        seen += len(y_batch)
    return total / seen


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.exp(logits - logits.max(axis=1, keepdims=True))  # subtract max: no overflow
    return shifted / shifted.sum(axis=1, keepdims=True)


def ordinal_probabilities(threshold_logits: np.ndarray) -> np.ndarray:
    """Turn P(band > k) for k = 0..K-2 into P(band = k) for k = 0..K-1:
        P(At risk)   = 1 - P(> At risk)
        P(Average)   = P(> At risk) - P(> Average)
        P(Good)      = P(> Average) - P(> Good)
        P(Excellent) = P(> Good)
    Never negative, because the ordered biases keep P(> k) decreasing in k."""
    above = 1 / (1 + np.exp(-threshold_logits))
    ones, zeros = np.ones((len(above), 1)), np.zeros((len(above), 1))
    return np.hstack([ones, above]) - np.hstack([above, zeros])


def fit_temperature(model: GradeNetClassifier, X_stop: np.ndarray, y_stop: np.ndarray) -> float:
    """Temperature scaling (Guo et al., 2017): find the T > 0 that makes the
    band probabilities from logits / T give the lowest negative log-likelihood
    on the stop set. T > 1 softens over-confident probabilities, T < 1 sharpens
    under-confident ones. Searched on log T so T stays positive."""
    logits = model.logits(X_stop)

    def negative_log_likelihood(log_t: float) -> float:
        probabilities = model.probabilities_from_logits(logits, np.exp(log_t))
        return -np.mean(np.log(probabilities[np.arange(len(y_stop)), y_stop] + 1e-12))

    best = minimize_scalar(negative_log_likelihood, bounds=(-3, 3), method="bounded")
    return float(np.exp(best.x))
