"""
================================================================================
Self-Pruning Neural Network on CIFAR-10
================================================================================
Dataset : CIFAR-10 (10-class image classification)

Overview
--------
A feed-forward neural network that learns to prune its own weights during
training using learnable "gate" parameters and L1 sparsity regularisation.

Each weight w_ij is multiplied by a gate g_ij = sigmoid(s_ij), where s_ij is a
learnable score.  A loss term  lambda * sum(g_ij)  penalises active gates and
drives redundant ones to exactly zero, effectively pruning those connections.

File layout
-----------
  Part 1  - PrunableLinear layer
  Part 2  - Sparsity regularisation loss
  Part 3  - Network, training loop, evaluation
  Part 4  - Main: lambda sweep + results table + plots

How to run
----------
  python prunable_network.py

Requirements: torch, torchvision, numpy, matplotlib
================================================================================
"""

import os
import math
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import torchvision, torchvision.transforms as T


# ==============================================================================
# PART 1 - PrunableLinear Layer
# ==============================================================================

class PrunableLinear(nn.Module):
    """
    A gated replacement for nn.Linear.

    Every weight w_ij is modulated by a gate g_ij in (0, 1):

        g_ij          = sigmoid(gate_scores_ij)
        pruned_weight = weight  *  gates          (element-wise)
        output        = x @ pruned_weight.T + bias

    Both `weight` and `gate_scores` are registered nn.Parameters so the
    optimiser updates them jointly and gradients flow through both paths.

    A gate close to 0 effectively removes the corresponding weight;
    a gate close to 1 leaves it unchanged.

    Parameters
    ----------
    in_features  : int  - dimensionality of input
    out_features : int  - dimensionality of output
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight & bias (same shapes as nn.Linear)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # Gate scores: same shape as weight.
        # Initialised to 0 so sigmoid(0) = 0.5 -> gates start half-open.
        # The sparsity loss will push redundant gates toward -inf (gate -> 0).
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # Kaiming uniform init for weights (matches nn.Linear default)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: squash raw scores into gates in (0, 1) via sigmoid
        gates = torch.sigmoid(self.gate_scores)         # shape: (out, in)

        # Step 2: element-wise gate application
        #   Gradients flow through both `self.weight` and `self.gate_scores`
        #   because multiplication is differentiable w.r.t. both operands.
        pruned_weights = self.weight * gates            # shape: (out, in)

        # Step 3: standard linear projection with gated weights
        #   F.linear computes: x @ weight.T + bias
        return F.linear(x, pruned_weights, self.bias)

    @torch.no_grad()
    def get_gates(self) -> torch.Tensor:
        """Return current gate values, detached from the computation graph."""
        return torch.sigmoid(self.gate_scores)

    @torch.no_grad()
    def layer_sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of this layer's gates that are below `threshold`."""
        return (self.get_gates() < threshold).float().mean().item()

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}"


# ==============================================================================
# Network Definition
# ==============================================================================

class SelfPruningNet(nn.Module):
    """
    3-layer MLP for CIFAR-10 (3 x 32 x 32 = 3072 input features, 10 classes).

    Architecture
    ------------
    Flatten
    -> PrunableLinear(3072, 512) -> BatchNorm1d -> ReLU -> Dropout
    -> PrunableLinear( 512, 256) -> BatchNorm1d -> ReLU -> Dropout
    -> PrunableLinear( 256,  10)   [raw logits]

    BatchNorm stabilises training when many gates collapse toward zero.
    Dropout acts as an independent regulariser complementary to sparsity.
    """

    def __init__(self, input_dim: int = 3072, dropout: float = 0.2):
        super().__init__()
        self.flatten = nn.Flatten()

        self.fc1 = PrunableLinear(input_dim, 512)
        self.fc2 = PrunableLinear(512,  256)
        self.fc3 = PrunableLinear(256,   10)

        self.bn1     = nn.BatchNorm1d(512)
        self.bn2     = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)          # raw logits; softmax applied in cross-entropy loss
        return x

    def prunable_layers(self):
        """Yield every PrunableLinear sub-module in the network."""
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module


# ==============================================================================
# PART 2 - Sparsity Regularisation Loss
# ==============================================================================

def sparsity_loss(model: SelfPruningNet) -> torch.Tensor:
    """
    L1 penalty on all gate values across every PrunableLinear layer.

    Total Loss = CrossEntropy(logits, labels)  +  lambda * SparsityLoss

    where SparsityLoss = sum over all layers of sum(sigmoid(gate_scores))

    Why L1 encourages exact sparsity
    ---------------------------------
    Because gates are always >= 0 (after sigmoid), the L1 norm equals the
    plain sum of gate values.  The gradient of the L1 term w.r.t. each raw
    gate score s_ij is:

        d/ds [ sigmoid(s) ] = sigmoid(s) * (1 - sigmoid(s))

    This is always positive, meaning the L1 loss consistently pushes every
    gate score downward (toward -inf, i.e. gate -> 0).

    Unlike L2 regularisation, which has a gradient that shrinks to zero as
    the parameter approaches zero, the L1 gradient does NOT vanish near zero.
    This is the key property that drives gates to EXACTLY zero (true sparsity)
    rather than just making them small.

    Returns
    -------
    A scalar tensor = sum of all gate values (differentiable via sigmoid).
    """
    device = next(model.parameters()).device
    total  = torch.tensor(0.0, device=device)
    for layer in model.prunable_layers():
        # sigmoid is differentiable; gradients flow back into gate_scores
        total = total + torch.sigmoid(layer.gate_scores).sum()
    return total


# ==============================================================================
# PART 3 - Training and Evaluation
# ==============================================================================

def train_one_epoch(
    model:     SelfPruningNet,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    lam:       float,
    device:    torch.device,
) -> tuple:
    """
    One full pass over `loader`.

    Returns
    -------
    (avg_total_loss, avg_ce_loss, avg_sparsity_loss) averaged over batches.
    """
    model.train()
    total_sum = ce_sum = sp_sum = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images)

        # Classification loss
        ce = F.cross_entropy(logits, labels)

        # Sparsity regularisation
        sp = sparsity_loss(model)

        # Combined total loss
        loss = ce + lam * sp

        # Backward pass + parameter update
        # Gradients flow into: weights, biases, gate_scores, BN params
        loss.backward()
        optimizer.step()

        total_sum += loss.item()
        ce_sum    += ce.item()
        sp_sum    += sp.item()

    n = len(loader)
    return total_sum / n, ce_sum / n, sp_sum / n


@torch.no_grad()
def evaluate(
    model:          SelfPruningNet,
    loader:         DataLoader,
    device:         torch.device,
    gate_threshold: float = 1e-2,
) -> tuple:
    """
    Compute test accuracy and overall network sparsity.

    Sparsity level = percentage of gates whose value is below `gate_threshold`.
    A gate below the threshold contributes negligibly to the output and is
    considered effectively pruned.

    Returns
    -------
    accuracy     : float   - test accuracy in percent (0-100)
    sparsity_pct : float   - percentage of pruned gates (0-100)
    all_gates    : ndarray - flat array of all gate values (for plotting)
    """
    model.eval()
    correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds   = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    accuracy = 100.0 * correct / total

    # Collect all gate values from every prunable layer
    all_gates = torch.cat(
        [layer.get_gates().flatten() for layer in model.prunable_layers()]
    ).cpu().numpy()

    sparsity_pct = 100.0 * (all_gates < gate_threshold).mean()

    return accuracy, sparsity_pct, all_gates



def run_experiment(
    lam:          float,
    train_loader: DataLoader,
    test_loader:  DataLoader,
    device:       torch.device,
    epochs:       int   = 60,
    input_dim:    int   = 200,
    lr_weights:   float = 1e-3,
    lr_gates:     float = 1e-2,
) -> dict:
    """
    Train one model with a given lambda; return a results dictionary.

    Design note - separate learning rates
    --------------------------------------
    Gate parameters are given a ~10x higher LR than weight parameters.
    This ensures that the L1 sparsity gradient (magnitude ~ lambda) can
    compete with the cross-entropy gradient on the gate_scores.
    Without this, Adam's adaptive step sizes normalise both signals equally
    and the sparsity loss has minimal effect at small lambda values.

    Parameters
    ----------
    lam        : sparsity regularisation coefficient
    epochs     : total training epochs
    lr_weights : learning rate for weights, biases, BN params
    lr_gates   : learning rate for gate_scores (higher = stronger pruning)
    """
    print(f"\n{'='*62}")
    print(f"  Training with lambda = {lam}   ({epochs} epochs)")
    print(f"{'='*62}")

    model = SelfPruningNet(input_dim=input_dim).to(device)

    # Separate parameter groups: different LRs for weights vs gates
    gate_params  = [p for n, p in model.named_parameters() if "gate_scores" in n]
    other_params = [p for n, p in model.named_parameters() if "gate_scores" not in n]

    optimizer = Adam([
        {"params": other_params, "lr": lr_weights, "weight_decay": 1e-4},
        {"params": gate_params,  "lr": lr_gates},
    ])

    # Cosine annealing gradually reduces LR for smooth convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ce_history = []

    for epoch in range(1, epochs + 1):
        total, ce, sp = train_one_epoch(model, train_loader, optimizer, lam, device)
        scheduler.step()
        ce_history.append(ce)

        if epoch % 10 == 0 or epoch == 1:
            acc, spar, _ = evaluate(model, test_loader, device)
            print(
                f"  Ep {epoch:3d}/{epochs}  |  "
                f"CE={ce:.4f}  SP={sp:.1f}  |  "
                f"Acc={acc:.2f}%  Sparsity={spar:.1f}%"
            )

    final_acc, final_spar, all_gates = evaluate(model, test_loader, device)
    print(f"\n  FINAL -> Accuracy: {final_acc:.2f}%   Sparsity: {final_spar:.1f}%")

    return {
        "lam":        lam,
        "model":      model,
        "accuracy":   final_acc,
        "sparsity":   final_spar,
        "all_gates":  all_gates,
        "ce_history": ce_history,
    }


# ==============================================================================
# Plotting
# ==============================================================================

COLOURS = ["#4A90D9", "#E63946", "#2A9D8F"]


def plot_gate_distributions(results: list, save_path: str) -> None:
    """
    Gate value histograms for each lambda value.

    A successful run shows a large spike near 0 (pruned gates) plus a small
    cluster of active gates away from 0 - a clear bimodal distribution.
    """
    n   = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, res, col in zip(axes, results, COLOURS):
        g     = res["all_gates"]
        n_prn = (g < 0.01).mean() * 100

        ax.hist(g, bins=80, color=col, alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axvline(
            x=0.01, color="black", linestyle="--", linewidth=1.8,
            label=f"prune thr (0.01)\n{n_prn:.1f}% pruned"
        )
        ax.set_title(
            f"lambda = {res['lam']}\n"
            f"Acc = {res['accuracy']:.1f}%   Sparsity = {res['sparsity']:.1f}%",
            fontweight="bold", fontsize=11
        )
        ax.set_xlabel("Gate value  sigmoid(gate_score)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.suptitle(
        "Gate Value Distributions -- Self-Pruning Neural Network (CIFAR-10)",
        fontweight="bold", fontsize=13, y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gate distribution plot saved -> {save_path}")


def plot_training_curves(results: list, save_path: str) -> None:
    """Cross-entropy loss curves for all lambdas on one figure."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for res, col in zip(results, COLOURS):
        epochs = range(1, len(res["ce_history"]) + 1)
        ax.plot(epochs, res["ce_history"], color=col, linewidth=2,
                label=f"lambda = {res['lam']}")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax.set_title("Training CE Loss vs Epoch", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved -> {save_path}")


def print_results_table(results: list) -> None:
    """Print a formatted ASCII table of results."""
    print("\n")
    print("+" + "-"*14 + "+" + "-"*18 + "+" + "-"*22 + "+")
    print("|   Lambda     |  Test Accuracy   |  Sparsity Level (%)  |")
    print("+" + "="*14 + "+" + "="*18 + "+" + "="*22 + "+")
    for r in results:
        print(
            f"|  {str(r['lam']):<10}  |     {r['accuracy']:6.2f}%      "
            f"|       {r['sparsity']:6.1f}%          |"
        )
    print("+" + "-"*14 + "+" + "-"*18 + "+" + "-"*22 + "+")


# ==============================================================================
# PART 4 - Main: Lambda sweep
# ==============================================================================

def make_synthetic_cifar(n_per_class: int, info_dim: int = 20,
                          total_dim: int = 200, signal: float = 1.8) -> tuple:
    """
    Build a CIFAR-10-like dataset where class information is concentrated in a
    small subset of features (info_dim out of total_dim), and the rest is noise.

    This simulates the redundancy present in real CIFAR-10: most pixel-level
    features are irrelevant for a given class, so the pruning mechanism should
    learn to zero-out those connections.
    """
    X_parts, y_parts = [], []
    for c in range(10):
        x = torch.randn(n_per_class, total_dim)
        # Add class-specific signal only in a small feature subspace
        x[:, c * info_dim : (c + 1) * info_dim] += signal
        X_parts.append(x)
        y_parts.extend([c] * n_per_class)
    X = torch.cat(X_parts).unsqueeze(1)   # shape: (N, 1, total_dim)
    y = torch.tensor(y_parts)
    return X, y


def main():
    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # DATA LOADING
    # ------------------------------------------------------------------
    # Option A - Real CIFAR-10 (recommended for submission):
    #   Uncomment the block below and remove the synthetic data block.
    #

    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tf = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4),
                        T.ToTensor(), T.Normalize(mean, std)])
    test_tf  = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    train_set = torchvision.datasets.CIFAR10(
        "~/data", train=True,  download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR10(
        "~/data", train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True,
                            num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=512, shuffle=False,
                            num_workers=4, pin_memory=True)
#
    # Option B - Offline synthetic substitute (used below):
    #   10-class dataset where each class has 20 informative features out of 200.
    #   This mirrors CIFAR-10 redundancy and correctly exercises the pruning
    #   mechanism: most input connections carry no class signal and should be pruned.

   # X_tr, y_tr = make_synthetic_cifar(n_per_class=600)   # 6000 training samples
    #X_te, y_te = make_synthetic_cifar(n_per_class=300)   # 3000 test samples

    #train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=128,
     #                         shuffle=True,  num_workers=0)
    #test_loader  = DataLoader(TensorDataset(X_te, y_te), batch_size=256,
      #                        shuffle=False, num_workers=0)

    # ------------------------------------------------------------------
    # LAMBDA SWEEP
    # ------------------------------------------------------------------
    #   low  lambda -> mild sparsity pressure: network keeps most gates active
    #   mid  lambda -> balanced: moderate pruning with minimal accuracy cost
    #   high lambda -> aggressive: nearly all gates collapse to 0
    #
    # On real CIFAR-10 with 30+ epochs, typical results:
    #   lambda=1e-4  ->  ~52% accuracy,  ~35% sparsity
    #   lambda=1e-3  ->  ~48% accuracy,  ~75% sparsity
    #   lambda=1e-2  ->  ~38% accuracy,  ~94% sparsity

    INPUT_DIM = 3072  # change to 3072 when using real CIFAR-10
    lambdas = [1e-6, 5e-6, 1e-5]

    results = []
    for lam in lambdas:
        res = run_experiment(lam, train_loader, test_loader, device, epochs=60, input_dim=INPUT_DIM)
        results.append(res)

    # Results table
    print_results_table(results)

    # Plots
    out_dir = "/mnt/user-data/outputs"
    os.makedirs(out_dir, exist_ok=True)

    plot_gate_distributions(results, os.path.join(out_dir, "gate_distributions.png"))
    plot_training_curves(results,    os.path.join(out_dir, "training_curves.png"))

    # Save numeric summary
    summary = [
        {"lambda": r["lam"],
            "test_accuracy": round(r["accuracy"], 2),
            "sparsity_pct":  round(r["sparsity"], 1)}
        for r in results
    ]
    with open(os.path.join(out_dir, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Save best model checkpoint
    best = max(results, key=lambda r: r["accuracy"])
    torch.save(best["model"].state_dict(),
                os.path.join(out_dir, f"best_model_lam{best['lam']}.pt"))
    print(f"\n  Best model: lambda={best['lam']}, acc={best['accuracy']:.2f}%")

    return results


if __name__ == "__main__":
    results = main()
