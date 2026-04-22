import os
import math
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")           
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import torchvision, torchvision.transforms as T




class PrunableLinear(nn.Module):
   

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

     
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

      
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        gates = torch.sigmoid(self.gate_scores)        


        pruned_weights = self.weight * gates          

      
        return F.linear(x, pruned_weights, self.bias)

    @torch.no_grad()
    def get_gates(self) -> torch.Tensor:
        
        return torch.sigmoid(self.gate_scores)

    @torch.no_grad()
    def layer_sparsity(self, threshold: float = 1e-2) -> float:
        
        return (self.get_gates() < threshold).float().mean().item()

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}"




class SelfPruningNet(nn.Module):
 

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

        x = self.fc3(x)          
        return x

    def prunable_layers(self):
        
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module




def sparsity_loss(model: SelfPruningNet) -> torch.Tensor:
 
    device = next(model.parameters()).device
    total  = torch.tensor(0.0, device=device)
    for layer in model.prunable_layers():
       
        total = total + torch.sigmoid(layer.gate_scores).sum()
    return total




def train_one_epoch(
    model:     SelfPruningNet,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    lam:       float,
    device:    torch.device,
) -> tuple:
  
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
    
    print(f"\n{'='*62}")
    print(f"  Training with lambda = {lam}   ({epochs} epochs)")
    print(f"{'='*62}")

    model = SelfPruningNet(input_dim=input_dim).to(device)

    gate_params  = [p for n, p in model.named_parameters() if "gate_scores" in n]
    other_params = [p for n, p in model.named_parameters() if "gate_scores" not in n]

    optimizer = Adam([
        {"params": other_params, "lr": lr_weights, "weight_decay": 1e-4},
        {"params": gate_params,  "lr": lr_gates},
    ])

    
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



COLOURS = ["#4A90D9", "#E63946", "#2A9D8F"]


def plot_gate_distributions(results: list, save_path: str) -> None:
   
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



def make_synthetic_cifar(n_per_class: int, info_dim: int = 20,
                          total_dim: int = 200, signal: float = 1.8) -> tuple:
   
    X_parts, y_parts = [], []
    for c in range(10):
        x = torch.randn(n_per_class, total_dim)
       
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


    INPUT_DIM = 3072  
    lambdas = [1e-6, 5e-6, 1e-5]

    results = []
    for lam in lambdas:
        res = run_experiment(lam, train_loader, test_loader, device, epochs=60, input_dim=INPUT_DIM)
        results.append(res)

        print_results_table(results)

    
    out_dir = "/mnt/user-data/outputs"
    os.makedirs(out_dir, exist_ok=True)

    plot_gate_distributions(results, os.path.join(out_dir, "gate_distributions.png"))
    plot_training_curves(results,    os.path.join(out_dir, "training_curves.png"))

    
    summary = [
        {"lambda": r["lam"],
            "test_accuracy": round(r["accuracy"], 2),
            "sparsity_pct":  round(r["sparsity"], 1)}
        for r in results
    ]
    with open(os.path.join(out_dir, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    
    best = max(results, key=lambda r: r["accuracy"])
    torch.save(best["model"].state_dict(),
                os.path.join(out_dir, f"best_model_lam{best['lam']}.pt"))
    print(f"\n  Best model: lambda={best['lam']}, acc={best['accuracy']:.2f}%")

    return results


if __name__ == "__main__":
    results = main()
