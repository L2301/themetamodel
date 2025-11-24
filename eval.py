# eval.py
import json
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math
from config import R, r, DEVICE, BATCH_SIZE, N_TEST, SEQ_LEN, V
from torus_markov import make_p_true
from data import make_dataloaders
from model import TinyTorusTransformer
from metrics import p_model_from_qk, kl_p_true_p_model, \
                    torus_pairwise_distances, kspace_pairwise_distances, \
                    distance_correlation, neighborhood_overlap, learned_radii

def load_latest_metrics(checkpoint_dir="checkpoints"):
    """Load the most recent metrics history file."""
    metrics_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("metrics_history_") and f.endswith(".json")]
    
    if not metrics_files:
        raise FileNotFoundError(f"No metrics history files found in {checkpoint_dir}/")
    
    # Sort by modification time (most recent first)
    metrics_files.sort(key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)), reverse=True)
    latest_file = metrics_files[0]
    
    metrics_path = os.path.join(checkpoint_dir, latest_file)
    print(f"Loading metrics from: {metrics_path}")
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics

def plot_metrics(metrics, save_path="training_metrics.png", P_true=None):
    """Generate plots for all training metrics."""
    steps = metrics["steps"]
    
    # Compute perplexities
    train_perplexity = [math.exp(loss) for loss in metrics["train_loss"]]
    val_perplexity = [math.exp(loss) for loss in metrics["val_loss"]]
    
    # Compute theoretical bounds if P_true provided
    if P_true is not None:
        best_loss, best_perplexity = compute_best_possible_perplexity(P_true)
        uniform_perplexity = V
    
    # Create figure with subplots (2x4 layout to fit all plots)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Training Metrics Over Time', fontsize=16, fontweight='bold')
    axes = axes.flatten()  # Flatten for easier indexing
    
    # 1. Loss curves
    ax = axes[0]
    ax.plot(steps, metrics["train_loss"], label='Train Loss', linewidth=2)
    ax.plot(steps, metrics["val_loss"], label='Val Loss', linewidth=2)
    if P_true is not None:
        ax.axhline(y=best_loss, color='green', linestyle='--', alpha=0.5, label=f'Best possible={best_loss:.4f}')
        ax.axhline(y=math.log(V), color='red', linestyle='--', alpha=0.5, label=f'Uniform={math.log(V):.4f}')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. KL Divergence
    ax = axes[1]
    ax.plot(steps, metrics["kl_divergence"], color='green', linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL(P_true || P_model)')
    ax.grid(True, alpha=0.3)
    
    # 3. Distance Correlation
    ax = axes[2]
    ax.plot(steps, metrics["distance_correlation"], color='purple', linewidth=2)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect correlation')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Correlation')
    ax.set_title('Distance Correlation (Torus vs K-space)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    # 4. Neighborhood Overlap
    ax = axes[3]
    ax.plot(steps, metrics["neighborhood_overlap"], color='orange', linewidth=2)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect overlap')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Overlap')
    ax.set_title('Neighborhood Overlap (k=3)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    # 5. Learned Radii
    ax = axes[4]
    ax.plot(steps, metrics["R_hat"], label='R_hat (learned)', linewidth=2, color='blue')
    ax.plot(steps, metrics["r_hat"], label='r_hat (learned)', linewidth=2, color='cyan')
    ax.axhline(y=R, color='blue', linestyle='--', alpha=0.5, label=f'R_true={R:.1f}')
    ax.axhline(y=r, color='cyan', linestyle='--', alpha=0.5, label=f'r_true={r:.1f}')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Radius')
    ax.set_title('Learned vs True Torus Radii')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Radii Ratio
    ax = axes[5]
    R_hat_array = np.array(metrics["R_hat"])
    r_hat_array = np.array(metrics["r_hat"])
    # Avoid division by zero
    ratio = np.where(r_hat_array > 0, R_hat_array / r_hat_array, 0)
    true_ratio = R / r
    ax.plot(steps, ratio, label='R_hat / r_hat', linewidth=2, color='red')
    ax.axhline(y=true_ratio, color='black', linestyle='--', alpha=0.5, label=f'True ratio={true_ratio:.2f}')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Ratio')
    ax.set_title('Radii Ratio (R/r)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 7. Perplexity (log scale)
    ax = axes[6]
    ax.plot(steps, train_perplexity, label='Train Perplexity', linewidth=2, color='blue')
    ax.plot(steps, val_perplexity, label='Val Perplexity', linewidth=2, color='orange')
    if P_true is not None:
        ax.axhline(y=best_perplexity, color='green', linestyle='--', alpha=0.5, label=f'Best possible={best_perplexity:.2f}')
        ax.axhline(y=uniform_perplexity, color='red', linestyle='--', alpha=0.5, label=f'Uniform={uniform_perplexity:.1f}')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Perplexity (log scale)')
    ax.set_title('Perplexity Over Time')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Hide the 8th subplot (unused)
    axes[7].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plots saved to: {save_path}")
    plt.show()

def load_latest_model(checkpoint_dir="checkpoints"):
    """Load the most recent final model and corresponding test sequences."""
    model_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("model_final_") and f.endswith(".pt")]
    
    if not model_files:
        raise FileNotFoundError(f"No final model files found in {checkpoint_dir}/")
    
    # Sort by modification time (most recent first)
    model_files.sort(key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)), reverse=True)
    latest_file = model_files[0]
    
    model_path = os.path.join(checkpoint_dir, latest_file)
    print(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # Try to load corresponding test sequences
    run_id = latest_file.replace("model_final_", "").replace(".pt", "")
    test_sequences_path = os.path.join(checkpoint_dir, f"test_sequences_{run_id}.pt")
    
    test_sequences = None
    if os.path.exists(test_sequences_path):
        test_sequences = torch.load(test_sequences_path, map_location=DEVICE)
        print(f"Loaded test sequences from: {test_sequences_path}")
    else:
        print(f"Warning: Test sequences not found at {test_sequences_path}, will generate new ones.")
    
    return checkpoint, test_sequences

def evaluate_on_test_set(model, test_loader, P_true, D_torus):
    """Evaluate model on test set and return all metrics."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    
    test_losses = []
    V = P_true.shape[0]
    
    with torch.no_grad():
        for test_input, test_target in test_loader:
            test_input = test_input.to(DEVICE)
            test_target = test_target.to(DEVICE)
            test_logits = model(test_input)
            test_loss = loss_fn(test_logits.view(-1, V), test_target.view(-1))
            test_losses.append(test_loss.item())
    
    avg_test_loss = sum(test_losses) / len(test_losses)
    
    # Compute geometry metrics
    Q_tok, K_tok = model.token_qk_matrices()
    P_model = p_model_from_qk(Q_tok, K_tok)
    KL_avg = kl_p_true_p_model(P_true, P_model)
    D_K = kspace_pairwise_distances(K_tok)
    corr = distance_correlation(D_torus, D_K)
    overlap = neighborhood_overlap(D_torus, D_K, k=3)
    R_hat, r_hat = learned_radii(K_tok)
    
    return {
        'test_loss': avg_test_loss,
        'kl_divergence': KL_avg,
        'distance_correlation': corr,
        'neighborhood_overlap': overlap,
        'R_hat': R_hat,
        'r_hat': r_hat
    }

def compute_best_possible_perplexity(P_true):
    """
    Compute the theoretical best possible perplexity from P_true.
    This is the entropy of the true transition distribution.
    """
    # Compute entropy for each state
    entropies = []
    for i in range(V):
        probs = P_true[i]
        # Entropy: H = -sum(p * log(p)) for p > 0
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        entropies.append(entropy.item())
    
    # Average entropy (best possible cross-entropy loss)
    avg_entropy = sum(entropies) / len(entropies)
    
    # Best possible perplexity
    best_perplexity = math.exp(avg_entropy)
    
    return avg_entropy, best_perplexity

def print_final_metrics(metrics, test_metrics=None, P_true=None):
    """Print final metrics summary."""
    print("\n" + "="*60)
    print("FINAL METRICS SUMMARY")
    print("="*60)
    
    final_idx = -1
    print(f"Final Step: {metrics['steps'][final_idx]}")
    
    # Compute best possible metrics if P_true is provided
    if P_true is not None:
        best_loss, best_perplexity = compute_best_possible_perplexity(P_true)
        uniform_loss = math.log(V)
        uniform_perplexity = V
    
    print(f"\nLoss (from training):")
    print(f"  Train Loss: {metrics['train_loss'][final_idx]:.4f}")
    print(f"  Val Loss: {metrics['val_loss'][final_idx]:.4f}")
    
    if test_metrics:
        print(f"  Test Loss: {test_metrics['test_loss']:.4f}  (fresh evaluation)")
    
    print(f"\nPerplexity:")
    train_perplexity = math.exp(metrics['train_loss'][final_idx])
    val_perplexity = math.exp(metrics['val_loss'][final_idx])
    print(f"  Train Perplexity: {train_perplexity:.4f}")
    print(f"  Val Perplexity: {val_perplexity:.4f}")
    if test_metrics:
        test_perplexity = math.exp(test_metrics['test_loss'])
        print(f"  Test Perplexity: {test_perplexity:.4f}")
    
    if P_true is not None:
        print(f"\nTheoretical Bounds:")
        print(f"  Uniform baseline: {uniform_loss:.4f} loss, {uniform_perplexity:.2f} perplexity")
        print(f"  Best possible: {best_loss:.4f} loss, {best_perplexity:.4f} perplexity")
        print(f"  Room for improvement: {metrics['val_loss'][final_idx] - best_loss:.4f} loss units")
        print(f"  Efficiency: {(uniform_loss - metrics['val_loss'][final_idx]) / (uniform_loss - best_loss) * 100:.1f}% of possible improvement")
    
    print(f"\nGeometry Metrics (from training):")
    print(f"  KL Divergence: {metrics['kl_divergence'][final_idx]:.4f}")
    print(f"  Distance Correlation: {metrics['distance_correlation'][final_idx]:.4f}")
    print(f"  Neighborhood Overlap: {metrics['neighborhood_overlap'][final_idx]:.4f}")
    
    if test_metrics:
        print(f"\nGeometry Metrics (on test set):")
        print(f"  KL Divergence: {test_metrics['kl_divergence']:.4f}")
        print(f"  Distance Correlation: {test_metrics['distance_correlation']:.4f}")
        print(f"  Neighborhood Overlap: {test_metrics['neighborhood_overlap']:.4f}")
    
    print(f"\nLearned Radii (from training):")
    print(f"  R_hat: {metrics['R_hat'][final_idx]:.4f}  (true R = {R:.1f})")
    print(f"  r_hat: {metrics['r_hat'][final_idx]:.4f}  (true r = {r:.1f})")
    print(f"  Ratio R/r: {metrics['R_hat'][final_idx]/metrics['r_hat'][final_idx]:.4f}  (true = {R/r:.2f})")
    
    if test_metrics:
        print(f"\nLearned Radii (on test set):")
        print(f"  R_hat: {test_metrics['R_hat']:.4f}  (true R = {R:.1f})")
        print(f"  r_hat: {test_metrics['r_hat']:.4f}  (true r = {r:.1f})")
        print(f"  Ratio R/r: {test_metrics['R_hat']/test_metrics['r_hat']:.4f}  (true = {R/r:.2f})")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    # Load metrics from training
    metrics = load_latest_metrics()
    
    # Load final model and evaluate on test set
    print("\nEvaluating on held-out test set...")
    checkpoint, test_sequences = load_latest_model()
    
    # Reconstruct model
    model = TinyTorusTransformer().to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create test set (use pre-generated sequences if available)
    P_true = make_p_true().to(DEVICE)
    if test_sequences is None:
        # Fallback: generate new test sequences (not ideal for reproducibility)
        print("Generating new test sequences (not using saved ones)...")
        from data import generate_sequences
        test_sequences = generate_sequences(P_true, N_TEST, SEQ_LEN, seed=44)
    
    _, _, test_loader = make_dataloaders(
        P_true, BATCH_SIZE, 
        include_test=True,
        test_sequences=test_sequences
    )
    D_torus = torus_pairwise_distances().to(DEVICE)
    
    # Evaluate on test set
    test_metrics = evaluate_on_test_set(model, test_loader, P_true, D_torus)
    
    # Print final metrics (including test set and theoretical bounds)
    print_final_metrics(metrics, test_metrics, P_true=P_true)
    
    # Generate plots (with theoretical bounds)
    plot_metrics(metrics, P_true=P_true)
