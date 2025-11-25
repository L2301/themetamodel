# sweep_e1.py
"""
Sweep through all experiment configurations and generate plots for each.
Specifically does a radii sweep: R from 1-10, r from 1-10 (100 configurations).
For each radii combination, tests all combinations of MLP/LN/Markov type.
"""
import torch
import torch.nn as nn
from torch.optim import Adam
import json
import os
import argparse
from datetime import datetime
from tqdm import tqdm
import itertools

from config import *
from torus_markov import make_p_true
from data import make_dataloaders, generate_sequences
from model import TinyTorusTransformer
from metrics import p_model_from_qk, kl_p_true_p_model, \
                    torus_pairwise_distances, kspace_pairwise_distances, \
                    distance_correlation, neighborhood_overlap, learned_radii
from train_e1 import generate_experiment_tags

def train_single_config(use_mlp, use_layernorm, markov_type, R_val, r_val, 
                        num_steps=10000, save_checkpoints=False, output_dir="sweep_results",
                        P_true=None, D_torus=None, train_sequences=None, val_sequences=None):
    """
    Train a single configuration and return final metrics.
    
    Args:
        use_mlp: Whether to use MLP
        use_layernorm: Whether to use LayerNorm
        markov_type: 'global' or 'local'
        R_val: Horizontal torus radius
        r_val: Vertical torus radius
        num_steps: Number of training steps
        save_checkpoints: Whether to save model checkpoints
        output_dir: Directory to save results
        P_true: Pre-computed P_true matrix (optional, will generate if None)
        D_torus: Pre-computed torus distance matrix (optional, will generate if None)
        train_sequences: Pre-generated train sequences (optional, will generate if None)
        val_sequences: Pre-generated val sequences (optional, will generate if None)
    
    Returns:
        dict with final metrics and experiment tags
    """
    # Temporarily override config
    import config as config_module
    original_markov_type = config_module.MARKOV_MODEL_TYPE
    original_R = config_module.R
    original_r = config_module.r
    config_module.MARKOV_MODEL_TYPE = markov_type
    config_module.R = R_val
    config_module.r = r_val
    
    try:
        # Create model
        model = TinyTorusTransformer(use_mlp=use_mlp, use_layernorm=use_layernorm).to(DEVICE)
        opt = Adam(model.parameters(), lr=LR)
        loss_fn = nn.CrossEntropyLoss()
        
        # Generate tags
        exp_tags = generate_experiment_tags(model, markov_type=markov_type, R_val=R_val, r_val=r_val)
        
        # Create P_true and D_torus if not provided
        if P_true is None:
            P_true = make_p_true().to(DEVICE)
        if D_torus is None:
            D_torus = torus_pairwise_distances().to(DEVICE)
        
        # Generate sequences if not provided
        if train_sequences is None:
            train_sequences = generate_sequences(P_true, N_TRAIN, SEQ_LEN, seed=42)
        if val_sequences is None:
            val_sequences = generate_sequences(P_true, N_VAL, SEQ_LEN, seed=43)
        
        train_loader, val_loader = make_dataloaders(
            P_true, BATCH_SIZE,
            include_test=False,
            train_sequences=train_sequences,
            val_sequences=val_sequences
        )
        
        # Training loop (simplified - just train for num_steps)
        model.train()
        metrics_history = {
            "steps": [],
            "train_loss": [],
            "val_loss": [],
            "kl_divergence": [],
            "distance_correlation": [],
            "neighborhood_overlap": [],
            "R_hat": [],
            "r_hat": []
        }
        
        for step in range(num_steps):
            # Training step
            for input_seq, target_seq in train_loader:
                input_seq = input_seq.to(DEVICE)
                target_seq = target_seq.to(DEVICE)
                
                opt.zero_grad()
                logits = model(input_seq)
                loss = loss_fn(logits.view(-1, V), target_seq.view(-1))
                loss.backward()
                opt.step()
                break  # Just one batch per step for speed
        
            # Evaluate periodically
            if (step + 1) % (num_steps // 10) == 0 or step == num_steps - 1:
                model.eval()
                with torch.no_grad():
                    # Validation loss
                    val_losses = []
                    for val_input, val_target in val_loader:
                        val_input = val_input.to(DEVICE)
                        val_target = val_target.to(DEVICE)
                        val_logits = model(val_input)
                        val_loss = loss_fn(val_logits.view(-1, V), val_target.view(-1))
                        val_losses.append(val_loss.item())
                    avg_val_loss = sum(val_losses) / len(val_losses)
                    
                    # Geometry metrics
                    Q_tok, K_tok = model.token_qk_matrices()
                    P_model = p_model_from_qk(Q_tok, K_tok)
                    KL_avg = kl_p_true_p_model(P_true, P_model)
                    D_K = kspace_pairwise_distances(K_tok)
                    corr = distance_correlation(D_torus, D_K)
                    overlap = neighborhood_overlap(D_torus, D_K, k=3)
                    R_hat, r_hat = learned_radii(K_tok)
                    
                    metrics_history["steps"].append(step + 1)
                    metrics_history["train_loss"].append(loss.item())
                    metrics_history["val_loss"].append(avg_val_loss)
                    metrics_history["kl_divergence"].append(KL_avg)
                    metrics_history["distance_correlation"].append(corr)
                    metrics_history["neighborhood_overlap"].append(overlap)
                    metrics_history["R_hat"].append(R_hat)
                    metrics_history["r_hat"].append(r_hat)
                
                model.train()
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            Q_tok, K_tok = model.token_qk_matrices()
            P_model = p_model_from_qk(Q_tok, K_tok)
            KL_avg = kl_p_true_p_model(P_true, P_model)
            D_K = kspace_pairwise_distances(K_tok)
            corr = distance_correlation(D_torus, D_K)
            overlap = neighborhood_overlap(D_torus, D_K, k=3)
            R_hat, r_hat = learned_radii(K_tok)
            
            # Final validation loss
            val_losses = []
            for val_input, val_target in val_loader:
                val_input = val_input.to(DEVICE)
                val_target = val_target.to(DEVICE)
                val_logits = model(val_input)
                val_loss = loss_fn(val_logits.view(-1, V), val_target.view(-1))
                val_losses.append(val_loss.item())
            avg_val_loss = sum(val_losses) / len(val_losses)
        
        # Add metadata
        metrics_history['experiment_tags'] = exp_tags
        metrics_history['config'] = {
            'use_mlp': use_mlp,
            'use_layernorm': use_layernorm,
            'markov_type': markov_type,
            'R': R_val,
            'r': r_val,
        }
        
        return {
            'metrics_history': metrics_history,
            'final_metrics': {
                'val_loss': avg_val_loss,
                'kl_divergence': KL_avg,
                'distance_correlation': corr,
                'neighborhood_overlap': overlap,
                'R_hat': R_hat,
                'r_hat': r_hat,
            },
            'exp_tags': exp_tags,
            'model': model,
            'P_true': P_true
        }
    
    finally:
        # Restore original config
        config_module.MARKOV_MODEL_TYPE = original_markov_type
        config_module.R = original_R
        config_module.r = original_r

def plot_metrics_sweep(metrics_history, save_path, P_true=None):
    """Generate plots for a single configuration (adapted from eval.py)."""
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    
    steps = metrics_history["steps"]
    
    # Compute perplexities
    train_perplexity = [math.exp(loss) for loss in metrics_history["train_loss"]]
    val_perplexity = [math.exp(loss) for loss in metrics_history["val_loss"]]
    
    # Compute theoretical bounds if P_true provided
    if P_true is not None:
        from eval import compute_best_possible_perplexity
        best_loss, best_perplexity = compute_best_possible_perplexity(P_true)
        uniform_perplexity = V
    
    # Create figure with subplots (2x4 layout)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Training Metrics: {metrics_history.get("experiment_tags", "unknown")}', 
                 fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    # 1. Loss curves
    ax = axes[0]
    ax.plot(steps, metrics_history["train_loss"], label='Train Loss', linewidth=2)
    ax.plot(steps, metrics_history["val_loss"], label='Val Loss', linewidth=2)
    if P_true is not None:
        ax.axhline(y=best_loss, color='green', linestyle='--', alpha=0.5, label=f'Best={best_loss:.4f}')
        ax.axhline(y=math.log(V), color='red', linestyle='--', alpha=0.5, label=f'Uniform={math.log(V):.4f}')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. KL Divergence
    ax = axes[1]
    ax.plot(steps, metrics_history["kl_divergence"], color='green', linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL(P_true || P_model)')
    ax.grid(True, alpha=0.3)
    
    # 3. Distance Correlation
    ax = axes[2]
    ax.plot(steps, metrics_history["distance_correlation"], color='purple', linewidth=2)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Correlation')
    ax.set_title('Distance Correlation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    # 4. Neighborhood Overlap
    ax = axes[3]
    ax.plot(steps, metrics_history["neighborhood_overlap"], color='orange', linewidth=2)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Overlap')
    ax.set_title('Neighborhood Overlap (k=3)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    # 5. Learned Radii
    ax = axes[4]
    config = metrics_history.get('config', {})
    R_true = config.get('R', R)
    r_true = config.get('r', r)
    ax.plot(steps, metrics_history["R_hat"], label='R_hat', linewidth=2, color='blue')
    ax.plot(steps, metrics_history["r_hat"], label='r_hat', linewidth=2, color='cyan')
    ax.axhline(y=R_true, color='blue', linestyle='--', alpha=0.5, label=f'R_true={R_true:.1f}')
    ax.axhline(y=r_true, color='cyan', linestyle='--', alpha=0.5, label=f'r_true={r_true:.1f}')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Radius')
    ax.set_title('Learned vs True Radii')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Radii Ratio
    ax = axes[5]
    R_hat_array = np.array(metrics_history["R_hat"])
    r_hat_array = np.array(metrics_history["r_hat"])
    ratio = np.where(r_hat_array > 0, R_hat_array / r_hat_array, 0)
    true_ratio = R_true / r_true
    ax.plot(steps, ratio, label='R_hat / r_hat', linewidth=2, color='red')
    ax.axhline(y=true_ratio, color='black', linestyle='--', alpha=0.5, label=f'True={true_ratio:.2f}')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Ratio')
    ax.set_title('Radii Ratio (R/r)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 7. Perplexity (log scale)
    ax = axes[6]
    ax.plot(steps, train_perplexity, label='Train', linewidth=2, color='blue')
    ax.plot(steps, val_perplexity, label='Val', linewidth=2, color='orange')
    if P_true is not None:
        ax.axhline(y=best_perplexity, color='green', linestyle='--', alpha=0.5, label=f'Best={best_perplexity:.2f}')
        ax.axhline(y=uniform_perplexity, color='red', linestyle='--', alpha=0.5, label=f'Uniform={uniform_perplexity:.1f}')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Perplexity (log scale)')
    ax.set_title('Perplexity Over Time')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Hide 8th subplot
    axes[7].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def run_sweep(num_steps=10000, output_dir="sweep_results", radii_only=False):
    """
    Run sweep through all configurations.
    
    Configurations:
    - MLP: True/False (2 options) - skipped if radii_only=True
    - LayerNorm: True/False (2 options) - skipped if radii_only=True
    - Markov type: global/local (2 options) - skipped if radii_only=True
    - R: 1-10 (10 options)
    - r: 1-10 (10 options)
    
    If radii_only=True: Only 100 configurations (R=1-10, r=1-10) with default MLP/LN/Markov
    Otherwise: 2 * 2 * 2 * 10 * 10 = 800 configurations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # All configuration combinations
    if radii_only:
        # Only radii sweep: use default MLP/LN/Markov settings
        mlp_options = [True]  # Default: use MLP
        ln_options = [True]  # Default: use LayerNorm
        markov_options = ['global']  # Default: global
    else:
        mlp_options = [True, False]
        ln_options = [True, False]
        markov_options = ['global', 'local']
    
    R_options = list(range(1, 11))  # 1-10
    r_options = list(range(1, 11))  # 1-10
    
    total_configs = len(mlp_options) * len(ln_options) * len(markov_options) * len(R_options) * len(r_options)
    
    print(f"\n{'='*60}")
    print(f"RUNNING SWEEP")
    print(f"{'='*60}")
    print(f"Total configurations: {total_configs}")
    print(f"Steps per config: {num_steps}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    # Results summary
    results_summary = []
    
    # Group configurations by (markov_type, R, r) for dataset reuse
    # First, get all unique (markov_type, R, r) combinations
    unique_data_configs = list(itertools.product(markov_options, R_options, r_options))
    num_unique_data_configs = len(unique_data_configs)
    
    # All MLP/LN combinations
    mlp_ln_configs = list(itertools.product(mlp_options, ln_options))
    
    print(f"Dataset optimization:")
    print(f"  Unique (markov_type, R, r) combinations: {num_unique_data_configs}")
    print(f"  MLP/LN variations per dataset: {len(mlp_ln_configs)}")
    print(f"  Sequence generations: {num_unique_data_configs} (vs {total_configs} without optimization)")
    print(f"  Speedup: {total_configs / num_unique_data_configs:.1f}x for sequence generation\n")
    
    # Progress tracking
    total_configs_processed = 0
    pbar = tqdm(total=total_configs, desc="Sweeping configurations")
    
    # Import config module once
    import config as config_module
    
    # For each unique (markov_type, R, r) combination
    for markov_type, R_val, r_val in unique_data_configs:
        # Temporarily override config for dataset generation
        original_markov_type = config_module.MARKOV_MODEL_TYPE
        original_R = config_module.R
        original_r = config_module.r
        config_module.MARKOV_MODEL_TYPE = markov_type
        config_module.R = float(R_val)
        config_module.r = float(r_val)
        
        try:
            # Generate datasets once for this (markov_type, R, r) combination
            P_true = make_p_true().to(DEVICE)
            D_torus = torus_pairwise_distances().to(DEVICE)
            train_sequences = generate_sequences(P_true, N_TRAIN, SEQ_LEN, seed=42)
            val_sequences = generate_sequences(P_true, N_VAL, SEQ_LEN, seed=43)
            
            # Now train all MLP/LN variations with these shared datasets
            for use_mlp, use_layernorm in mlp_ln_configs:
                try:
                    # Update progress bar
                    exp_tags = f"{'mlp' if use_mlp else 'nomlp'}_{'ln' if use_layernorm else 'noln'}_{markov_type}_R{R_val}r{r_val}"
                    pbar.set_description(f"Training: {exp_tags}")
                    
                    # Train configuration with pre-generated datasets
                    result = train_single_config(
                        use_mlp=use_mlp,
                        use_layernorm=use_layernorm,
                        markov_type=markov_type,
                        R_val=float(R_val),
                        r_val=float(r_val),
                        num_steps=num_steps,
                        output_dir=output_dir,
                        P_true=P_true,
                        D_torus=D_torus,
                        train_sequences=train_sequences,
                        val_sequences=val_sequences
                    )
                    
                    # Save metrics JSON
                    metrics_file = os.path.join(output_dir, f"metrics_{exp_tags}.json")
                    with open(metrics_file, 'w') as f:
                        json.dump(result['metrics_history'], f, indent=2)
                    
                    # Generate and save plot
                    plot_file = os.path.join(output_dir, f"plot_{exp_tags}.png")
                    plot_metrics_sweep(result['metrics_history'], plot_file, P_true=P_true)
                    
                    # Add to summary
                    results_summary.append({
                        'exp_tags': exp_tags,
                        'config': result['metrics_history']['config'],
                        'final_metrics': result['final_metrics'],
                        'metrics_file': metrics_file,
                        'plot_file': plot_file
                    })
                    
                    total_configs_processed += 1
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\nError in configuration {exp_tags}: {e}")
                    import traceback
                    traceback.print_exc()
                    total_configs_processed += 1
                    pbar.update(1)
                    continue
        
        finally:
            # Restore original config
            config_module.MARKOV_MODEL_TYPE = original_markov_type
            config_module.R = original_R
            config_module.r = original_r
    
    pbar.close()
    
    # Save summary
    summary_file = os.path.join(output_dir, "sweep_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"Summary: {summary_file}")
    print(f"Total configurations completed: {len(results_summary)}/{total_configs}")
    print(f"\nOptimization Summary:")
    print(f"  Unique dataset generations: {num_unique_data_configs}")
    print(f"  Total configurations: {total_configs}")
    print(f"  Dataset reuse factor: {total_configs / num_unique_data_configs:.1f}x")
    print(f"  Estimated time saved: ~{(total_configs - num_unique_data_configs) * 0.1:.1f}s (assuming 0.1s per sequence generation)")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Sweep through all experiment configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Radii sweep only (100 configurations): R=1-10, r=1-10
  python sweep_e1.py --radii-only
  
  # Full sweep (800 configurations): all MLP/LN/Markov combinations + radii
  python sweep_e1.py
  
  # Custom number of steps
  python sweep_e1.py --radii-only --num-steps 500
        """
    )
    parser.add_argument('--num-steps', type=int, default=10000,
                        help='Number of training steps per configuration (default: 1000)')
    parser.add_argument('--output-dir', type=str, default='sweep_results',
                        help='Output directory for results (default: sweep_results)')
    parser.add_argument('--radii-only', action='store_true',
                        help='Only sweep radii (R=1-10, r=1-10) with default MLP/LN/Markov (100 configs)')
    
    args = parser.parse_args()
    run_sweep(num_steps=args.num_steps, output_dir=args.output_dir, radii_only=args.radii_only)

