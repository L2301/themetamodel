# train_e1.py
import torch
import torch.nn as nn
from torch.optim import Adam
import json
import os
import argparse
from datetime import datetime
from tqdm import tqdm

from config import *
from torus_markov import make_p_true
from data import make_dataloaders, generate_sequences
from model import TinyTorusTransformer
from metrics import p_model_from_qk, kl_p_true_p_model, \
                    torus_pairwise_distances, kspace_pairwise_distances, \
                    distance_correlation, neighborhood_overlap, learned_radii

def generate_experiment_tags(model, markov_type=None, R_val=None, r_val=None):
    """
    Generate semantic tags for the experiment configuration.
    Returns a string like: "mlp_ln_global_R10r1"
    
    Args:
        model: The model instance
        markov_type: Override MARKOV_MODEL_TYPE if provided
        R_val: Override R if provided
        r_val: Override r if provided
    """
    tags = []
    
    # MLP presence
    has_mlp = hasattr(model, 'mlp') and model.mlp is not None
    tags.append("mlp" if has_mlp else "nomlp")
    
    # LayerNorm presence
    has_ln = (hasattr(model, 'ln1') or hasattr(model, 'ln')) and \
             (model.ln1 is not None if hasattr(model, 'ln1') else model.ln is not None)
    tags.append("ln" if has_ln else "noln")
    
    # Markov model type
    markov = markov_type if markov_type is not None else MARKOV_MODEL_TYPE
    tags.append(markov)  # "global" or "local"
    
    # R/r ratio
    R_tag = R_val if R_val is not None else R
    r_tag = r_val if r_val is not None else r
    tags.append(f"R{R_tag:.0f}r{r_tag:.0f}")
    
    return "_".join(tags)

def train(args=None):
    """
    Train the model with optional command-line arguments.
    
    Args:
        args: argparse.Namespace with experiment configuration, or None to use defaults
    """
    # Parse arguments or use defaults
    if args is None:
        args = argparse.Namespace(
            no_mlp=False,
            no_layernorm=False,
            markov_type=None,
            R=None,
            r=None
        )
    
    # Override config values from args
    use_mlp = not args.no_mlp
    use_layernorm = not args.no_layernorm
    markov_type = args.markov_type if args.markov_type else MARKOV_MODEL_TYPE
    R_val = args.R if args.R is not None else R
    r_val = args.r if args.r is not None else r
    
    # Temporarily override config for torus_markov
    import config as config_module
    original_markov_type = config_module.MARKOV_MODEL_TYPE
    original_R = config_module.R
    original_r = config_module.r
    config_module.MARKOV_MODEL_TYPE = markov_type
    config_module.R = R_val
    config_module.r = r_val
    
    # Create checkpoints directory
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create model with specified configuration
    model = TinyTorusTransformer(use_mlp=use_mlp, use_layernorm=use_layernorm).to(DEVICE)
    
    # Generate experiment tags
    exp_tags = generate_experiment_tags(model, markov_type=markov_type, R_val=R_val, r_val=r_val)
    
    # Create run timestamp with tags
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{exp_tags}_{timestamp}"
    
    # Print experiment configuration
    print("\n" + "="*60)
    print("EXPERIMENT CONFIGURATION")
    print("="*60)
    print(f"Experiment Tags: {exp_tags}")
    print(f"  - MLP: {'Yes' if use_mlp else 'No'}")
    print(f"  - LayerNorm: {'Yes' if use_layernorm else 'No'}")
    print(f"  - Markov Model: {markov_type}")
    print(f"  - Torus Radii: R={R_val:.1f}, r={r_val:.1f} (ratio={R_val/r_val:.1f})")
    print(f"  - Model Size: D_MODEL={D_MODEL}, D_HEAD={D_HEAD}, D_FF={D_FF if 'D_FF' in globals() else 'N/A'}")
    print(f"Run ID: {run_id}")
    print("="*60 + "\n")
    
    # Initialize metrics history
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
    
    # Use overridden config values for P_true
    P_true = make_p_true().to(DEVICE)
    
    # Restore original config values
    config_module.MARKOV_MODEL_TYPE = original_markov_type
    config_module.R = original_R
    config_module.r = original_r
    
    # Pre-generate all sequences before training for fixed train/val/test splits
    print("Generating sequences for train/val/test sets...")
    train_sequences = generate_sequences(P_true, N_TRAIN, SEQ_LEN, seed=42)
    val_sequences = generate_sequences(P_true, N_VAL, SEQ_LEN, seed=43)
    test_sequences = generate_sequences(P_true, N_TEST, SEQ_LEN, seed=44)
    print("Done generating sequences.")
    
    train_loader, val_loader = make_dataloaders(
        P_true, BATCH_SIZE, 
        include_test=False,
        train_sequences=train_sequences,
        val_sequences=val_sequences
    )

    # Model already created above for tag generation, now initialize optimizer
    opt = Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    D_torus = torus_pairwise_distances().to(DEVICE)

    # Create iterator for training batches
    train_iter = iter(train_loader)

    # Progress bar
    pbar = tqdm(range(NUM_STEPS), desc="Training", unit="step")
    
    for step in pbar:
        # 1) sample batch, do forward/backward, update
        model.train()
        opt.zero_grad()
        
        # Get batch (recreate iterator if exhausted)
        try:
            input_seq, target_seq = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            input_seq, target_seq = next(train_iter)
        
        input_seq = input_seq.to(DEVICE)
        target_seq = target_seq.to(DEVICE)
        
        # Forward pass
        logits = model(input_seq)  # [B, T, V]
        
        # Compute loss (flatten for CrossEntropyLoss)
        loss = loss_fn(logits.view(-1, V), target_seq.view(-1))
        
        # Backward pass
        loss.backward()
        opt.step()
        
        # Update progress bar with current loss
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 2) periodically compute metrics
        if (step + 1) % 500 == 0 or step == 0:
            pbar.set_description("Computing metrics...")
            model.eval()
            with torch.no_grad():
                # Compute validation loss (with progress bar)
                val_losses = []
                val_pbar = tqdm(val_loader, desc="  Validating", leave=False, unit="batch")
                for val_input, val_target in val_pbar:
                    val_input = val_input.to(DEVICE)
                    val_target = val_target.to(DEVICE)
                    val_logits = model(val_input)
                    val_loss = loss_fn(val_logits.view(-1, V), val_target.view(-1))
                    val_losses.append(val_loss.item())
                avg_val_loss = sum(val_losses) / len(val_losses)
                
                # Get token-level Q/K matrices
                Q_tok, K_tok = model.token_qk_matrices()
                
                # Build P_model
                P_model = p_model_from_qk(Q_tok, K_tok)
                
                # Compute KL divergence
                KL_avg = kl_p_true_p_model(P_true, P_model)
                
                # Compute K-space distances (now vectorized - much faster!)
                D_K = kspace_pairwise_distances(K_tok)
                
                # Compute distance correlation
                corr = distance_correlation(D_torus, D_K)
                
                # Compute neighborhood overlap
                overlap = neighborhood_overlap(D_torus, D_K, k=3)
                
                # Compute learned radii
                R_hat, r_hat = learned_radii(K_tok)
                
                # Save metrics to history
                metrics_history["steps"].append(step + 1)
                metrics_history["train_loss"].append(loss.item())
                metrics_history["val_loss"].append(avg_val_loss)
                metrics_history["kl_divergence"].append(KL_avg)
                metrics_history["distance_correlation"].append(corr)
                metrics_history["neighborhood_overlap"].append(overlap)
                metrics_history["R_hat"].append(R_hat)
                metrics_history["r_hat"].append(r_hat)
                
                # Log metrics
                pbar.write(f"\nStep {step + 1}/{NUM_STEPS}")
                pbar.write(f"  Train Loss: {loss.item():.4f}")
                pbar.write(f"  Val Loss: {avg_val_loss:.4f}")
                pbar.write(f"  KL(P_true || P_model): {KL_avg:.4f}")
                pbar.write(f"  Distance Correlation: {corr:.4f}")
                pbar.write(f"  Neighborhood Overlap (k=3): {overlap:.4f}")
                pbar.write(f"  Learned Radii: R_hat={R_hat:.4f}, r_hat={r_hat:.4f}")
                pbar.write(f"  True Radii: R={R_val:.1f}, r={r_val:.1f}")
                
                # Save model checkpoint with tags
                checkpoint_path = os.path.join(checkpoint_dir, f"model_step_{step + 1}_{exp_tags}.pt")
                torch.save({
                    'step': step + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'train_loss': loss.item(),
                    'val_loss': avg_val_loss,
                    'experiment_tags': exp_tags,
                    'config': {
                        'GRID_SIZE': GRID_SIZE,
                        'V': V,
                        'D_MODEL': D_MODEL,
                        'D_HEAD': D_HEAD,
                        'D_FF': D_FF if 'D_FF' in globals() else None,
                        'R': R_val,
                        'r': r_val,
                        'SIGMA': SIGMA,
                        'MARKOV_MODEL_TYPE': markov_type,
                        'has_mlp': hasattr(model, 'mlp') and model.mlp is not None,
                        'has_layernorm': hasattr(model, 'ln1') or hasattr(model, 'ln'),
                    }
                }, checkpoint_path)
                pbar.write(f"  Saved checkpoint: {checkpoint_path}\n")
                pbar.set_description("Training")

    # Save final model with tags
    final_model_path = os.path.join(checkpoint_dir, f"model_final_{run_id}.pt")
    torch.save({
        'step': NUM_STEPS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'experiment_tags': exp_tags,
        'config': {
            'GRID_SIZE': GRID_SIZE,
            'V': V,
            'D_MODEL': D_MODEL,
            'D_HEAD': D_HEAD,
            'D_FF': D_FF if 'D_FF' in globals() else None,
            'R': R_val,
            'r': r_val,
            'SIGMA': SIGMA,
            'MARKOV_MODEL_TYPE': markov_type,
            'has_mlp': hasattr(model, 'mlp') and model.mlp is not None,
            'has_layernorm': hasattr(model, 'ln1') or hasattr(model, 'ln'),
        }
    }, final_model_path)
    print(f"Training complete! Final model saved to: {final_model_path}")
    
    # Save metrics history with tags (backward compatible structure)
    metrics_path = os.path.join(checkpoint_dir, f"metrics_history_{run_id}.json")
    # Add metadata at top level, but keep metrics_history structure for backward compatibility
    metrics_history['experiment_tags'] = exp_tags
    metrics_history['config'] = {
        'GRID_SIZE': GRID_SIZE,
        'V': V,
        'D_MODEL': D_MODEL,
        'D_HEAD': D_HEAD,
        'D_FF': D_FF if 'D_FF' in globals() else None,
        'R': R_val,
        'r': r_val,
        'SIGMA': SIGMA,
        'MARKOV_MODEL_TYPE': markov_type,
        'has_mlp': hasattr(model, 'mlp') and model.mlp is not None,
        'has_layernorm': hasattr(model, 'ln1') or hasattr(model, 'ln'),
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_history, f, indent=2)
    print(f"Metrics history saved to: {metrics_path}")
    
    # Save test sequences for evaluation with tags
    test_sequences_path = os.path.join(checkpoint_dir, f"test_sequences_{run_id}.pt")
    torch.save(test_sequences, test_sequences_path)
    print(f"Test sequences saved to: {test_sequences_path} (for eval.py)")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train TinyTorusTransformer with configurable architecture',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: MLP + LayerNorm + Global + R=10, r=1
  python train_e1.py
  
  # No MLP, no LayerNorm
  python train_e1.py --no-mlp --no-layernorm
  
  # Local Markov model with different radii
  python train_e1.py --markov-type local --R 5 --r 1
  
  # All combinations
  python train_e1.py --no-mlp --markov-type local --R 20 --r 2
        """
    )
    
    parser.add_argument('--no-mlp', action='store_true',
                        help='Disable MLP feed-forward network')
    parser.add_argument('--no-layernorm', action='store_true',
                        help='Disable LayerNorm')
    parser.add_argument('--markov-type', type=str, choices=['global', 'local'],
                        default=None,
                        help='Markov model type: global (distance-weighted) or local (neighbors only)')
    parser.add_argument('--R', type=float, default=None,
                        help='Horizontal torus radius R (default: from config.py)')
    parser.add_argument('--r', type=float, default=None,
                        help='Vertical torus radius r (default: from config.py)')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
