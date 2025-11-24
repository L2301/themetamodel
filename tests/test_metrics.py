# tests/test_metrics.py
import sys
import os
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import V, D_HEAD, R, r
from torus_markov import make_p_true
from metrics import p_model_from_qk, kl_p_true_p_model, learned_radii

def test_p_model_row_stochastic():
    """Test that P_model is row-stochastic"""
    # Create random Q_tok and K_tok
    Q_tok = torch.randn(V, D_HEAD)
    K_tok = torch.randn(V, D_HEAD)
    
    P_model = p_model_from_qk(Q_tok, K_tok)
    
    assert P_model.shape == (V, V), f"P_model shape should be ({V}, {V}), got {P_model.shape}"
    
    # Check rows sum to 1
    row_sums = P_model.sum(dim=1)
    for i in range(V):
        assert abs(row_sums[i].item() - 1.0) < 1e-6, \
            f"Row {i} should sum to 1, got {row_sums[i].item()}"
    
    # Check all entries >= 0
    assert (P_model >= 0).all(), "All entries in P_model should be >= 0"

def test_kl_zero_for_equal():
    """Test that KL divergence is zero when distributions are equal"""
    P_true = make_p_true()
    
    # Use same distribution for both
    P_model = P_true.clone()
    
    kl = kl_p_true_p_model(P_true, P_model)
    
    assert abs(kl) < 1e-6, \
        f"KL divergence should be ~0 for identical distributions, got {kl}"

def test_learned_radii_consistency():
    """Test learned_radii function with a constructed K_tok"""
    # Create a K_tok that roughly matches torus geometry
    # Map each token to coordinates, then scale appropriately
    from torus_markov import idx_to_coord
    
    K_tok = torch.zeros(V, D_HEAD)
    d_h = R / 4  # horizontal step length
    d_v = r / 4  # vertical step length
    
    for i in range(V):
        row, col = idx_to_coord(i)
        # Map to 2D space (rough approximation)
        K_tok[i, 0] = row * d_h
        K_tok[i, 1] = col * d_v
    
    R_hat, r_hat = learned_radii(K_tok)
    
    # Check that learned radii are reasonable
    # Note: The test K_tok mapping is approximate, so we allow wider bounds
    assert R_hat > 0, f"R_hat should be > 0, got {R_hat}"
    assert r_hat > 0, f"r_hat should be > 0, got {r_hat}"
    # Allow up to 4x for approximate mapping
    assert R_hat < R * 4, f"R_hat ({R_hat}) should be reasonable compared to R ({R})"
    assert r_hat < r * 4, f"r_hat ({r_hat}) should be reasonable compared to r ({r})"

