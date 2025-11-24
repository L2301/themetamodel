# tests/test_model.py
import sys
import os
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import V, D_MODEL, D_HEAD, SEQ_LEN
from model import TinyTorusTransformer

def test_forward_shapes():
    """Test that forward pass returns correct shapes"""
    model = TinyTorusTransformer()
    B = 2
    T = SEQ_LEN - 1
    
    x = torch.randint(0, V, (B, T))
    logits = model(x)
    
    assert logits.shape == (B, T, V), \
        f"logits shape should be ({B}, {T}, {V}), got {logits.shape}"

def test_token_qk_shapes():
    """Test that token_qk_matrices returns correct shapes"""
    model = TinyTorusTransformer()
    Q_tok, K_tok = model.token_qk_matrices()
    
    assert Q_tok.shape == (V, D_HEAD), \
        f"Q_tok shape should be ({V}, {D_HEAD}), got {Q_tok.shape}"
    assert K_tok.shape == (V, D_HEAD), \
        f"K_tok shape should be ({V}, {D_HEAD}), got {K_tok.shape}"

def test_no_nan_outputs():
    """Test that model outputs are finite"""
    model = TinyTorusTransformer()
    B = 2
    T = SEQ_LEN - 1
    
    x = torch.randint(0, V, (B, T))
    
    # Run a few forward passes
    for _ in range(5):
        logits = model(x)
        assert torch.isfinite(logits).all(), \
            "Model outputs should be finite (no NaN or Inf)"

