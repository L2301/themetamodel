# tests/test_data.py
import sys
import os
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SEQ_LEN, V
from torus_markov import make_p_true
from data import MarkovSequenceDataset, make_dataloaders

def test_dataset_shapes():
    """Test that dataset returns correct shapes"""
    P_true = make_p_true()
    ds = MarkovSequenceDataset(P_true, num_seqs=10, seq_len=SEQ_LEN)
    
    input_seq, target_seq = ds[0]
    
    assert input_seq.shape == (SEQ_LEN - 1,), \
        f"input_seq shape should be ({SEQ_LEN - 1},), got {input_seq.shape}"
    assert target_seq.shape == (SEQ_LEN - 1,), \
        f"target_seq shape should be ({SEQ_LEN - 1},), got {target_seq.shape}"
    
    assert input_seq.dtype == torch.long, "input_seq should be LongTensor"
    assert target_seq.dtype == torch.long, "target_seq should be LongTensor"

def test_markov_consistency():
    """Test that dataset samples follow Markov chain approximately"""
    P_true = make_p_true()
    
    # Use smaller numbers for faster test
    num_seqs = 2000
    seq_len = 4
    
    ds = MarkovSequenceDataset(P_true, num_seqs=num_seqs, seq_len=seq_len)
    
    # Build empirical transition counts
    transition_counts = torch.zeros(V, V)
    
    for i in range(num_seqs):
        input_seq, target_seq = ds[i]
        for t in range(len(input_seq)):
            from_state = input_seq[t].item()
            to_state = target_seq[t].item()
            transition_counts[from_state, to_state] += 1
    
    # Normalize to get empirical transition matrix
    row_sums = transition_counts.sum(dim=1, keepdim=True)
    row_sums = torch.where(row_sums > 0, row_sums, torch.ones_like(row_sums))  # avoid division by zero
    P_emp = transition_counts / row_sums
    
    # Check that empirical and true transition matrices are correlated
    # (coarse check: average absolute difference should be reasonable)
    diff = torch.abs(P_emp - P_true)
    avg_diff = diff.mean().item()
    
    # Allow for some sampling variance - average diff should be < 0.2
    assert avg_diff < 0.2, \
        f"Average absolute difference between P_emp and P_true ({avg_diff}) should be < 0.2"

