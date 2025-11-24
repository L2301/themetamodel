# tests/test_torus_markov.py
import sys
import os
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import V, GRID_SIZE, R, r
from torus_markov import idx_to_coord, coord_to_idx, torus_distance_matrix, make_p_true

def test_idx_coord_roundtrip():
    """Test that coord_to_idx(*idx_to_coord(i)) == i for all i"""
    for i in range(V):
        row, col = idx_to_coord(i)
        i_recovered = coord_to_idx(row, col)
        assert i_recovered == i, f"Roundtrip failed for i={i}: got {i_recovered}"

def test_torus_distance_loops():
    """Test that loop distances match expected torus radii"""
    D = torus_distance_matrix()
    
    # Horizontal loop: [4, 5, 6, 7, 4]
    horizontal_loop = [4, 5, 6, 7, 4]
    total_horizontal = 0.0
    for idx in range(len(horizontal_loop) - 1):
        i, j = horizontal_loop[idx], horizontal_loop[idx + 1]
        total_horizontal += D[i, j].item()
    
    assert abs(total_horizontal - R) < 1e-6, \
        f"Horizontal loop distance ({total_horizontal}) should equal R ({R})"
    
    # Vertical loop: [1, 5, 9, 13, 1]
    vertical_loop = [1, 5, 9, 13, 1]
    total_vertical = 0.0
    for idx in range(len(vertical_loop) - 1):
        i, j = vertical_loop[idx], vertical_loop[idx + 1]
        total_vertical += D[i, j].item()
    
    assert abs(total_vertical - r) < 1e-6, \
        f"Vertical loop distance ({total_vertical}) should equal r ({r})"

def test_p_true_row_stochastic():
    """Test that P_true is row-stochastic"""
    P_true = make_p_true()
    
    assert P_true.shape == (V, V), f"P_true shape should be ({V}, {V}), got {P_true.shape}"
    
    # Check diagonal is zero
    for i in range(V):
        assert abs(P_true[i, i].item()) < 1e-6, \
            f"P_true[{i}, {i}] should be 0, got {P_true[i, i].item()}"
    
    # Check rows sum to 1
    row_sums = P_true.sum(dim=1)
    for i in range(V):
        assert abs(row_sums[i].item() - 1.0) < 1e-6, \
            f"Row {i} should sum to 1, got {row_sums[i].item()}"

