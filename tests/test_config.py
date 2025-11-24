# tests/test_config.py
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GRID_SIZE, V, D_MODEL, D_HEAD, SEQ_LEN, N_TRAIN, N_VAL, R, r, SIGMA

def test_v_equals_grid_size_squared():
    """Test that V == GRID_SIZE * GRID_SIZE"""
    assert V == GRID_SIZE * GRID_SIZE, f"V ({V}) should equal GRID_SIZE^2 ({GRID_SIZE * GRID_SIZE})"

def test_d_model_ge_2():
    """Test that D_MODEL >= 2"""
    assert D_MODEL >= 2, f"D_MODEL ({D_MODEL}) should be >= 2"

def test_d_head_equals_2():
    """Test that D_HEAD == 2"""
    assert D_HEAD == 2, f"D_HEAD ({D_HEAD}) should equal 2"

def test_seq_len_gt_1():
    """Test that SEQ_LEN > 1"""
    assert SEQ_LEN > 1, f"SEQ_LEN ({SEQ_LEN}) should be > 1"

def test_n_train_gt_0():
    """Test that N_TRAIN > 0"""
    assert N_TRAIN > 0, f"N_TRAIN ({N_TRAIN}) should be > 0"

def test_n_val_gt_0():
    """Test that N_VAL > 0"""
    assert N_VAL > 0, f"N_VAL ({N_VAL}) should be > 0"

def test_torus_params():
    """Test that torus parameters are positive"""
    assert R > 0, f"R ({R}) should be > 0"
    assert r > 0, f"r ({r}) should be > 0"
    assert SIGMA > 0, f"SIGMA ({SIGMA}) should be > 0"

