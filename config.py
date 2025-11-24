# config.py
import torch

# Grid & vocab
GRID_SIZE = 4
V = 16

# Model
D_MODEL = 4
D_HEAD = 2
D_FF = 16  # Feed-forward dimension (typically 4 * D_MODEL)
ROPE_BASE = 10000  # Base frequency for rotary positional encoding

# Data
SEQ_LEN = 16
N_TRAIN = 50_000
N_VAL = 10_000
N_TEST = 10_000  # Held-out test set for final evaluation

# Training
BATCH_SIZE = 256
LR = 5e-4  # Conservative LR for tiny network (D_MODEL=4, D_HEAD=2)
NUM_STEPS = 20_000  # Reduced from 20k - model converges much earlier

# Torus kernel
R = 10.0
r = 1.0
SIGMA = 1.0

# Markov chain model type
# "global": All states reachable, probabilities decay with distance (Gaussian)
# "local": Only immediate 8 neighbors reachable, uniform probabilities
MARKOV_MODEL_TYPE = "global"  # Change to "local" to use neighborhood-only model

# Device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

