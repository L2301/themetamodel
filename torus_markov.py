# torus_markov.py
import torch
from config import GRID_SIZE, SIGMA, V, R, r, MARKOV_MODEL_TYPE

def idx_to_coord(i: int) -> tuple[int, int]:
    """Map index 0..V-1 to (row, col) coordinates."""
    row = i // GRID_SIZE
    col = i % GRID_SIZE
    return (row, col)

def coord_to_idx(row: int, col: int) -> int:
    """Map (row, col) coordinates to index, with wrap-around."""
    row = row % GRID_SIZE
    col = col % GRID_SIZE
    return row * GRID_SIZE + col

def torus_step_counts(i: int, j: int) -> tuple[int, int]:
    """Return (s_row, s_col) = minimal wrapped steps between i and j."""
    row_i, col_i = idx_to_coord(i)
    row_j, col_j = idx_to_coord(j)
    
    # Compute absolute differences
    delta_row_abs = abs(row_i - row_j)
    delta_col_abs = abs(col_i - col_j)
    
    # Take minimum of forward and backward wrap
    s_row = min(delta_row_abs, GRID_SIZE - delta_row_abs)
    s_col = min(delta_col_abs, GRID_SIZE - delta_col_abs)
    
    return (s_row, s_col)

def torus_distance_matrix() -> torch.Tensor:
    """Return D[i, j] = torus distance between states i, j."""
    D = torch.zeros(V, V)
    
    # Step lengths
    d_h = R / GRID_SIZE  # horizontal step length
    d_v = r / GRID_SIZE  # vertical step length
    
    for i in range(V):
        for j in range(V):
            if i == j:
                D[i, j] = 0.0
            else:
                s_row, s_col = torus_step_counts(i, j)
                # Euclidean distance on torus
                # Note: s_row corresponds to vertical grid movement (uses d_v)
                #       s_col corresponds to horizontal grid movement (uses d_h)
                d = torch.sqrt(torch.tensor((s_row * d_v) ** 2 + (s_col * d_h) ** 2))
                D[i, j] = d
    
    return D

def get_immediate_neighbors(i: int) -> list[int]:
    """
    Return list of immediate neighbor indices for state i (8 neighbors on torus).
    This implements the LOCAL model: only immediate neighbors are reachable. Easier to conceptualize
    but because non sqrt(2) diagonals are not reachable, and therefore mixed banachian.
    """
    row_i, col_i = idx_to_coord(i)
    neighbors = []
    
    # 8 directions: up, down, left, right, and 4 diagonals
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue  # skip self
            # Wrap around using coord_to_idx
            neighbor_row = (row_i + dr) % GRID_SIZE
            neighbor_col = (col_i + dc) % GRID_SIZE
            neighbor_idx = coord_to_idx(neighbor_row, neighbor_col)
            neighbors.append(neighbor_idx)
    
    return neighbors

def build_p_true(D: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    GLOBAL MODEL: Return P_true[i, j] using exp(-D^2 / (2 * sigma^2)), zero on diagonal.
    
    This is a distance-weighted model where:
    - ALL 16 states are reachable from any state
    - Probabilities decay with distance (Gaussian)
    - Close states have high probability, far states have low (but non-zero) probability
    - More like a "diffusion process"
    
    Use this for testing if the model learns global geometry.
    """
    P = torch.zeros_like(D)
    
    for i in range(V):
        weights = torch.exp(-D[i, :] ** 2 / (2 * sigma ** 2))
        weights[i] = 0.0  # zero diagonal
        # Normalize to make row-stochastic
        total = weights.sum()
        if total > 0:
            P[i, :] = weights / total
        else:
            # Fallback: uniform over non-self
            P[i, :] = 1.0 / (V - 1)
            P[i, i] = 0.0
    
    return P

def build_p_true_local(D: torch.Tensor) -> torch.Tensor:
    """
    LOCAL MODEL: Return P_true[i, j] where only immediate 8 neighbors are reachable.
    
    This is a neighborhood-only model where:
    - Only the 8 immediate neighbors (on torus) can be reached from each state
    - All other states have probability 0
    - Probabilities are uniform over the 8 neighbors
    - More like a "local random walk"
    
    This matches the mental model: 4x4 grid where each cell has a 3x3 neighborhood
    (8 neighbors + self, with self = 0 probability).
    
    Use this for a simpler, more constrained Markov chain.
    """
    P = torch.zeros_like(D)
    
    for i in range(V):
        neighbors = get_immediate_neighbors(i)
        
        if len(neighbors) > 0:
            # Uniform probability over neighbors
            prob = 1.0 / len(neighbors)
            for neighbor_idx in neighbors:
                P[i, neighbor_idx] = prob
        
        # Diagonal is already 0 (no self-transitions)
        # All non-neighbors are already 0
    
    return P

def make_p_true() -> torch.Tensor:
    """
    Convenience function that builds and returns P_true.
    Uses MARKOV_MODEL_TYPE from config to choose between global and local models.
    
    Returns:
        - GLOBAL MODEL: All states reachable, probabilities decay with distance (Gaussian)
        - LOCAL MODEL: Only immediate 8 neighbors reachable, uniform probabilities
    """
    if MARKOV_MODEL_TYPE == "local":
        return make_p_true_local()
    else:  # default to global
        D = torus_distance_matrix()
        P_true = build_p_true(D, SIGMA)
        return P_true

def make_p_true_local() -> torch.Tensor:
    """
    LOCAL MODEL: Convenience function that builds and returns P_true using only immediate neighbors.
    Uses build_p_true_local() - only 8 neighbors reachable, uniform probabilities.
    """
    D = torus_distance_matrix()  # Still need D for shape, but won't use distance values
    P_true = build_p_true_local(D)
    return P_true
