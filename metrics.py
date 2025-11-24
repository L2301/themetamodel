# metrics.py
import torch
from config import V, GRID_SIZE, D_HEAD, R, r
from torus_markov import torus_distance_matrix

def p_model_from_qk(Q_tok: torch.Tensor, K_tok: torch.Tensor) -> torch.Tensor:
    """Return P_model[i, j] from token-level Q,K. Shape [V, V]."""
    # Compute attention scores
    sqrt_d_head = torch.sqrt(torch.tensor(D_HEAD, dtype=torch.float32, device=Q_tok.device))
    S = torch.matmul(Q_tok, K_tok.T) / sqrt_d_head  # [V, V]
    
    # Row-softmax to get transition probabilities
    P_model = torch.softmax(S, dim=-1)  # [V, V]
    
    return P_model

def kl_p_true_p_model(P_true: torch.Tensor, P_model: torch.Tensor) -> float:
    """
    Average KL(P_true[i] || P_model[i]) over i.
    
    KL(P||Q) = sum_j P[j] * log(P[j] / Q[j])
    
    Note: We use the original distributions (not normalized after adding epsilon)
    to preserve the true probabilities. Epsilon is only added inside the log to avoid log(0).
    """
    eps = 1e-10
    
    # Compute KL divergence row-wise: sum_j P_true[i,j] * log(P_true[i,j] / P_model[i,j])
    # Add epsilon inside log to avoid log(0), but use original distributions for weighting
    kl_per_row = torch.sum(
        P_true * torch.log((P_true + eps) / (P_model + eps)), 
        dim=-1
    )
    
    # Average over rows
    kl_avg = kl_per_row.mean().item()
    
    return kl_avg

def torus_pairwise_distances() -> torch.Tensor:
    """Recompute or load D_torus[i, j] used earlier."""
    return torus_distance_matrix()

def kspace_pairwise_distances(K_tok: torch.Tensor) -> torch.Tensor:
    """Compute D_K[i, j] from K_tok using vectorized operations."""
    # K_tok: [V, D_HEAD]
    # Vectorized computation: much faster than nested loops
    # D_K[i, j] = ||K_tok[i] - K_tok[j]||
    V_dim = K_tok.shape[0]
    device = K_tok.device
    
    # Expand K_tok to compute all pairwise differences
    # K_tok[i] - K_tok[j] for all i, j
    K_expanded_i = K_tok.unsqueeze(1)  # [V, 1, D_HEAD]
    K_expanded_j = K_tok.unsqueeze(0)  # [1, V, D_HEAD]
    
    # Compute all pairwise differences
    diff = K_expanded_i - K_expanded_j  # [V, V, D_HEAD]
    
    # Compute Euclidean norm for each pair
    D_K = torch.norm(diff, dim=2)  # [V, V]
    
    return D_K

def distance_correlation(D_torus: torch.Tensor, D_K: torch.Tensor) -> float:
    """Pearson correlation between flattened upper triangles."""
    # Ensure both tensors are on the same device
    device = D_torus.device
    D_K = D_K.to(device)
    
    # Get upper triangle indices (i < j)
    upper_triangle_mask = torch.triu(torch.ones(V, V, dtype=torch.bool, device=device), diagonal=1)
    
    # Flatten upper triangles
    d_torus_flat = D_torus[upper_triangle_mask]
    d_K_flat = D_K[upper_triangle_mask]
    
    # Compute Pearson correlation manually
    if d_torus_flat.std().item() == 0 or d_K_flat.std().item() == 0:
        return 0.0
    
    # Center the data
    d_torus_centered = d_torus_flat - d_torus_flat.mean()
    d_K_centered = d_K_flat - d_K_flat.mean()
    
    # Compute correlation coefficient
    numerator = (d_torus_centered * d_K_centered).sum()
    denominator = torch.sqrt((d_torus_centered ** 2).sum() * (d_K_centered ** 2).sum())
    
    if denominator.item() == 0:
        return 0.0
    
    correlation = (numerator / denominator).item()
    return correlation

def neighborhood_overlap(D_torus: torch.Tensor, D_K: torch.Tensor, k: int) -> float:
    """Average overlap of k-NN sets under the two metrics."""
    # Ensure both tensors are on the same device
    device = D_torus.device
    D_K = D_K.to(device)
    
    overlaps = []
    
    for i in range(V):
        # Get k-NN under torus distance (exclude self)
        dists_torus = D_torus[i].clone()
        dists_torus[i] = float('inf')  # exclude self
        _, nn_torus = torch.topk(dists_torus, k, largest=False)
        nn_torus_set = set(nn_torus.cpu().tolist())
        
        # Get k-NN under K-space distance (exclude self)
        dists_K = D_K[i].clone()
        dists_K[i] = float('inf')  # exclude self
        _, nn_K = torch.topk(dists_K, k, largest=False)
        nn_K_set = set(nn_K.cpu().tolist())
        
        # Compute overlap
        overlap = len(nn_torus_set & nn_K_set) / k
        overlaps.append(overlap)
    
    return sum(overlaps) / len(overlaps)

def learned_radii(K_tok: torch.Tensor) -> tuple[float, float]:
    """Compute R_hat, r_hat from K-space distances."""
    D_K = kspace_pairwise_distances(K_tok)
    
    # Horizontal loop: [4, 5, 6, 7, 4]
    horizontal_loop = [4, 5, 6, 7, 4]
    R_hat = 0.0
    for idx in range(len(horizontal_loop) - 1):
        i, j = horizontal_loop[idx], horizontal_loop[idx + 1]
        R_hat += D_K[i, j].item()
    
    # Vertical loop: [1, 5, 9, 13, 1]
    vertical_loop = [1, 5, 9, 13, 1]
    r_hat = 0.0
    for idx in range(len(vertical_loop) - 1):
        i, j = vertical_loop[idx], vertical_loop[idx + 1]
        r_hat += D_K[i, j].item()
    
    return (R_hat, r_hat)
