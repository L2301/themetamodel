# model.py
import torch
import torch.nn as nn
from config import V, D_MODEL, D_HEAD, D_FF, ROPE_BASE

def apply_rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """
    Apply Rotary Positional Encoding (RoPE) to embeddings.
    
    Args:
        x: Embeddings tensor [B, T, D_MODEL] or [T, D_MODEL]
        positions: Position indices [T] (0-indexed)
    
    Returns:
        Rotated embeddings with same shape as x
    """
    device = x.device
    dtype = x.dtype
    
    # Handle both [B, T, D] and [T, D] shapes
    if x.dim() == 2:
        x = x.unsqueeze(0)  # [1, T, D]
        squeeze_output = True
    else:
        squeeze_output = False
    
    B, T, D = x.shape
    
    # Ensure positions is on correct device and dtype
    if isinstance(positions, torch.Tensor):
        positions = positions.to(device).float()
    else:
        positions = torch.tensor(positions, dtype=torch.float32, device=device)
    
    # Reshape positions for broadcasting: [T] -> [1, T, 1]
    pos = positions.unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
    
    # Compute rotation angles for each dimension pair
    # For dimension pair (2i, 2i+1), theta = pos / (base^(2i/D))
    num_pairs = D // 2
    dim_indices = torch.arange(0, num_pairs, dtype=torch.float32, device=device)  # [0, 1, ..., num_pairs-1]
    
    # Compute frequencies: base^(2i/D) for each pair
    freqs = ROPE_BASE ** (2 * dim_indices / D)  # [num_pairs]
    freqs = freqs.unsqueeze(0).unsqueeze(0)  # [1, 1, num_pairs]
    
    # Compute angles: [1, T, num_pairs]
    angles = pos / freqs  # Broadcasting: [1, T, 1] / [1, 1, num_pairs] -> [1, T, num_pairs]
    
    # Compute cos and sin: [1, T, num_pairs]
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)
    
    # Reshape x to separate pairs: [B, T, num_pairs, 2]
    x_reshaped = x.view(B, T, num_pairs, 2)  # [B, T, num_pairs, 2]
    
    # Extract pairs: x_even = [B, T, num_pairs], x_odd = [B, T, num_pairs]
    x_even = x_reshaped[:, :, :, 0]  # [B, T, num_pairs]
    x_odd = x_reshaped[:, :, :, 1]   # [B, T, num_pairs]
    
    # Apply rotation to each pair
    # x_rotated[2i] = x[2i] * cos(theta) - x[2i+1] * sin(theta)
    # x_rotated[2i+1] = x[2i] * sin(theta) + x[2i+1] * cos(theta)
    x_even_rotated = x_even * cos_vals - x_odd * sin_vals  # [B, T, num_pairs]
    x_odd_rotated = x_even * sin_vals + x_odd * cos_vals   # [B, T, num_pairs]
    
    # Stack back: [B, T, num_pairs, 2]
    x_rotated = torch.stack([x_even_rotated, x_odd_rotated], dim=-1)
    
    # Reshape back to [B, T, D_MODEL]
    x_rotated = x_rotated.view(B, T, D)
    
    if squeeze_output:
        x_rotated = x_rotated.squeeze(0)  # [T, D]
    
    return x_rotated

class TinyTorusTransformer(nn.Module):
    def __init__(self, use_mlp=True, use_layernorm=True):
        """
        Args:
            use_mlp: If True, include MLP feed-forward network
            use_layernorm: If True, include LayerNorm after attention and MLP
        """
        super().__init__()
        self.token_emb = nn.Embedding(V, D_MODEL)

        self.W_Q = nn.Linear(D_MODEL, D_HEAD, bias=False)
        self.W_K = nn.Linear(D_MODEL, D_HEAD, bias=False)
        self.W_V = nn.Linear(D_MODEL, D_HEAD, bias=False)
        self.W_O = nn.Linear(D_HEAD, D_MODEL, bias=False)

        # Post-norm LayerNorm for attention
        if use_layernorm:
            self.ln1 = nn.LayerNorm(D_MODEL)
        else:
            self.ln1 = None
        
        # MLP (Feed-Forward Network)
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(D_MODEL, D_FF),
                nn.GELU(),  # GELU activation
                nn.Linear(D_FF, D_MODEL)
            )
        else:
            self.mlp = None
        
        # Post-norm LayerNorm for MLP
        if use_layernorm:
            self.ln2 = nn.LayerNorm(D_MODEL)
        else:
            self.ln2 = None

        # tied output: use token_emb.weight as W_out^T
        self.out_bias = nn.Parameter(torch.zeros(V))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: LongTensor [B, T]
        returns logits: [B, T, V]
        """
        B, T = x.shape
        
        # Embedding
        H0 = self.token_emb(x)  # [B, T, D_MODEL]
        
        # Apply RoPE to embeddings
        positions = torch.arange(T, device=x.device)  # [T]
        H0_rope = apply_rope(H0, positions)  # [B, T, D_MODEL]
        
        # Compute Q, K, V from RoPE-encoded embeddings
        Q = self.W_Q(H0_rope)  # [B, T, D_HEAD]
        K = self.W_K(H0_rope)  # [B, T, D_HEAD]
        V = self.W_V(H0_rope)  # [B, T, D_HEAD]
        
        # Attention scores
        sqrt_d_head = torch.sqrt(torch.tensor(D_HEAD, dtype=torch.float32, device=x.device))
        S = torch.matmul(Q, K.transpose(-1, -2)) / sqrt_d_head  # [B, T, T]
        
        # Apply causal mask (lower triangular)
        mask = torch.tril(torch.ones(T, T, device=x.device))  # [T, T]
        S = S.masked_fill(mask == 0, float('-inf'))  # [B, T, T]
        
        # Attention weights
        A = torch.softmax(S, dim=-1)  # [B, T, T]
        
        # Context
        H_attn = torch.matmul(A, V)  # [B, T, D_HEAD]
        
        # Project back
        H_attn_proj = self.W_O(H_attn)  # [B, T, D_MODEL]
        
        # Post-norm: residual connection + LayerNorm (attention block)
        if self.ln1 is not None:
            H1 = self.ln1(H0 + H_attn_proj)  # [B, T, D_MODEL]
        else:
            H1 = H0 + H_attn_proj  # [B, T, D_MODEL]
        
        # MLP (Feed-Forward Network)
        if self.mlp is not None:
            H_mlp = self.mlp(H1)  # [B, T, D_MODEL]
            # Post-norm: residual connection + LayerNorm (MLP block)
            if self.ln2 is not None:
                H2 = self.ln2(H1 + H_mlp)  # [B, T, D_MODEL]
            else:
                H2 = H1 + H_mlp  # [B, T, D_MODEL]
        else:
            H2 = H1  # Skip MLP
        
        # Logits using weight tying
        logits = torch.matmul(H2, self.token_emb.weight.T) + self.out_bias  # [B, T, V]
        
        return logits

    def token_qk_matrices(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (Q_tok, K_tok) using the embedding matrix only:
        Q_tok, K_tok: [V, D_HEAD]
        
        Note: RoPE is not applied here since token-level matrices represent
        static relationships without positional context.
        """
        E = self.token_emb.weight  # [V, D_MODEL]
        # Apply RoPE at position 0 for all tokens (or skip - token-level has no position)
        # For consistency with forward pass, we'll apply RoPE at position 0
        positions = torch.zeros(V, device=E.device)  # All tokens at position 0
        E_rope = apply_rope(E, positions)  # [V, D_MODEL]
        Q_tok = self.W_Q(E_rope)  # [V, D_HEAD]
        K_tok = self.W_K(E_rope)  # [V, D_HEAD]
        return (Q_tok, K_tok)
