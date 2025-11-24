# data.py
import torch
from torch.utils.data import Dataset, DataLoader
from config import SEQ_LEN, N_TRAIN, N_VAL, N_TEST, V

class MarkovSequenceDataset(Dataset):
    def __init__(self, P_true: torch.Tensor, num_seqs: int, seq_len: int, sequences: torch.Tensor = None):
        """
        P_true: [V, V] transition probability matrix (row-stochastic)
        num_seqs: number of sequences in dataset
        seq_len: length of each sequence
        sequences: Pre-generated sequences tensor [num_seqs, seq_len]. If None, will generate on-the-fly (slow).
        """
        self.P_true = P_true
        self.num_seqs = num_seqs
        self.seq_len = seq_len
        self.sequences = sequences  # Pre-generated sequences

    def __len__(self) -> int:
        return self.num_seqs

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (input_seq, target_seq) as LongTensors of shape [seq_len-1].
        """
        if self.sequences is not None:
            # Use pre-generated sequence
            sequence = self.sequences[idx]
        else:
            # Generate on-the-fly (slow, not recommended)
            x_0 = torch.randint(0, V, (1,), device=self.P_true.device).item()
            sequence = [x_0]
            current_state = x_0
            
            for t in range(self.seq_len - 1):
                probs = self.P_true[current_state]
                next_state = torch.multinomial(probs, 1).item()
                sequence.append(next_state)
                current_state = next_state
            
            sequence = torch.tensor(sequence, dtype=torch.long)
        
        # input_seq = x[0:seq_len-1], target_seq = x[1:seq_len]
        input_seq = sequence[:-1]
        target_seq = sequence[1:]
        
        return (input_seq, target_seq)

def generate_sequences(P_true: torch.Tensor, num_seqs: int, seq_len: int, seed: int = None):
    """
    Pre-generate all sequences using vectorized operations.
    This ensures fixed train/val/test splits for reproducibility.
    
    Args:
        P_true: [V, V] transition probability matrix
        num_seqs: number of sequences to generate
        seq_len: length of each sequence
        seed: random seed for reproducibility (optional)
    
    Returns:
        sequences: [num_seqs, seq_len] tensor of sequences
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    device = P_true.device
    sequences = torch.zeros(num_seqs, seq_len, dtype=torch.long, device=device)
    
    # Sample initial states for all sequences at once
    sequences[:, 0] = torch.randint(0, V, (num_seqs,), device=device)
    
    # Generate sequences in batches for efficiency
    batch_size = min(1000, num_seqs)  # Process in chunks
    for batch_start in range(0, num_seqs, batch_size):
        batch_end = min(batch_start + batch_size, num_seqs)
        
        current_states = sequences[batch_start:batch_end, 0].clone()
        
        for t in range(1, seq_len):
            # Vectorized sampling: sample next state for all sequences in batch
            # Get probabilities for current states
            probs = P_true[current_states]  # [batch_size, V]
            # Sample next states
            next_states = torch.multinomial(probs, 1).squeeze(-1)  # [batch_size]
            sequences[batch_start:batch_end, t] = next_states
            current_states = next_states
    
    return sequences

def make_dataloaders(P_true: torch.Tensor, batch_size: int, include_test: bool = False, 
                     train_sequences: torch.Tensor = None, val_sequences: torch.Tensor = None, 
                     test_sequences: torch.Tensor = None):
    """
    Return train_loader, val_loader, and optionally test_loader.
    
    Args:
        P_true: Transition probability matrix
        batch_size: Batch size for all loaders
        include_test: If True, also return test_loader (for final evaluation)
        train_sequences: Pre-generated train sequences [N_TRAIN, SEQ_LEN] (optional)
        val_sequences: Pre-generated val sequences [N_VAL, SEQ_LEN] (optional)
        test_sequences: Pre-generated test sequences [N_TEST, SEQ_LEN] (optional)
    
    Returns:
        train_loader, val_loader, (test_loader if include_test=True)
    """
    train_dataset = MarkovSequenceDataset(P_true, N_TRAIN, SEQ_LEN, sequences=train_sequences)
    val_dataset = MarkovSequenceDataset(P_true, N_VAL, SEQ_LEN, sequences=val_sequences)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead for small dataset
        pin_memory=False  # Not needed for small tensors
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    if include_test:
        test_dataset = MarkovSequenceDataset(P_true, N_TEST, SEQ_LEN, sequences=test_sequences)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader
