from typing import List
import os
import torch
import numpy as np
import torch
import random


# === Helper Functions ===

def get_top_k_moves(policy_as_dict, k=5):
    """Return the top k (move_uci, probability) pairs sorted by descending probability."""
    sorted_pairs = sorted(policy_as_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_pairs[:k]


def list_pgn_files(directory: str) -> List[str]:
    """Recursively lists all PGN files in a directory."""
    pgn_paths: List[str] = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".pgn"):
                pgn_paths.append(os.path.join(root, file))
    return pgn_paths


def set_device():
    """
    Sets the available device: CUDA, MPS, or CPU.

    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA is available. Using GPU.")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("MPS is available. Using Apple Silicon GPU.")
    else:
        device = torch.device('cpu')
        print("No GPU detected. Using CPU.")
    return device


def set_all_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_determinism(seed=42):
    # Set all random seeds
    set_all_seeds(seed=seed)

    # Force deterministic PyTorch behavior
    torch.use_deterministic_algorithms(True)

    # Disable CuDNN benchmarking which selects non-deterministic algorithms
    torch.backends.cudnn.benchmark = False

    # Force deterministic CuDNN operations
    torch.backends.cudnn.deterministic = True
