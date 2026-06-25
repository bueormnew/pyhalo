import torch
import pytest
from halo.attention.graph import generate_neighbor_lists

def test_generate_neighbor_lists_shape():
    seq_len = 100
    local_window = 64
    num_globals = 2
    dilated_offsets = [1, 2, 4, 8]
    num_random = 2
    
    neighbors = generate_neighbor_lists(
        seq_len=seq_len,
        local_window=local_window,
        num_globals=num_globals,
        dilated_offsets=dilated_offsets,
        num_random=num_random
    )
    
    expected_neighbors = num_globals + local_window + 2 * len(dilated_offsets) + num_random
    assert neighbors.shape == (seq_len, expected_neighbors), f"Expected shape {(seq_len, expected_neighbors)}, got {neighbors.shape}"
    
def test_out_of_bounds_clamping():
    seq_len = 10
    neighbors = generate_neighbor_lists(
        seq_len=seq_len,
        local_window=20, # Mayor que la secuencia
        num_globals=2,
        dilated_offsets=[],
        num_random=0
    )
    
    # Asegurar que ningún índice sea menor que 0 o mayor o igual a seq_len
    assert torch.all(neighbors >= 0)
    assert torch.all(neighbors < seq_len)

def test_deterministic_random():
    seq_len = 100
    
    n1 = generate_neighbor_lists(seq_len, layer_id=0)
    n2 = generate_neighbor_lists(seq_len, layer_id=0)
    
    # Deben ser idénticos si se usa el mismo layer_id
    assert torch.equal(n1, n2)
    
    n3 = generate_neighbor_lists(seq_len, layer_id=1)
    
    # Deben ser diferentes para distintas capas
    assert not torch.equal(n1, n3)
