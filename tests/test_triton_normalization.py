"""
Test to verify NumPy-based L2 normalization works correctly and is thread-safe.
"""

import sys
import os
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_l2_normalization():
    """Test that L2 normalization produces correct results."""
    from src.triton_client import TritonClient
    
    # Create a mock client (we only need the normalization method)
    client = TritonClient(url="localhost:8000")
    
    # Test case 1: Simple vectors
    embeddings = np.array([
        [3.0, 4.0],      # Should normalize to [0.6, 0.8]
        [1.0, 0.0],      # Should normalize to [1.0, 0.0]
        [0.0, 0.0],      # Should stay [0.0, 0.0] (zero vector case)
    ])
    
    normalized = client.l2_normalize(embeddings)
    
    # Check first vector: [3, 4] -> [0.6, 0.8] (norm = 5)
    expected_1 = np.array([0.6, 0.8])
    assert np.allclose(normalized[0], expected_1, rtol=1e-5), f"Expected {expected_1}, got {normalized[0]}"
    
    # Check second vector: [1, 0] -> [1, 0] (norm = 1)
    expected_2 = np.array([1.0, 0.0])
    assert np.allclose(normalized[1], expected_2, rtol=1e-5), f"Expected {expected_2}, got {normalized[1]}"
    
    # Check third vector: [0, 0] -> [0, 0] (zero vector)
    expected_3 = np.array([0.0, 0.0])
    assert np.allclose(normalized[2], expected_3, rtol=1e-5), f"Expected {expected_3}, got {normalized[2]}"
    
    # Verify all vectors have unit norm (except zero vector)
    norms = np.linalg.norm(normalized, axis=1)
    assert np.allclose(norms[0], 1.0, rtol=1e-5), f"Norm should be 1.0, got {norms[0]}"
    assert np.allclose(norms[1], 1.0, rtol=1e-5), f"Norm should be 1.0, got {norms[1]}"
    # Zero vector stays zero
    assert np.allclose(norms[2], 0.0, rtol=1e-5), f"Zero vector norm should be 0.0, got {norms[2]}"
    
    print("✓ L2 normalization produces correct results")


def test_thread_safety():
    """Test that normalization can be called from multiple threads."""
    from src.triton_client import TritonClient
    
    client = TritonClient(url="localhost:8000")
    
    def normalize_in_thread(thread_id):
        """Normalize embeddings in a worker thread."""
        # Create random embeddings
        embeddings = np.random.randn(10, 128).astype(np.float32)
        
        # Normalize
        normalized = client.l2_normalize(embeddings)
        
        # Verify norms are close to 1.0
        norms = np.linalg.norm(normalized, axis=1)
        assert np.allclose(norms, 1.0, rtol=1e-5), f"Thread {thread_id}: norms not close to 1.0"
        
        return thread_id
    
    # Run normalization in multiple threads
    num_threads = 5
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(normalize_in_thread, i) for i in range(num_threads)]
        results = [f.result() for f in futures]
    
    assert len(results) == num_threads, f"Expected {num_threads} results, got {len(results)}"
    print(f"✓ L2 normalization is thread-safe (tested with {num_threads} threads)")


def test_batch_normalization():
    """Test normalization with different batch sizes."""
    from src.triton_client import TritonClient
    
    client = TritonClient(url="localhost:8000")
    
    # Test different batch sizes
    for batch_size in [1, 10, 100]:
        embeddings = np.random.randn(batch_size, 512).astype(np.float32)
        normalized = client.l2_normalize(embeddings)
        
        # Check shape is preserved
        assert normalized.shape == embeddings.shape, f"Shape mismatch: {normalized.shape} vs {embeddings.shape}"
        
        # Check all vectors have unit norm
        norms = np.linalg.norm(normalized, axis=1)
        assert np.allclose(norms, 1.0, rtol=1e-5), f"Batch size {batch_size}: norms not close to 1.0"
    
    print(f"✓ L2 normalization works with different batch sizes")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing NumPy-based L2 Normalization")
    print("=" * 60)
    
    try:
        test_l2_normalization()
        test_thread_safety()
        test_batch_normalization()
        
        print("=" * 60)
        print("✓✓✓ All tests passed! ✓✓✓")
        print("=" * 60)
        print("\nNumPy-based L2 normalization is:")
        print("  ✓ Mathematically correct")
        print("  ✓ Thread-safe (no JAX thread errors)")
        print("  ✓ Works with different batch sizes")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
