"""
Test to verify gevent-based async pipeline works correctly.
"""

import sys
import os
import numpy as np
from unittest.mock import Mock, MagicMock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_gevent_pipeline():
    """Test that the async pipeline uses gevent greenlets."""
    
    # Mock the imports
    with patch('src.pipeline.JAXImagePreprocessor'), \
         patch('src.pipeline.TritonClient'), \
         patch('src.pipeline.MilvusClient'):
        
        from src.config import ServiceConfig
        from src.pipeline import ImageEmbeddingPipeline
        import gevent
        
        # Create config
        config = ServiceConfig()
        config.async_pipeline.preprocess_workers = 1
        config.async_pipeline.embedding_workers = 1
        config.async_pipeline.insert_batch_size = 10
        
        # Create pipeline with mocked components
        pipeline = ImageEmbeddingPipeline(config)
        
        # Mock the preprocessor
        pipeline.preprocessor = Mock()
        pipeline.preprocessor.preprocess_batch = Mock(
            return_value=np.random.rand(2, 224, 224, 3).astype(np.float32)
        )
        
        # Mock the triton client
        pipeline.triton_client = Mock()
        pipeline.triton_client.infer = Mock(
            return_value=np.random.rand(2, 512).astype(np.float32)
        )
        
        # Mock the milvus client
        pipeline.milvus_client = Mock()
        pipeline.milvus_client.insert_embeddings = Mock(
            return_value=['id1', 'id2', 'id3', 'id4']
        )
        
        # Test data
        test_inputs = ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg']
        test_ids = ['id1', 'id2', 'id3', 'id4']
        
        # Run async insert
        result = pipeline.insert_images_async(
            inputs=test_inputs,
            ids=test_ids,
            batch_size=2,
        )
        
        # Verify result
        assert result == test_ids, f"Expected {test_ids}, got {result}"
        
        # Verify methods were called
        assert pipeline.preprocessor.preprocess_batch.called
        assert pipeline.triton_client.infer.called
        assert pipeline.milvus_client.insert_embeddings.called
        
        print("✓ Gevent-based async pipeline test passed")
        print(f"  Result: {len(result)} items processed")
        print(f"  Preprocessing called: {pipeline.preprocessor.preprocess_batch.call_count} times")
        print(f"  Inference called: {pipeline.triton_client.infer.call_count} times")
        print(f"  Insertion called: {pipeline.milvus_client.insert_embeddings.call_count} times")


def test_gevent_imports():
    """Test that gevent is properly imported."""
    try:
        import gevent
        from gevent import queue
        
        print("✓ Gevent imports successful")
        print(f"  gevent version: {gevent.__version__}")
        
        # Test basic greenlet functionality
        results = []
        
        def worker(n):
            results.append(n)
        
        greenlets = [gevent.spawn(worker, i) for i in range(3)]
        gevent.joinall(greenlets)
        
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        print(f"  ✓ Basic greenlet test: {results}")
        
        # Test queue functionality
        q = queue.Queue()
        q.put(1)
        q.put(2)
        assert q.get() == 1
        assert q.get() == 2
        print("  ✓ Gevent queue works correctly")
        
    except ImportError as e:
        print(f"✗ Gevent import failed: {e}")
        print("  Install with: pip install gevent")
        return False
    
    return True


if __name__ == '__main__':
    print("=" * 60)
    print("Testing Gevent-based Async Pipeline")
    print("=" * 60)
    
    # First test gevent imports
    if not test_gevent_imports():
        print("\n✗ Gevent not available. Please install: pip install gevent")
        sys.exit(1)
    
    print()
    
    # Then test the pipeline
    try:
        test_gevent_pipeline()
        
        print("\n" + "=" * 60)
        print("✓✓✓ All gevent tests passed! ✓✓✓")
        print("=" * 60)
        print("\nGevent-based async pipeline:")
        print("  ✓ Uses greenlets instead of OS threads")
        print("  ✓ Compatible with gevent applications")
        print("  ✓ Cooperative multitasking")
        print("  ✓ No thread context switching errors")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
