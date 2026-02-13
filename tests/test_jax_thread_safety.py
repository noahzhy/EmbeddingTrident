"""
Test to verify JAX thread-safety fix in async pipeline.
"""

import sys
import os
import unittest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import threading

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import ServiceConfig


class TestJAXThreadSafety(unittest.TestCase):
    """Test that JAX preprocessing stays in main thread."""
    
    def test_preprocessing_runs_in_main_thread(self):
        """Verify that preprocessing happens in main thread, not producer thread."""
        
        main_thread_id = threading.current_thread().ident
        preprocess_thread_ids = []
        producer_thread_ids = []
        
        # Mock preprocessor that records which thread it's called from
        class MockPreprocessor:
            def preprocess_batch(self, inputs):
                preprocess_thread_ids.append(threading.current_thread().ident)
                # Return mock preprocessed data
                return np.random.rand(len(inputs), 224, 224, 3).astype(np.float32)
        
        # Mock Triton client
        class MockTritonClient:
            def infer(self, inputs, normalize=True):
                return np.random.rand(inputs.shape[0], 512).astype(np.float32)
        
        # Mock Milvus client
        class MockMilvusClient:
            def insert_embeddings(self, ids, embeddings, metadata=None, collection_name=None):
                return ids
        
        # Patch the imports
        with patch('src.pipeline.JAXImagePreprocessor', return_value=MockPreprocessor()), \
             patch('src.pipeline.TritonClient', return_value=MockTritonClient()), \
             patch('src.pipeline.MilvusClient', return_value=MockMilvusClient()):
            
            from src.pipeline import ImageEmbeddingPipeline
            
            config = ServiceConfig()
            config.async_pipeline.preprocess_workers = 1
            config.async_pipeline.embedding_workers = 1
            config.async_pipeline.insert_batch_size = 10
            
            # Create pipeline
            pipeline = ImageEmbeddingPipeline(config)
            
            # Run async insert
            test_inputs = ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg']
            test_ids = ['id1', 'id2', 'id3', 'id4']
            
            result = pipeline.insert_images_async(
                inputs=test_inputs,
                ids=test_ids,
                batch_size=2,
            )
            
            # Verify preprocessing happened in main thread
            self.assertTrue(len(preprocess_thread_ids) > 0, "Preprocessing should have been called")
            for thread_id in preprocess_thread_ids:
                self.assertEqual(
                    thread_id, 
                    main_thread_id,
                    f"Preprocessing should run in main thread (ID: {main_thread_id}), "
                    f"but ran in thread {thread_id}"
                )
            
            # Verify result
            self.assertEqual(result, test_ids)
            
            print("✓ Test passed: Preprocessing runs in main thread")
            print(f"  Main thread ID: {main_thread_id}")
            print(f"  Preprocessing calls: {len(preprocess_thread_ids)}")
            print(f"  All preprocessing in main thread: {all(tid == main_thread_id for tid in preprocess_thread_ids)}")


def run_tests():
    """Run all tests."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestJAXThreadSafety)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
