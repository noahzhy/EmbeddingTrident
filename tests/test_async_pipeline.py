"""
Test async pipeline functionality.
"""

import sys
import os
import unittest
import numpy as np
from unittest.mock import Mock, MagicMock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline import ImageEmbeddingPipeline
from src.config import ServiceConfig, AsyncPipelineConfig


class TestAsyncPipeline(unittest.TestCase):
    """Test async pipeline implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ServiceConfig()
        self.config.async_pipeline = AsyncPipelineConfig(
            preprocess_workers=2,
            embedding_workers=1,
            insert_batch_size=10,
            queue_maxsize=20,
        )
    
    def test_async_pipeline_config(self):
        """Test async pipeline configuration."""
        config = ServiceConfig()
        
        # Test default values
        self.assertEqual(config.async_pipeline.preprocess_workers, 2)
        self.assertEqual(config.async_pipeline.embedding_workers, 1)
        self.assertEqual(config.async_pipeline.insert_batch_size, 100)
        self.assertEqual(config.async_pipeline.queue_maxsize, 100)
        
        # Test custom values
        config.async_pipeline.preprocess_workers = 4
        config.async_pipeline.embedding_workers = 2
        self.assertEqual(config.async_pipeline.preprocess_workers, 4)
        self.assertEqual(config.async_pipeline.embedding_workers, 2)
    
    def test_async_pipeline_parameters(self):
        """Test async pipeline accepts correct parameters."""
        with patch('src.pipeline.JAXImagePreprocessor'), \
             patch('src.pipeline.TritonClient'), \
             patch('src.pipeline.MilvusClient'):
            
            pipeline = ImageEmbeddingPipeline(self.config)
            
            # Check that method exists
            self.assertTrue(hasattr(pipeline, 'insert_images_async'))
            self.assertTrue(callable(getattr(pipeline, 'insert_images_async')))
    
    def test_async_insert_validation(self):
        """Test async insert input validation."""
        with patch('src.pipeline.JAXImagePreprocessor'), \
             patch('src.pipeline.TritonClient'), \
             patch('src.pipeline.MilvusClient'):
            
            pipeline = ImageEmbeddingPipeline(self.config)
            
            # Test mismatched inputs and ids
            with self.assertRaises(ValueError):
                pipeline.insert_images_async(
                    inputs=['img1.jpg', 'img2.jpg'],
                    ids=['id1'],  # Mismatched length
                )
    
    @patch('src.pipeline.TritonClient')
    @patch('src.pipeline.MilvusClient')
    @patch('src.pipeline.JAXImagePreprocessor')
    def test_async_pipeline_flow(self, mock_preprocessor, mock_milvus, mock_triton):
        """Test async pipeline execution flow."""
        # Setup mocks
        mock_preprocessor_instance = MagicMock()
        mock_triton_instance = MagicMock()
        mock_milvus_instance = MagicMock()
        
        mock_preprocessor.return_value = mock_preprocessor_instance
        mock_triton.return_value = mock_triton_instance
        mock_milvus.return_value = mock_milvus_instance
        
        # Mock preprocessing output
        mock_preprocessor_instance.preprocess_batch.return_value = np.random.rand(2, 224, 224, 3).astype(np.float32)
        
        # Mock embedding output
        mock_triton_instance.infer.return_value = np.random.rand(2, 512).astype(np.float32)
        
        # Mock Milvus insert
        mock_milvus_instance.insert_embeddings.return_value = ['id1', 'id2']
        
        # Create pipeline
        pipeline = ImageEmbeddingPipeline(self.config)
        
        # Test async insert
        inputs = ['img1.jpg', 'img2.jpg']
        ids = ['id1', 'id2']
        
        result = pipeline.insert_images_async(
            inputs=inputs,
            ids=ids,
            batch_size=2,
        )
        
        # Verify results
        self.assertEqual(len(result), 2)
        self.assertEqual(result, ids)
        
        # Verify mocks were called
        mock_preprocessor_instance.preprocess_batch.assert_called()
        mock_triton_instance.infer.assert_called()
        mock_milvus_instance.insert_embeddings.assert_called()
    
    def test_config_yaml_with_async_pipeline(self):
        """Test configuration save/load with async pipeline settings."""
        config = ServiceConfig()
        config.async_pipeline.preprocess_workers = 4
        config.async_pipeline.embedding_workers = 2
        config.async_pipeline.insert_batch_size = 200
        config.async_pipeline.queue_maxsize = 150
        
        # Save to YAML
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.to_yaml(f.name)
            temp_file = f.name
        
        try:
            # Load from YAML
            loaded_config = ServiceConfig.from_yaml(temp_file)
            
            # Verify async pipeline settings are loaded
            self.assertEqual(loaded_config.async_pipeline.preprocess_workers, 4)
            self.assertEqual(loaded_config.async_pipeline.embedding_workers, 2)
            self.assertEqual(loaded_config.async_pipeline.insert_batch_size, 200)
            self.assertEqual(loaded_config.async_pipeline.queue_maxsize, 150)
        finally:
            # Cleanup
            if os.path.exists(temp_file):
                os.remove(temp_file)


def run_tests():
    """Run all tests."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAsyncPipeline)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
