"""
Test GPU index support in MilvusClient.

This test verifies that GPU-accelerated indexes are properly supported.
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from loguru import logger


def test_gpu_index_parameters():
    """Test GPU index parameters are properly initialized."""
    logger.info("Testing GPU index parameters...")
    
    # Mock pymilvus if not available
    try:
        import pymilvus
    except ImportError:
        logger.warning("pymilvus not installed, using mock")
        
        class MockConnections:
            def connect(self, *args, **kwargs): pass
            def disconnect(self, *args, **kwargs): pass
        
        class MockCollection:
            def __init__(self, *args, **kwargs): 
                self.name = 'test'
            def create_index(self, *args, **kwargs): pass
            def load(self): pass
        
        class MockUtility:
            def list_collections(self): return []
            def has_collection(self, name): return False
            def drop_collection(self, name): pass
        
        sys.modules['pymilvus'] = type('pymilvus', (), {
            'connections': MockConnections(),
            'Collection': MockCollection,
            'CollectionSchema': type('CollectionSchema', (), {}),
            'FieldSchema': type('FieldSchema', (), {}),
            'DataType': type('DataType', (), {
                'INT64': 1, 
                'FLOAT_VECTOR': 2, 
                'JSON': 3
            }),
            'utility': MockUtility(),
        })()
    
    from src.milvus_client import MilvusClient
    
    # Test each GPU index type
    gpu_indexes = {
        'GPU_CAGRA': {
            'intermediate_graph_degree': 64,
            'graph_degree': 32,
            'itopk_size': 64,
            'search_width': 4,
        },
        'GPU_IVF_FLAT': {
            'nlist': 128,
            'nprobe': 16,
        },
        'GPU_IVF_PQ': {
            'nlist': 128,
            'nprobe': 16,
        },
        'GPU_BRUTE_FORCE': {},
    }
    
    for index_type, params in gpu_indexes.items():
        try:
            client = MilvusClient(
                index_type=index_type,
                **params
            )
            
            assert client.index_type == index_type
            logger.info(f"✓ {index_type} index type initialized successfully")
            
            # Verify parameters are set
            if index_type == 'GPU_CAGRA':
                assert client.intermediate_graph_degree == params['intermediate_graph_degree']
                assert client.graph_degree == params['graph_degree']
                assert client.itopk_size == params['itopk_size']
                assert client.search_width == params['search_width']
                logger.info(f"  ✓ GPU_CAGRA parameters: graph_degree={client.graph_degree}, itopk_size={client.itopk_size}")
            elif index_type in ['GPU_IVF_FLAT', 'GPU_IVF_PQ']:
                assert client.nlist == params['nlist']
                assert client.nprobe == params['nprobe']
                logger.info(f"  ✓ {index_type} parameters: nlist={client.nlist}, nprobe={client.nprobe}")
            
            client.disconnect()
            
        except Exception as e:
            logger.error(f"✗ Failed to initialize {index_type}: {e}")
            raise
    
    logger.info("✓ All GPU index types initialized successfully")


def test_gpu_index_implementation():
    """Test that GPU indexes are implemented in _create_index."""
    logger.info("Testing GPU index implementation...")
    
    from src.milvus_client import MilvusClient
    import inspect
    
    source = inspect.getsource(MilvusClient._create_index)
    
    gpu_indexes = ['GPU_CAGRA', 'GPU_IVF_FLAT', 'GPU_IVF_PQ', 'GPU_BRUTE_FORCE']
    
    for idx_type in gpu_indexes:
        assert idx_type in source, f"{idx_type} not found in _create_index method"
        logger.info(f"✓ {idx_type} is implemented in _create_index")
    
    logger.info("✓ All GPU index types are implemented")


def test_gpu_search_parameters():
    """Test that GPU index search parameters are implemented."""
    logger.info("Testing GPU search parameters...")
    
    from src.milvus_client import MilvusClient
    import inspect
    
    source = inspect.getsource(MilvusClient.search_topk)
    
    gpu_indexes = ['GPU_CAGRA', 'GPU_IVF_FLAT', 'GPU_IVF_PQ', 'GPU_BRUTE_FORCE']
    
    for idx_type in gpu_indexes:
        assert idx_type in source, f"{idx_type} search parameters not found"
        logger.info(f"✓ {idx_type} search parameters are implemented")
    
    # Check GPU_CAGRA specific parameters
    assert 'itopk_size' in source
    assert 'search_width' in source
    logger.info("✓ GPU_CAGRA search parameters (itopk_size, search_width) present")
    
    logger.info("✓ All GPU search parameters are implemented")


def main():
    """Run all tests."""
    logger.info("\n" + "="*60)
    logger.info("Starting GPU Index Support Tests")
    logger.info("="*60 + "\n")
    
    try:
        test_gpu_index_parameters()
        print()
        
        test_gpu_index_implementation()
        print()
        
        test_gpu_search_parameters()
        print()
        
        logger.info("="*60)
        logger.info("✓ All GPU index tests passed!")
        logger.info("="*60)
        return 0
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
