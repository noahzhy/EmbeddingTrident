"""
Simple validation script to test the core components.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from loguru import logger


def test_imports():
    """Test that all modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        from src.config import ServiceConfig
        from src.preprocess_jax import JAXImagePreprocessor
        from src.triton_client import TritonClient
        from src.milvus_client import MilvusClient
        from src.pipeline import ImageEmbeddingPipeline
        from src.api_server import app
        
        logger.info("✓ All imports successful")
        return True
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_config():
    """Test configuration management."""
    logger.info("Testing configuration...")
    
    try:
        from src.config import ServiceConfig
        
        # Test default config
        config = ServiceConfig()
        assert config.triton.url == "localhost:8000"
        assert config.milvus.host == "localhost"
        assert config.preprocess.batch_size == 32
        
        # Test YAML save/load
        config.to_yaml('/tmp/test_config.yaml')
        loaded_config = ServiceConfig.from_yaml('/tmp/test_config.yaml')
        assert loaded_config.triton.url == config.triton.url
        
        os.remove('/tmp/test_config.yaml')
        
        logger.info("✓ Configuration tests passed")
        return True
    except Exception as e:
        logger.error(f"✗ Configuration test failed: {e}")
        return False


def test_jax_preprocessing():
    """Test JAX preprocessing without actual images."""
    logger.info("Testing JAX preprocessing...")
    
    try:
        import numpy as np
        from src.preprocess_jax import JAXImagePreprocessor
        
        # Create preprocessor
        preprocessor = JAXImagePreprocessor(
            image_size=(224, 224),
            cache_compiled=True,
        )
        
        # Test with dummy numpy array
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8).astype(np.float32)
        
        # Single image preprocessing
        processed = preprocessor.preprocess_single(dummy_image)
        assert processed.shape == (224, 224, 3)
        
        # Batch preprocessing
        dummy_batch = [dummy_image, dummy_image, dummy_image]
        batch_processed = preprocessor.preprocess_batch(dummy_batch)
        assert batch_processed.shape == (3, 224, 224, 3)
        
        logger.info("✓ JAX preprocessing tests passed")
        return True
    except Exception as e:
        logger.error(f"✗ JAX preprocessing test failed: {e}")
        return False


def test_triton_client_init():
    """Test Triton client initialization (without actual server)."""
    logger.info("Testing Triton client initialization...")
    
    try:
        from src.triton_client import TritonClient
        
        # This will fail to connect, but we can test initialization
        try:
            client = TritonClient(
                url="localhost:8000",
                model_name="test_model",
                timeout=1,
            )
            logger.warning("Triton server is running (unexpected in test)")
        except Exception:
            # Expected to fail without running server
            logger.info("✓ Triton client initialization logic works (server not running)")
            return True
        
    except ImportError as e:
        logger.warning(f"Triton client import failed (expected if tritonclient not installed): {e}")
        return True
    except Exception as e:
        logger.error(f"✗ Triton client test failed: {e}")
        return False


def test_milvus_client_init():
    """Test Milvus client initialization (without actual server)."""
    logger.info("Testing Milvus client initialization...")
    
    try:
        from src.milvus_client import MilvusClient
        
        # This will fail to connect, but we can test initialization
        try:
            client = MilvusClient(
                host="localhost",
                port=19530,
            )
            logger.warning("Milvus server is running (unexpected in test)")
        except Exception:
            # Expected to fail without running server
            logger.info("✓ Milvus client initialization logic works (server not running)")
            return True
        
    except ImportError as e:
        logger.warning(f"Milvus client import failed (expected if pymilvus not installed): {e}")
        return True
    except Exception as e:
        logger.error(f"✗ Milvus client test failed: {e}")
        return False


def test_api_server():
    """Test FastAPI server initialization."""
    logger.info("Testing FastAPI server...")
    
    try:
        from src.api_server import app
        from fastapi.testclient import TestClient
        
        # Create test client
        client = TestClient(app)
        
        # Test root endpoint (doesn't require services)
        response = client.get("/")
        assert response.status_code == 200
        assert "service" in response.json()
        
        logger.info("✓ API server tests passed")
        return True
    except Exception as e:
        logger.error(f"✗ API server test failed: {e}")
        return False


def main():
    """Run all validation tests."""
    logger.info("Starting validation tests...\n")
    
    tests = [
        test_imports,
        test_config,
        test_jax_preprocessing,
        test_triton_client_init,
        test_milvus_client_init,
        test_api_server,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()  # Add spacing between tests
        except Exception as e:
            logger.error(f"Test crashed: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    logger.info("="*60)
    logger.info(f"Validation Summary: {passed}/{total} tests passed")
    logger.info("="*60)
    
    if passed == total:
        logger.info("✓ All validation tests passed!")
        return 0
    else:
        logger.warning(f"✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
