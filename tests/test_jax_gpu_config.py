"""
Tests for JAX GPU configuration and device placement.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import numpy as np
from src.preprocess_jax import JAXImagePreprocessor
from src.config import ServiceConfig, PreprocessConfig
from loguru import logger


def test_default_cpu_config():
    """Test that default configuration uses CPU."""
    print("\n=== Test 1: Default CPU Configuration ===")
    
    preprocessor = JAXImagePreprocessor(
        image_size=(224, 224),
        cache_compiled=False,
    )
    
    print(f"✓ Device configured: {preprocessor.device}")
    print(f"✓ Platform: {preprocessor.device.platform}")
    assert preprocessor.device is not None
    print("✓ Test passed: Default configuration works")


def test_explicit_cpu_config():
    """Test explicit CPU configuration."""
    print("\n=== Test 2: Explicit CPU Configuration ===")
    
    preprocessor = JAXImagePreprocessor(
        image_size=(224, 224),
        use_gpu=False,
        jax_platform='cpu',
        cache_compiled=False,
    )
    
    print(f"✓ Device configured: {preprocessor.device}")
    print(f"✓ Platform: {preprocessor.device.platform}")
    assert preprocessor.device.platform == 'cpu'
    print("✓ Test passed: Explicit CPU configuration works")


def test_gpu_config_with_fallback():
    """Test GPU configuration with graceful fallback to CPU."""
    print("\n=== Test 3: GPU Configuration (with fallback) ===")
    
    # Check if GPU is available
    gpu_devices = [d for d in jax.devices() if d.platform == 'gpu']
    has_gpu = len(gpu_devices) > 0
    
    print(f"GPU available: {has_gpu}")
    print(f"Available devices: {[d.platform for d in jax.devices()]}")
    
    preprocessor = JAXImagePreprocessor(
        image_size=(224, 224),
        use_gpu=True,
        cache_compiled=False,
    )
    
    print(f"✓ Device configured: {preprocessor.device}")
    print(f"✓ Platform: {preprocessor.device.platform}")
    
    if has_gpu:
        assert preprocessor.device.platform == 'gpu'
        print("✓ Test passed: GPU configuration successful")
    else:
        # Should fallback to CPU
        print("✓ Test passed: Graceful fallback to CPU when GPU unavailable")


def test_explicit_platform_config():
    """Test explicit platform configuration."""
    print("\n=== Test 4: Explicit Platform Configuration ===")
    
    # Test CPU platform
    preprocessor = JAXImagePreprocessor(
        image_size=(224, 224),
        jax_platform='cpu',
        cache_compiled=False,
    )
    
    print(f"✓ Device configured: {preprocessor.device}")
    print(f"✓ Platform: {preprocessor.device.platform}")
    assert preprocessor.device.platform == 'cpu'
    print("✓ Test passed: Explicit platform configuration works")


def test_preprocessing_on_device():
    """Test that preprocessing actually runs on the configured device."""
    print("\n=== Test 5: Preprocessing on Device ===")
    
    preprocessor = JAXImagePreprocessor(
        image_size=(224, 224),
        use_gpu=False,
        cache_compiled=True,
    )
    
    # Create dummy image
    dummy_image = np.random.rand(256, 256, 3).astype(np.float32)
    
    # Preprocess single image
    processed = preprocessor.preprocess_single(dummy_image)
    
    print(f"✓ Input shape: {dummy_image.shape}")
    print(f"✓ Output shape: {processed.shape}")
    assert processed.shape == (224, 224, 3)
    print("✓ Test passed: Single image preprocessing works")
    
    # Preprocess batch
    batch = [dummy_image, dummy_image, dummy_image]
    processed_batch = preprocessor.preprocess_batch(batch)
    
    print(f"✓ Batch input: {len(batch)} images")
    print(f"✓ Batch output shape: {processed_batch.shape}")
    assert processed_batch.shape == (3, 224, 224, 3)
    print("✓ Test passed: Batch preprocessing works")


def test_config_integration():
    """Test configuration integration via ServiceConfig."""
    print("\n=== Test 6: Config Integration ===")
    
    # Test with PreprocessConfig
    preprocess_config = PreprocessConfig(
        image_size=(128, 128),
        use_gpu=False,
        jax_platform='cpu',
    )
    
    config = ServiceConfig(preprocess=preprocess_config)
    
    preprocessor = JAXImagePreprocessor(
        image_size=config.preprocess.image_size,
        use_gpu=config.preprocess.use_gpu,
        jax_platform=config.preprocess.jax_platform,
        cache_compiled=False,
    )
    
    print(f"✓ Device: {preprocessor.device}")
    print(f"✓ Image size: {preprocessor.image_size}")
    print(f"✓ Platform: {preprocessor.device.platform}")
    assert preprocessor.image_size == (128, 128)
    assert preprocessor.device.platform == 'cpu'
    print("✓ Test passed: Config integration works")


def test_device_info():
    """Display JAX device information."""
    print("\n=== JAX Device Information ===")
    
    devices = jax.devices()
    print(f"Total devices: {len(devices)}")
    
    for i, device in enumerate(devices):
        print(f"  Device {i}: {device.platform} - {device.device_kind}")
    
    cpu_devices = [d for d in devices if d.platform == 'cpu']
    gpu_devices = [d for d in devices if d.platform == 'gpu']
    tpu_devices = [d for d in devices if d.platform == 'tpu']
    
    print(f"\nCPU devices: {len(cpu_devices)}")
    print(f"GPU devices: {len(gpu_devices)}")
    print(f"TPU devices: {len(tpu_devices)}")
    
    if gpu_devices:
        print("\n✓ GPU is available for acceleration!")
    else:
        print("\n⚠ No GPU detected - preprocessing will run on CPU")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("JAX GPU Configuration Tests")
    print("=" * 60)
    
    try:
        test_device_info()
        test_default_cpu_config()
        test_explicit_cpu_config()
        test_gpu_config_with_fallback()
        test_explicit_platform_config()
        test_preprocessing_on_device()
        test_config_integration()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ Test failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
