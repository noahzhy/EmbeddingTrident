#!/usr/bin/env python3
"""
Simple performance benchmark for streaming vs sequential preprocessing.
"""

import sys
import os
import time
import tempfile
import shutil
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.streaming_preprocessor import StreamingMultiprocessPreprocessor
from src.preprocess_jax import JAXImagePreprocessor


def create_test_images(num_images=32, size=(512, 512)):
    """Create temporary test images."""
    temp_dir = tempfile.mkdtemp()
    image_paths = []
    
    for i in range(num_images):
        img_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        path = os.path.join(temp_dir, f"test_image_{i:03d}.jpg")
        img.save(path)
        image_paths.append(path)
    
    return temp_dir, image_paths


def main():
    print("\n" + "="*60)
    print("Performance Benchmark: Streaming vs Sequential")
    print("="*60)
    
    temp_dir = None
    try:
        # Create test images
        num_images = 32
        print(f"\nCreating {num_images} test images...")
        temp_dir, image_paths = create_test_images(num_images=num_images)
        print(f"✓ Created {num_images} test images")
        
        # Test sequential preprocessing
        print("\n--- Sequential Preprocessing ---")
        sequential_preprocessor = JAXImagePreprocessor(
            image_size=(224, 224),
            data_format='NCHW',
            cache_compiled=True,
            max_workers=4,
        )
        
        start_time = time.time()
        sequential_result = sequential_preprocessor.preprocess_batch(image_paths)
        sequential_time = time.time() - start_time
        
        print(f"Time: {sequential_time:.3f}s")
        print(f"Throughput: {num_images/sequential_time:.1f} images/sec")
        print(f"Shape: {sequential_result.shape}")
        
        # Test streaming multiprocessing preprocessing
        print("\n--- Streaming Multiprocessing Preprocessing ---")
        streaming_preprocessor = StreamingMultiprocessPreprocessor(
            num_workers=4,
            batch_size=8,
            image_size=(224, 224),
            data_format='NCHW',
            cache_compiled=True,
        )
        
        start_time = time.time()
        with streaming_preprocessor:
            streaming_result = streaming_preprocessor.preprocess_batch_sync(image_paths, batch_size=8)
        streaming_time = time.time() - start_time
        
        print(f"Time: {streaming_time:.3f}s")
        print(f"Throughput: {num_images/streaming_time:.1f} images/sec")
        print(f"Shape: {streaming_result.shape}")
        
        # Calculate speedup
        speedup = sequential_time / streaming_time
        print("\n" + "="*60)
        print(f"🚀 Speedup: {speedup:.2f}x")
        print(f"⚡ Performance gain: {(speedup-1)*100:.1f}%")
        print("="*60)
        
        # Validate shapes match
        if sequential_result.shape == streaming_result.shape:
            print("✓ Shapes match")
        else:
            print(f"✗ Shape mismatch: {sequential_result.shape} vs {streaming_result.shape}")
            
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
