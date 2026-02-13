#!/usr/bin/env python3
"""
Example: Using Streaming Multiprocessing Preprocessing

This example demonstrates how to use the new streaming multiprocessing 
preprocessing feature for high-throughput image preprocessing.
"""

import sys
import os
import tempfile
import shutil
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.streaming_preprocessor import StreamingMultiprocessPreprocessor


def create_sample_images(num_images=100):
    """Create sample images for demonstration."""
    temp_dir = tempfile.mkdtemp()
    image_paths = []
    
    print(f"Creating {num_images} sample images...")
    for i in range(num_images):
        # Create random image
        img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Save to temp directory
        path = os.path.join(temp_dir, f"sample_image_{i:04d}.jpg")
        img.save(path)
        image_paths.append(path)
    
    print(f"✓ Created {num_images} sample images in {temp_dir}")
    return temp_dir, image_paths


def example_basic_streaming():
    """
    Example 1: Basic Streaming Preprocessing
    
    This example shows the simplest way to use streaming preprocessing.
    """
    print("\n" + "="*60)
    print("Example 1: Basic Streaming Preprocessing")
    print("="*60 + "\n")
    
    temp_dir = None
    try:
        # Create sample images
        temp_dir, image_paths = create_sample_images(num_images=50)
        
        # Create streaming preprocessor
        print("Creating streaming preprocessor...")
        preprocessor = StreamingMultiprocessPreprocessor(
            num_workers=4,
            batch_size=10,
            image_size=(224, 224),
            data_format='NCHW',
        )
        
        # Process images
        print("\nProcessing images with streaming...")
        with preprocessor:
            batch_count = 0
            for batch_result in preprocessor.preprocess_stream(image_paths, batch_size=10):
                batch_count += 1
                preprocessed = batch_result['preprocessed']
                print(f"  Batch {batch_count}: shape={preprocessed.shape}")
        
        print(f"\n✓ Processed {len(image_paths)} images in {batch_count} batches")
        
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir)


def example_with_metadata():
    """
    Example 2: Streaming with IDs and Metadata
    
    This example shows how to preserve IDs and metadata through the streaming pipeline.
    """
    print("\n" + "="*60)
    print("Example 2: Streaming with IDs and Metadata")
    print("="*60 + "\n")
    
    temp_dir = None
    try:
        # Create sample images
        temp_dir, image_paths = create_sample_images(num_images=30)
        
        # Create IDs and metadata for each image
        ids = [f"img_{i:04d}" for i in range(len(image_paths))]
        metadata = [
            {"index": i, "category": f"cat_{i % 3}", "score": i * 0.1}
            for i in range(len(image_paths))
        ]
        
        print("Processing images with IDs and metadata...")
        
        # Create preprocessor
        preprocessor = StreamingMultiprocessPreprocessor(
            num_workers=3,
            batch_size=10,
            image_size=(224, 224),
        )
        
        # Process with metadata
        with preprocessor:
            for batch_result in preprocessor.preprocess_stream(
                image_paths,
                ids=ids,
                metadata=metadata,
                batch_size=10
            ):
                batch_ids = batch_result['ids']
                batch_metadata = batch_result['metadata']
                preprocessed = batch_result['preprocessed']
                
                print(f"\n  Batch {batch_result['batch_idx']}:")
                print(f"    IDs: {batch_ids[:3]}...")
                print(f"    Metadata samples: {batch_metadata[:2]}")
                print(f"    Shape: {preprocessed.shape}")
        
        print(f"\n✓ Processed {len(image_paths)} images with preserved IDs and metadata")
        
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir)


def example_sync_processing():
    """
    Example 3: Synchronous Batch Processing
    
    This example shows how to use streaming preprocessing in a synchronous manner,
    blocking until all results are ready.
    """
    print("\n" + "="*60)
    print("Example 3: Synchronous Batch Processing")
    print("="*60 + "\n")
    
    temp_dir = None
    try:
        # Create sample images
        temp_dir, image_paths = create_sample_images(num_images=40)
        
        # Create preprocessor
        preprocessor = StreamingMultiprocessPreprocessor(
            num_workers=4,
            batch_size=10,
            image_size=(384, 384),
            data_format='NHWC',  # Note: different format
        )
        
        # Process synchronously (blocks until complete)
        print("Processing images synchronously...")
        with preprocessor:
            all_preprocessed = preprocessor.preprocess_batch_sync(
                image_paths,
                batch_size=10
            )
        
        print(f"\n✓ Processed all {len(image_paths)} images")
        print(f"  Result shape: {all_preprocessed.shape}")
        print(f"  Data format: NHWC (batch, height, width, channels)")
        
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir)


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("Streaming Multiprocessing Preprocessing Examples")
    print("="*60)
    
    try:
        # Run examples
        example_basic_streaming()
        example_with_metadata()
        example_sync_processing()
        
        print("\n" + "="*60)
        print("✓ All examples completed successfully!")
        print("="*60 + "\n")
        
        # Print usage tips
        print("Usage Tips:")
        print("  • Best for large batches (>1000 images)")
        print("  • Use num_workers = CPU count for best performance")
        print("  • Larger batch_size utilizes JAX vmap better")
        print("  • Consider startup overhead (~1-2s) for small batches")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
