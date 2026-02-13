# Custom Preprocessor Guide

This document explains how to create and use custom image preprocessors with the EmbeddingTrident pipeline.

## Overview

The EmbeddingTrident pipeline now supports custom preprocessors through the `ImagePreprocessor` protocol. This allows you to:

- Add custom preprocessing steps (e.g., augmentation, filtering)
- Use different preprocessing libraries (e.g., torchvision, OpenCV)
- Implement domain-specific preprocessing logic
- Maintain full compatibility with the pipeline

## Quick Start

### 1. Import the Interface

```python
from src import ImagePreprocessor, ImageEmbeddingPipeline, ServiceConfig
import numpy as np
from typing import Union, List
```

### 2. Create Your Custom Preprocessor

Your preprocessor must implement three methods:

```python
class MyCustomPreprocessor:
    """Custom preprocessor implementation."""
    
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
    
    def preprocess_single(
        self, 
        image: Union[np.ndarray, str]
    ) -> np.ndarray:
        """
        Preprocess a single image.
        
        Args:
            image: Image array (H, W, C) or path/URL
            
        Returns:
            Preprocessed image array
        """
        # Your preprocessing logic here
        ...
        return processed_image
    
    def preprocess_batch(
        self,
        images: List[Union[np.ndarray, str]]
    ) -> np.ndarray:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of image arrays or paths/URLs
            
        Returns:
            Batch of preprocessed images (B, H, W, C) or (B, C, H, W)
        """
        # Your batch preprocessing logic here
        ...
        return processed_batch
    
    def __call__(
        self,
        images: Union[str, np.ndarray, List[Union[str, np.ndarray]]]
    ) -> np.ndarray:
        """
        Unified interface for single or batch preprocessing.
        
        Args:
            images: Single image or list of images
            
        Returns:
            Preprocessed image(s) with batch dimension
        """
        if isinstance(images, list):
            return self.preprocess_batch(images)
        else:
            result = self.preprocess_single(images)
            return result[np.newaxis, ...]  # Add batch dimension
```

### 3. Use with Pipeline

```python
# Create your custom preprocessor
custom_preprocessor = MyCustomPreprocessor(image_size=(256, 256))

# Load configuration
config = ServiceConfig.from_yaml('configs/config.yaml')

# Create pipeline with custom preprocessor
pipeline = ImageEmbeddingPipeline(
    config=config,
    preprocessor=custom_preprocessor  # Pass your custom preprocessor
)

# Use the pipeline normally
embeddings = pipeline.embed_images(image_paths)
```

## Complete Example: Custom Augmentation Preprocessor

Here's a complete example showing how to add custom augmentation:

```python
import numpy as np
from PIL import Image
from typing import Union, List
from src import ImageEmbeddingPipeline, ServiceConfig

class AugmentedPreprocessor:
    """Preprocessor with random augmentation."""
    
    def __init__(
        self,
        image_size=(224, 224),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        horizontal_flip=True,
        random_crop=True,
    ):
        self.image_size = image_size
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        self.horizontal_flip = horizontal_flip
        self.random_crop = random_crop
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load image from path or URL."""
        if path.startswith(('http://', 'https://')):
            import requests
            import io
            response = requests.get(path, timeout=10)
            img = Image.open(io.BytesIO(response.content))
        else:
            img = Image.open(path)
        
        img = img.convert('RGB')
        return np.array(img, dtype=np.float32)
    
    def _apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply random augmentation."""
        # Random horizontal flip
        if self.horizontal_flip and np.random.rand() > 0.5:
            image = np.fliplr(image)
        
        # Random crop (if enabled)
        if self.random_crop:
            h, w = image.shape[:2]
            crop_h, crop_w = self.image_size
            
            # Only crop if image is larger than target
            if h > crop_h and w > crop_w:
                top = np.random.randint(0, h - crop_h)
                left = np.random.randint(0, w - crop_w)
                image = image[top:top+crop_h, left:left+crop_w]
        
        return image
    
    def _resize_and_normalize(self, image: np.ndarray) -> np.ndarray:
        """Resize and normalize image."""
        # Resize if needed
        if image.shape[:2] != self.image_size:
            pil_img = Image.fromarray(image.astype(np.uint8))
            pil_img = pil_img.resize(self.image_size, Image.BILINEAR)
            image = np.array(pil_img, dtype=np.float32)
        
        # Normalize
        image = image / 255.0
        image = (image - self.mean) / self.std
        
        return image
    
    def preprocess_single(self, image: Union[np.ndarray, str]) -> np.ndarray:
        """Preprocess single image with augmentation."""
        # Load if path
        if isinstance(image, str):
            image = self._load_image(image)
        
        # Apply augmentation
        image = self._apply_augmentation(image)
        
        # Resize and normalize
        image = self._resize_and_normalize(image)
        
        return image
    
    def preprocess_batch(self, images: List[Union[np.ndarray, str]]) -> np.ndarray:
        """Preprocess batch of images."""
        processed = [self.preprocess_single(img) for img in images]
        return np.stack(processed, axis=0)
    
    def __call__(
        self, 
        images: Union[str, np.ndarray, List[Union[str, np.ndarray]]]
    ) -> np.ndarray:
        """Unified interface."""
        if isinstance(images, list):
            return self.preprocess_batch(images)
        else:
            result = self.preprocess_single(images)
            return result[np.newaxis, ...]

# Usage
preprocessor = AugmentedPreprocessor(
    image_size=(256, 256),
    horizontal_flip=True,
    random_crop=True,
)

config = ServiceConfig.from_env()
pipeline = ImageEmbeddingPipeline(
    config=config,
    preprocessor=preprocessor
)

# Process images
embeddings = pipeline.embed_images(image_paths)
```

## Protocol Validation

Your preprocessor can be validated against the `ImagePreprocessor` protocol:

```python
from src import ImagePreprocessor

# Check if your class implements the protocol
assert isinstance(my_preprocessor, ImagePreprocessor)
```

## Output Format

Your preprocessor's output format should match your model's expected input:

- **NHWC format**: `(batch, height, width, channels)` - shape: `(B, H, W, C)`
- **NCHW format**: `(batch, channels, height, width)` - shape: `(B, C, H, W)`

Most PyTorch models expect NCHW, while TensorFlow models often use NHWC.

## Best Practices

1. **Handle both arrays and paths**: Support loading images from file paths and URLs
2. **Add batch dimension**: The `__call__` method should always return arrays with a batch dimension
3. **Use consistent dtypes**: Return `np.float32` arrays for better performance
4. **Document requirements**: Clearly document any additional dependencies your preprocessor needs
5. **Test thoroughly**: Test with single images, batches, and edge cases

## Debugging Tips

### Check protocol compliance
```python
from src import ImagePreprocessor
assert isinstance(my_preprocessor, ImagePreprocessor)
```

### Verify shapes
```python
# Single image should return (1, ...)
result = preprocessor(image)
assert result.ndim >= 3  # At least batch + 2 spatial dims

# Batch should return (N, ...)
batch_result = preprocessor([img1, img2, img3])
assert batch_result.shape[0] == 3
```

### Compare with default
```python
from src import JAXImagePreprocessor

# Compare outputs
jax_prep = JAXImagePreprocessor()
custom_prep = MyCustomPreprocessor()

jax_output = jax_prep(image)
custom_output = custom_prep(image)

print(f"JAX shape: {jax_output.shape}")
print(f"Custom shape: {custom_output.shape}")
```

## Examples

See the complete examples in:
- `examples/custom_preprocessor.py` - Full example with custom preprocessing
- `tests/test_custom_preprocessor.py` - Test cases showing protocol implementation

## Migration Guide

### Before (hardcoded preprocessor)
```python
pipeline = ImageEmbeddingPipeline(config)
# Always uses JAXImagePreprocessor
```

### After (custom preprocessor support)
```python
# Option 1: Use default (same as before)
pipeline = ImageEmbeddingPipeline(config)

# Option 2: Use custom preprocessor
custom_prep = MyCustomPreprocessor()
pipeline = ImageEmbeddingPipeline(config, preprocessor=custom_prep)
```

## FAQ

**Q: Do I need to implement all three methods?**  
A: Yes, `preprocess_single`, `preprocess_batch`, and `__call__` are all required for protocol compliance.

**Q: Can I use JAX, PyTorch, or TensorFlow in my custom preprocessor?**  
A: Yes! You can use any library as long as you return NumPy arrays.

**Q: What if my preprocessor is slow?**  
A: Consider:
- Using batch operations
- Multi-threading for I/O operations
- GPU acceleration (JAX, PyTorch, TensorFlow)
- Caching compiled functions

**Q: Can I access the pipeline's configuration in my preprocessor?**  
A: Yes, pass config parameters to your preprocessor's `__init__` method.

## Related Documentation

- [README.md](../README.md) - Main documentation
- [examples/basic_usage.py](../examples/basic_usage.py) - Basic pipeline usage
- [src/preprocess_jax.py](../src/preprocess_jax.py) - Reference JAX implementation
