# Custom JAX Preprocessor Guide

This document explains how to create and use custom JAX-based image preprocessors with the EmbeddingTrident pipeline.

## Overview

The EmbeddingTrident pipeline requires all custom preprocessors to use JAX for high-performance preprocessing. This ensures:

- **High Performance**: JIT compilation and vectorization with vmap
- **GPU Acceleration**: Seamless GPU support through JAX
- **Type Safety**: Strong typing with JAX arrays
- **Consistency**: All preprocessors use the same high-performance framework

Custom preprocessors must inherit from `BaseJAXPreprocessor` and implement JAX-based preprocessing logic.

## Quick Start

### 1. Import Required Components

```python
from src import BaseJAXPreprocessor, ImageEmbeddingPipeline, ServiceConfig
import jax
import jax.numpy as jnp
```

### 2. Create Your Custom JAX Preprocessor

Your preprocessor must inherit from `BaseJAXPreprocessor` and implement the `_preprocess_single_jax()` method:

```python
class MyCustomPreprocessor(BaseJAXPreprocessor):
    """Custom JAX-based preprocessor."""
    
    def __init__(self, image_size=(224, 224), **kwargs):
        # Initialize base class
        super().__init__(image_size=image_size, **kwargs)
        # Your custom initialization
    
    def _preprocess_single_jax(self, image: jnp.ndarray) -> jnp.ndarray:
        """
        JAX-based preprocessing for a single image.
        
        This method is automatically JIT-compiled and vectorized with vmap.
        Use JAX operations for best performance.
        
        Args:
            image: Input image as JAX array (H, W, C) in [0, 255]
            
        Returns:
            Preprocessed image as JAX array
        """
        # Use JAX operations for preprocessing
        resized = jax.image.resize(
            image, 
            (*self.image_size, 3), 
            method='bilinear'
        )
        normalized = resized / 255.0
        return normalized
```

### 3. Use with Pipeline

```python
# Create your custom JAX preprocessor
custom_preprocessor = MyCustomPreprocessor(
    image_size=(256, 256),
    use_gpu=True  # Enable GPU acceleration
)

# Load configuration
config = ServiceConfig.from_yaml('configs/config.yaml')

# Create pipeline with custom preprocessor
pipeline = ImageEmbeddingPipeline(
    config=config,
    preprocessor=custom_preprocessor  # Must inherit from BaseJAXPreprocessor
)

# Use the pipeline normally
embeddings = pipeline.embed_images(image_paths)
```

## Complete Example: Custom JAX Preprocessor with Augmentation

Here's a complete example showing custom preprocessing with augmentation:

```python
from src import BaseJAXPreprocessor
import jax
import jax.numpy as jnp

class AugmentedJAXPreprocessor(BaseJAXPreprocessor):
    """JAX preprocessor with custom augmentation."""
    
    def __init__(
        self,
        image_size=(224, 224),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        contrast_adjustment=1.0,
        **kwargs
    ):
        super().__init__(image_size=image_size, **kwargs)
        self.mean = jnp.array(mean, dtype=jnp.float32).reshape(1, 1, 3)
        self.std = jnp.array(std, dtype=jnp.float32).reshape(1, 1, 3)
        self.contrast_adjustment = contrast_adjustment
    
    def _preprocess_single_jax(self, image: jnp.ndarray) -> jnp.ndarray:
        """JAX-based preprocessing with custom augmentation."""
        # 1. Resize using JAX
        resized = jax.image.resize(
            image,
            shape=(*self.image_size, image.shape[2]),
            method='bilinear'
        )
        
        # 2. Normalize to [0, 1]
        normalized = resized / 255.0
        
        # 3. Apply contrast adjustment (custom augmentation)
        if self.contrast_adjustment != 1.0:
            mean_val = jnp.mean(normalized)
            normalized = mean_val + self.contrast_adjustment * (normalized - mean_val)
            normalized = jnp.clip(normalized, 0.0, 1.0)
        
        # 4. Standard normalization
        normalized = (normalized - self.mean) / self.std
        
        return normalized

# Usage
preprocessor = AugmentedJAXPreprocessor(
    image_size=(256, 256),
    contrast_adjustment=1.2,  # 20% contrast increase
    use_gpu=True
)

# Test preprocessing
import numpy as np
image = np.random.rand(512, 512, 3).astype(np.float32) * 255
processed = preprocessor.preprocess_single(image)
print(f"Processed shape: {processed.shape}")
```

## What BaseJAXPreprocessor Provides

The `BaseJAXPreprocessor` base class provides:

### JAX Device Management
- Automatic device configuration (CPU/GPU/TPU)
- Device placement utilities
- Graceful fallback if requested device unavailable

### Image Loading
- `load_image(path)` - Load from local path or URL
- `load_image_from_path(path)` - Load from local file
- `load_image_from_url(url)` - Load from HTTP(S) URL
- `load_images_parallel(paths)` - Parallel loading with ThreadPoolExecutor

### Batch Processing
- `preprocess_batch(images)` - Automatic vmap vectorization
- `__call__(images)` - Unified interface for single or batch

### Methods to Implement

You only need to implement:

```python
def _preprocess_single_jax(self, image: jnp.ndarray) -> jnp.ndarray:
    """
    Your custom JAX preprocessing logic.
    
    This method will be:
    - JIT-compiled for fast execution
    - Vectorized with vmap for batch processing
    - Placed on configured device (CPU/GPU/TPU)
    """
    # Use JAX operations here
    ...
    return processed_image
```

## JAX Best Practices

### Use JAX Operations
Always use JAX operations instead of NumPy:

```python
# ✅ Good - JAX operations
resized = jax.image.resize(image, (224, 224, 3), method='bilinear')
normalized = image / 255.0
clipped = jnp.clip(normalized, 0, 1)

# ❌ Bad - NumPy operations (won't JIT compile properly)
resized = cv2.resize(image, (224, 224))
normalized = np.divide(image, 255.0)
```

### Avoid Conditionals on Values

JAX JIT compilation works best with static control flow:

```python
# ✅ Good - Configuration-based branching (known at compile time)
def __init__(self, apply_augmentation=True, **kwargs):
    super().__init__(**kwargs)
    self.apply_augmentation = apply_augmentation

def _preprocess_single_jax(self, image: jnp.ndarray) -> jnp.ndarray:
    if self.apply_augmentation:  # Static - known at init
        image = self._augment(image)
    return image / 255.0

# ❌ Bad - Value-based branching (dynamic)
def _preprocess_single_jax(self, image: jnp.ndarray) -> jnp.ndarray:
    if jnp.mean(image) > 128:  # Dynamic - depends on image content
        return image / 255.0
    return image / 128.0
```

For dynamic conditionals, use `jax.lax.cond`:

```python
def _preprocess_single_jax(self, image: jnp.ndarray) -> jnp.ndarray:
    mean_val = jnp.mean(image)
    # Use jax.lax.cond for value-dependent branching
    return jax.lax.cond(
        mean_val > 128,
        lambda x: x / 255.0,
        lambda x: x / 128.0,
        image
    )
```

### Use Pure Functions

JAX requires pure functions (no side effects):

```python
# ✅ Good - Pure function
def _preprocess_single_jax(self, image: jnp.ndarray) -> jnp.ndarray:
    resized = jax.image.resize(image, (*self.image_size, 3), method='bilinear')
    return resized / 255.0

# ❌ Bad - Side effects (logging, modifying external state)
def _preprocess_single_jax(self, image: jnp.ndarray) -> jnp.ndarray:
    logger.info("Processing image...")  # Side effect!
    self.processed_count += 1  # Side effect!
    return image / 255.0
```

## Advanced Features

### Custom Configuration

```python
class ConfigurablePreprocessor(BaseJAXPreprocessor):
    def __init__(
        self,
        image_size=(224, 224),
        normalize_method='standard',  # Custom config
        **kwargs
    ):
        super().__init__(image_size=image_size, **kwargs)
        self.normalize_method = normalize_method
        
        if normalize_method == 'standard':
            self.mean = jnp.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
            self.std = jnp.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        elif normalize_method == 'imagenet':
            self.mean = jnp.array([0.5, 0.5, 0.5]).reshape(1, 1, 3)
            self.std = jnp.array([0.5, 0.5, 0.5]).reshape(1, 1, 3)
    
    def _preprocess_single_jax(self, image: jnp.ndarray) -> jnp.ndarray:
        resized = jax.image.resize(image, (*self.image_size, 3), method='bilinear')
        normalized = resized / 255.0
        return (normalized - self.mean) / self.std
```

### GPU Acceleration

```python
# Enable GPU acceleration
preprocessor = MyCustomPreprocessor(
    image_size=(224, 224),
    use_gpu=True,  # Use GPU if available
    jax_platform='gpu'  # Explicitly specify GPU
)

# The base class handles device configuration automatically
# Your JAX operations will run on GPU
```

### Grayscale Conversion

```python
class GrayscalePreprocessor(BaseJAXPreprocessor):
    def _preprocess_single_jax(self, image: jnp.ndarray) -> jnp.ndarray:
        # Resize
        resized = jax.image.resize(image, (*self.image_size, 3), method='bilinear')
        
        # Convert to grayscale using JAX
        gray = 0.299 * resized[:, :, 0] + 0.587 * resized[:, :, 1] + 0.114 * resized[:, :, 2]
        
        # Replicate to 3 channels
        grayscale = jnp.stack([gray, gray, gray], axis=-1)
        
        return grayscale / 255.0
```

## Validation

### Type Checking

The pipeline validates that preprocessors inherit from `BaseJAXPreprocessor`:

```python
# ✅ Valid - Inherits from BaseJAXPreprocessor
class ValidPreprocessor(BaseJAXPreprocessor):
    def _preprocess_single_jax(self, image: jnp.ndarray) -> jnp.ndarray:
        return jax.image.resize(image, (*self.image_size, 3), method='bilinear') / 255.0

valid_prep = ValidPreprocessor()
pipeline = ImageEmbeddingPipeline(config=config, preprocessor=valid_prep)  # ✅ Works

# ❌ Invalid - Does NOT inherit from BaseJAXPreprocessor
class InvalidPreprocessor:
    def preprocess_single(self, image): return image / 255.0
    def preprocess_batch(self, images): return images / 255.0
    def __call__(self, images): return images / 255.0

invalid_prep = InvalidPreprocessor()
pipeline = ImageEmbeddingPipeline(config=config, preprocessor=invalid_prep)  # ❌ TypeError!
# TypeError: Custom preprocessor must inherit from BaseJAXPreprocessor
```

## Performance Tips

1. **Let JAX handle batching**: Don't implement manual batch processing, the base class uses vmap automatically
2. **Use JIT-friendly operations**: Stick to JAX operations for best performance
3. **Enable GPU when available**: Set `use_gpu=True` for 4-5x speedup
4. **Avoid Python loops**: Use JAX operations like `vmap`, `scan` instead
5. **Profile your code**: Use JAX profiling tools to identify bottlenecks

## Troubleshooting

### "Must inherit from BaseJAXPreprocessor"

```python
# Problem: Preprocessor doesn't inherit from BaseJAXPreprocessor
class MyPreprocessor:  # ❌ Missing inheritance
    pass

# Solution: Inherit from BaseJAXPreprocessor
class MyPreprocessor(BaseJAXPreprocessor):  # ✅ Correct
    def _preprocess_single_jax(self, image: jnp.ndarray) -> jnp.ndarray:
        return image / 255.0
```

### "Abstract method not implemented"

```python
# Problem: Forgot to implement _preprocess_single_jax
class MyPreprocessor(BaseJAXPreprocessor):
    pass  # ❌ Missing implementation

# Solution: Implement the required method
class MyPreprocessor(BaseJAXPreprocessor):
    def _preprocess_single_jax(self, image: jnp.ndarray) -> jnp.ndarray:  # ✅
        return jax.image.resize(image, (*self.image_size, 3), method='bilinear') / 255.0
```

### JAX Tracer Errors

```python
# Problem: Using non-JAX operations or side effects
def _preprocess_single_jax(self, image: jnp.ndarray) -> jnp.ndarray:
    print(f"Shape: {image.shape}")  # ❌ Side effect
    return image / 255.0

# Solution: Remove side effects, use JAX operations
def _preprocess_single_jax(self, image: jnp.ndarray) -> jnp.ndarray:
    # Use JAX operations only
    return jax.image.resize(image, (*self.image_size, 3), method='bilinear') / 255.0
```

## Examples

See complete examples in:
- `examples/custom_preprocessor.py` - Full example with custom JAX preprocessing
- `tests/test_custom_jax_preprocessor.py` - Test cases showing JAX implementation

## Migration from Protocol-Based to JAX-Based

If you have an old preprocessor that used the Protocol:

### Before (Protocol-based)
```python
class OldPreprocessor:
    def preprocess_single(self, image): ...
    def preprocess_batch(self, images): ...
    def __call__(self, images): ...
```

### After (JAX-based)
```python
class NewPreprocessor(BaseJAXPreprocessor):
    def _preprocess_single_jax(self, image: jnp.ndarray) -> jnp.ndarray:
        # Convert your logic to use JAX operations
        resized = jax.image.resize(image, (*self.image_size, 3), method='bilinear')
        return resized / 255.0
    
    # preprocess_batch, __call__ are provided by base class
```

## Why JAX?

JAX provides several benefits:

1. **Performance**: JIT compilation makes code as fast as hand-optimized C
2. **GPU Support**: Seamless acceleration without code changes
3. **Vectorization**: Automatic batch processing with vmap
4. **Gradient Support**: Can compute gradients if needed for advanced use cases
5. **NumPy Compatible**: Familiar API for NumPy users

## Related Documentation

- [README.md](../README.md) - Main documentation
- [examples/basic_usage.py](../examples/basic_usage.py) - Basic pipeline usage
- [src/base_preprocessor.py](../src/base_preprocessor.py) - Base class reference
- [src/preprocess_jax.py](../src/preprocess_jax.py) - Reference JAX implementation
