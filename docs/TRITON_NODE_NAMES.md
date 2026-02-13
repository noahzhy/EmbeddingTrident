# Feature: Triton Server Input/Output Node Name Specification

## Overview

This feature allows users to specify custom input and output tensor names for Triton Inference Server, providing flexibility for different model architectures.

## Problem Statement (问题描述)

Previously, the Triton client used hardcoded default values ("input" and "output") for tensor names. This limitation prevented users from working with models that use different tensor naming conventions.

## Solution (解决方案)

Added configurable input/output node names at multiple levels:
1. Configuration file (YAML)
2. Environment variables
3. Direct client initialization
4. Per-call override

## Implementation Details

### 1. Configuration Changes

**TritonConfig (src/config.py)**
```python
@dataclass
class TritonConfig:
    url: str = "localhost:8000"
    model_name: str = "embedding_model"
    model_version: str = "1"
    protocol: str = "http"
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    input_name: str = "input"      # NEW
    output_name: str = "output"    # NEW
```

### 2. Client Changes

**TritonClient (src/triton_client.py)**
```python
def __init__(
    self,
    url: str = "localhost:8000",
    model_name: str = "embedding_model",
    model_version: str = "1",
    protocol: str = "http",
    timeout: int = 60,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    input_name: str = "input",     # NEW
    output_name: str = "output",   # NEW
):
    ...
    self.input_name = input_name
    self.output_name = output_name
```

**Inference Methods Updated**
- `infer()`: Uses instance defaults, allows per-call override
- `infer_batch()`: Uses instance defaults, allows per-call override
- `async_infer()`: Uses instance defaults, allows per-call override

### 3. Configuration File Support

**config.yaml**
```yaml
triton:
  url: "localhost:8000"
  model_name: "embedding_model"
  input_name: "input"      # Can customize
  output_name: "output"    # Can customize
```

**Environment Variables**
```bash
TRITON_INPUT_NAME=custom_input
TRITON_OUTPUT_NAME=custom_output
```

## Usage Examples

### Example 1: YAML Configuration

```python
from src.config import ServiceConfig
from src.pipeline import ImageEmbeddingPipeline

# Edit config.yaml:
# triton:
#   input_name: "images"
#   output_name: "embeddings"

config = ServiceConfig.from_yaml('configs/config.yaml')
pipeline = ImageEmbeddingPipeline(config)
```

### Example 2: Environment Variables

```python
import os
from src.config import ServiceConfig

os.environ['TRITON_INPUT_NAME'] = 'input_tensor'
os.environ['TRITON_OUTPUT_NAME'] = 'output_tensor'

config = ServiceConfig.from_env()
pipeline = ImageEmbeddingPipeline(config)
```

### Example 3: Direct Client Usage

```python
from src.triton_client import TritonClient

# Set defaults at initialization
client = TritonClient(
    url="localhost:8000",
    model_name="my_model",
    input_name="custom_input",
    output_name="custom_output"
)

# Uses defaults
embeddings = client.infer(preprocessed_images)

# Override per call if needed
embeddings = client.infer(
    preprocessed_images,
    input_name="different_input",
    output_name="different_output"
)
```

### Example 4: Pipeline Integration

```python
from src.config import ServiceConfig
from src.pipeline import ImageEmbeddingPipeline

# Configure via code
config = ServiceConfig()
config.triton.input_name = "images"
config.triton.output_name = "embeddings"

# Create pipeline with custom names
with ImageEmbeddingPipeline(config) as pipeline:
    # All inference calls will use configured names
    results = pipeline.search_images("query.jpg", topk=5)
```

## Benefits

✅ **Flexibility**: Works with any model architecture
✅ **Backward Compatible**: Default values maintain existing behavior
✅ **Multiple Configuration Methods**: YAML, env vars, or code
✅ **Per-Call Override**: Can still override for specific calls
✅ **Type Safe**: Full type hints maintained
✅ **Well Tested**: Comprehensive test coverage

## Testing

A comprehensive test suite was added in `tests/test_triton_node_names.py`:

- ✅ Default value testing
- ✅ Custom value testing
- ✅ YAML serialization/deserialization
- ✅ Environment variable configuration
- ✅ Client initialization with custom names
- ✅ All validation tests pass (6/6)

## Files Modified

1. `src/config.py` - Added input_name/output_name to TritonConfig
2. `src/triton_client.py` - Updated TritonClient to accept and use custom names
3. `src/pipeline.py` - Pass node names from config to client
4. `configs/config.yaml` - Added example configuration
5. `.env.example` - Added environment variable examples
6. `README.md` - Added documentation and usage examples
7. `tests/test_triton_node_names.py` - Added comprehensive tests

## Backward Compatibility

✅ All existing code continues to work without changes
✅ Default values ("input", "output") maintain previous behavior
✅ No breaking changes to API

## Future Enhancements

Potential future improvements:
- Support for multiple input/output tensors
- Automatic node name detection from model metadata
- Node name validation against model schema

---

**Feature Status**: ✅ Complete and Tested
**Backward Compatible**: ✅ Yes
**Tests Passing**: ✅ 6/6
