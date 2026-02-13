"""
Base preprocessor interface for image preprocessing.

This module defines the protocol/interface that all preprocessors must implement.
Users can create custom preprocessors by implementing this interface.
"""

from typing import Protocol, Union, List, runtime_checkable
import numpy as np


@runtime_checkable
class ImagePreprocessor(Protocol):
    """
    Protocol defining the interface for image preprocessors.
    
    Custom preprocessors must implement this interface to be compatible
    with the ImageEmbeddingPipeline.
    
    Example:
        ```python
        class MyCustomPreprocessor:
            def __init__(self, image_size=(224, 224)):
                self.image_size = image_size
            
            def preprocess_single(self, image: Union[np.ndarray, str]) -> np.ndarray:
                # Your custom preprocessing logic
                ...
                return processed_image
            
            def preprocess_batch(self, images: List[Union[np.ndarray, str]]) -> np.ndarray:
                # Your custom batch preprocessing logic
                ...
                return processed_batch
            
            def __call__(self, images: Union[str, np.ndarray, List[Union[str, np.ndarray]]]) -> np.ndarray:
                # Unified interface
                if isinstance(images, list):
                    return self.preprocess_batch(images)
                else:
                    result = self.preprocess_single(images)
                    return result[np.newaxis, ...]
        ```
    """
    
    def preprocess_single(
        self, 
        image: Union[np.ndarray, str]
    ) -> np.ndarray:
        """
        Preprocess a single image.
        
        Args:
            image: Image array (H, W, C) or path/URL to image
            
        Returns:
            Preprocessed image array. Shape depends on implementation,
            but typically (H, W, C) for NHWC or (C, H, W) for NCHW format.
        """
        ...
    
    def preprocess_batch(
        self,
        images: List[Union[np.ndarray, str]]
    ) -> np.ndarray:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of image arrays or paths/URLs
            
        Returns:
            Batch of preprocessed images. Shape depends on implementation,
            but typically (B, H, W, C) for NHWC or (B, C, H, W) for NCHW format.
        """
        ...
    
    def __call__(
        self,
        images: Union[str, np.ndarray, List[Union[str, np.ndarray]]]
    ) -> np.ndarray:
        """
        Preprocess images (single or batch).
        
        This provides a unified interface for preprocessing.
        
        Args:
            images: Single image or list of images (paths/URLs or arrays)
            
        Returns:
            Preprocessed image(s) with batch dimension.
            Shape: (1, ...) for single image, (B, ...) for batch.
        """
        ...
