"""
API client example demonstrating FastAPI endpoint usage.
"""

import requests
import json
from typing import List, Dict, Any
from loguru import logger


class EmbeddingAPIClient:
    """Client for the image embedding API."""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def embed_images(
        self,
        inputs: List[str],
        batch_size: int = None,
    ) -> Dict[str, Any]:
        """
        Extract embeddings from images.
        
        Args:
            inputs: List of image paths or URLs
            batch_size: Optional batch size
            
        Returns:
            Response with embeddings
        """
        payload = {"inputs": inputs}
        if batch_size is not None:
            payload["batch_size"] = batch_size
        
        response = requests.post(
            f"{self.base_url}/embed",
            json=payload,
        )
        response.raise_for_status()
        return response.json()
    
    def insert_images(
        self,
        inputs: List[str],
        ids: List[str],
        metadata: List[Dict[str, Any]] = None,
        collection_name: str = None,
        batch_size: int = None,
    ) -> Dict[str, Any]:
        """
        Insert images into Milvus.
        
        Args:
            inputs: List of image paths or URLs
            ids: List of unique IDs
            metadata: Optional metadata for each image
            collection_name: Target collection name
            batch_size: Optional batch size
            
        Returns:
            Response with inserted IDs
        """
        payload = {
            "inputs": inputs,
            "ids": ids,
        }
        
        if metadata is not None:
            payload["metadata"] = metadata
        if collection_name is not None:
            payload["collection_name"] = collection_name
        if batch_size is not None:
            payload["batch_size"] = batch_size
        
        response = requests.post(
            f"{self.base_url}/insert",
            json=payload,
        )
        response.raise_for_status()
        return response.json()
    
    def search_images(
        self,
        query_input: str,
        topk: int = 10,
        filter_expr: str = None,
        collection_name: str = None,
    ) -> Dict[str, Any]:
        """
        Search for similar images.
        
        Args:
            query_input: Query image path or URL
            topk: Number of results
            filter_expr: Optional filter expression
            collection_name: Target collection name
            
        Returns:
            Search results
        """
        payload = {
            "query_input": query_input,
            "topk": topk,
        }
        
        if filter_expr is not None:
            payload["filter_expr"] = filter_expr
        if collection_name is not None:
            payload["collection_name"] = collection_name
        
        response = requests.post(
            f"{self.base_url}/search",
            json=payload,
        )
        response.raise_for_status()
        return response.json()
    
    def create_collection(
        self,
        name: str,
        dim: int = None,
        description: str = "",
    ) -> Dict[str, Any]:
        """
        Create a new collection.
        
        Args:
            name: Collection name
            dim: Embedding dimension
            description: Collection description
            
        Returns:
            Response message
        """
        payload = {
            "name": name,
            "description": description,
        }
        
        if dim is not None:
            payload["dim"] = dim
        
        response = requests.post(
            f"{self.base_url}/collections/create",
            json=payload,
        )
        response.raise_for_status()
        return response.json()
    
    def delete_collection(self, name: str) -> Dict[str, Any]:
        """
        Delete a collection.
        
        Args:
            name: Collection name
            
        Returns:
            Response message
        """
        response = requests.delete(f"{self.base_url}/collections/{name}")
        response.raise_for_status()
        return response.json()
    
    def list_collections(self) -> Dict[str, Any]:
        """
        List all collections.
        
        Returns:
            List of collections
        """
        response = requests.get(f"{self.base_url}/collections")
        response.raise_for_status()
        return response.json()
    
    def get_collection_stats(self, name: str) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Args:
            name: Collection name
            
        Returns:
            Collection statistics
        """
        response = requests.get(f"{self.base_url}/collections/{name}/stats")
        response.raise_for_status()
        return response.json()
    
    def delete_embeddings(
        self,
        ids: List[str] = None,
        filter_expr: str = None,
        collection_name: str = None,
    ) -> Dict[str, Any]:
        """
        Delete embeddings by IDs or filter.
        
        Args:
            ids: List of IDs to delete
            filter_expr: Filter expression
            collection_name: Target collection name
            
        Returns:
            Response message
        """
        payload = {}
        
        if ids is not None:
            payload["ids"] = ids
        if filter_expr is not None:
            payload["filter_expr"] = filter_expr
        if collection_name is not None:
            payload["collection_name"] = collection_name
        
        response = requests.post(
            f"{self.base_url}/delete",
            json=payload,
        )
        response.raise_for_status()
        return response.json()


def main():
    """Example API client usage."""
    
    logger.info("Starting API client example...")
    
    # Initialize client
    client = EmbeddingAPIClient("http://localhost:8080")
    
    try:
        # Check health
        logger.info("Checking service health...")
        health = client.health_check()
        logger.info(f"Service health: {health}")
        
        # Create collection
        collection_name = "api_demo"
        logger.info(f"Creating collection '{collection_name}'...")
        response = client.create_collection(
            name=collection_name,
            dim=512,
            description="API demo collection",
        )
        logger.info(f"Response: {response}")
        
        # Example image paths/URLs
        image_urls = [
            "https://example.com/image1.jpg",
            "https://example.com/image2.jpg",
            "https://example.com/image3.jpg",
        ]
        
        # Insert images
        logger.info("Inserting images...")
        ids = [f"api_img_{i}" for i in range(len(image_urls))]
        metadata = [
            {"source": "api", "category": "test"},
            {"source": "api", "category": "test"},
            {"source": "api", "category": "demo"},
        ]
        
        response = client.insert_images(
            inputs=image_urls,
            ids=ids,
            metadata=metadata,
            collection_name=collection_name,
        )
        logger.info(f"Inserted {response['count']} images")
        
        # Search
        logger.info("Searching for similar images...")
        response = client.search_images(
            query_input=image_urls[0],
            topk=5,
            collection_name=collection_name,
        )
        
        logger.info(f"Search results ({response['count']} found):")
        for i, result in enumerate(response['results'], 1):
            logger.info(
                f"  {i}. ID: {result['id']}, "
                f"Score: {result['score']:.4f}"
            )
        
        # Get collection stats
        stats = client.get_collection_stats(collection_name)
        logger.info(f"Collection stats: {stats}")
        
        # List all collections
        collections = client.list_collections()
        logger.info(f"All collections: {collections}")
        
        # Delete embeddings
        logger.info("Deleting embeddings...")
        response = client.delete_embeddings(
            ids=[ids[0]],
            collection_name=collection_name,
        )
        logger.info(f"Delete response: {response}")
        
        # Cleanup
        logger.info("Cleaning up...")
        client.delete_collection(collection_name)
        
        logger.info("API client example completed!")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise


if __name__ == "__main__":
    main()
