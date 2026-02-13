"""
FastAPI server for image embedding service.
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import numpy as np
from loguru import logger
import tempfile
import os

from .pipeline import ImageEmbeddingPipeline
from .config import ServiceConfig

# Initialize FastAPI app
app = FastAPI(
    title="Image Embedding Service",
    description="Production-ready image embedding service with JAX, Triton, and Milvus",
    version="0.1.0",
)

# Global pipeline instance
pipeline: Optional[ImageEmbeddingPipeline] = None


# Pydantic models
class EmbedRequest(BaseModel):
    """Request for embedding extraction."""
    inputs: List[str] = Field(..., description="List of image paths or URLs")
    batch_size: Optional[int] = Field(None, description="Batch size for processing")


class EmbedResponse(BaseModel):
    """Response for embedding extraction."""
    embeddings: List[List[float]] = Field(..., description="Extracted embeddings")
    count: int = Field(..., description="Number of embeddings")
    embedding_dim: int = Field(..., description="Embedding dimension")


class InsertRequest(BaseModel):
    """Request for inserting images."""
    inputs: List[str] = Field(..., description="List of image paths or URLs")
    ids: List[str] = Field(..., description="List of unique IDs")
    metadata: Optional[List[Dict[str, Any]]] = Field(None, description="Optional metadata")
    collection_name: Optional[str] = Field(None, description="Target collection name")
    batch_size: Optional[int] = Field(None, description="Batch size for processing")


class InsertResponse(BaseModel):
    """Response for insert operation."""
    inserted_ids: List[str] = Field(..., description="List of inserted IDs")
    count: int = Field(..., description="Number of inserted embeddings")


class SearchRequest(BaseModel):
    """Request for image search."""
    query_input: str = Field(..., description="Query image path or URL")
    topk: int = Field(10, ge=1, le=1000, description="Number of results to return")
    filter_expr: Optional[str] = Field(None, description="Filter expression")
    collection_name: Optional[str] = Field(None, description="Target collection name")


class SearchResponse(BaseModel):
    """Response for search operation."""
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    count: int = Field(..., description="Number of results")
    query_time_ms: float = Field(..., description="Query time in milliseconds")


class CreateCollectionRequest(BaseModel):
    """Request for creating a collection."""
    name: str = Field(..., description="Collection name")
    dim: Optional[int] = Field(None, description="Embedding dimension")
    description: str = Field("", description="Collection description")


class DeleteRequest(BaseModel):
    """Request for deleting embeddings."""
    ids: Optional[List[str]] = Field(None, description="List of IDs to delete")
    filter_expr: Optional[str] = Field(None, description="Filter expression for deletion")
    collection_name: Optional[str] = Field(None, description="Target collection name")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Overall status")
    components: Dict[str, bool] = Field(..., description="Component health status")


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup."""
    global pipeline
    
    logger.info("Starting image embedding service...")
    
    try:
        # Load configuration
        config = ServiceConfig.from_env()
        
        # Initialize pipeline
        pipeline = ImageEmbeddingPipeline(config)
        
        logger.info("Service started successfully")
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global pipeline
    
    logger.info("Shutting down image embedding service...")
    
    if pipeline is not None:
        pipeline.close()
    
    logger.info("Service shut down")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "service": "Image Embedding Service",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        health = pipeline.health_check()
        
        # Determine overall status
        all_healthy = all(health.values())
        status = "healthy" if all_healthy else "degraded"
        
        return HealthResponse(
            status=status,
            components=health,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed", response_model=EmbedResponse)
async def embed_images(request: EmbedRequest):
    """
    Extract embeddings from images.
    
    Args:
        request: Embed request with image paths/URLs
        
    Returns:
        Extracted embeddings
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        embeddings = pipeline.embed_images(
            inputs=request.inputs,
            batch_size=request.batch_size,
        )
        
        return EmbedResponse(
            embeddings=embeddings.tolist(),
            count=len(embeddings),
            embedding_dim=embeddings.shape[1],
        )
    except Exception as e:
        logger.error(f"Embedding extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/insert", response_model=InsertResponse)
async def insert_images(request: InsertRequest):
    """
    Extract embeddings and insert into Milvus.
    
    Args:
        request: Insert request with images and metadata
        
    Returns:
        Inserted IDs
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        inserted_ids = pipeline.insert_images(
            inputs=request.inputs,
            ids=request.ids,
            metadata=request.metadata,
            collection_name=request.collection_name,
            batch_size=request.batch_size,
        )
        
        return InsertResponse(
            inserted_ids=inserted_ids,
            count=len(inserted_ids),
        )
    except Exception as e:
        logger.error(f"Insert failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search_images(request: SearchRequest):
    """
    Search for similar images.
    
    Args:
        request: Search request with query image
        
    Returns:
        Search results
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        import time
        start_time = time.time()
        
        results = pipeline.search_images(
            query_input=request.query_input,
            topk=request.topk,
            filter_expr=request.filter_expr,
            collection_name=request.collection_name,
        )
        
        query_time = (time.time() - start_time) * 1000
        
        return SearchResponse(
            results=results,
            count=len(results),
            query_time_ms=query_time,
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/collections/create")
async def create_collection(request: CreateCollectionRequest):
    """
    Create a new collection.
    
    Args:
        request: Create collection request
        
    Returns:
        Success message
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        pipeline.create_collection(
            name=request.name,
            dim=request.dim,
            description=request.description,
        )
        
        return {"message": f"Collection '{request.name}' created successfully"}
    except Exception as e:
        logger.error(f"Collection creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/collections/{name}")
async def delete_collection(name: str):
    """
    Delete a collection.
    
    Args:
        name: Collection name
        
    Returns:
        Success message
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        pipeline.delete_collection(name)
        
        return {"message": f"Collection '{name}' deleted successfully"}
    except Exception as e:
        logger.error(f"Collection deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections")
async def list_collections():
    """
    List all collections.
    
    Returns:
        List of collection names
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        collections = pipeline.list_collections()
        
        return {"collections": collections, "count": len(collections)}
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections/{name}/stats")
async def get_collection_stats(name: str):
    """
    Get collection statistics.
    
    Args:
        name: Collection name
        
    Returns:
        Collection statistics
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        stats = pipeline.get_collection_stats(name)
        
        return stats
    except Exception as e:
        logger.error(f"Failed to get collection stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/delete")
async def delete_embeddings(request: DeleteRequest):
    """
    Delete embeddings by IDs or filter.
    
    Args:
        request: Delete request
        
    Returns:
        Success message
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        if request.ids is not None:
            pipeline.delete_by_ids(
                ids=request.ids,
                collection_name=request.collection_name,
            )
            return {"message": f"Deleted {len(request.ids)} embeddings"}
        elif request.filter_expr is not None:
            pipeline.delete_by_filter(
                expr=request.filter_expr,
                collection_name=request.collection_name,
            )
            return {"message": f"Deleted embeddings with filter: {request.filter_expr}"}
        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'ids' or 'filter_expr' must be provided",
            )
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    logger.add("logs/api_server.log", rotation="500 MB", retention="10 days")
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info",
    )
