"""
Milvus vector database client for embedding storage and search.
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from loguru import logger
import time

try:
    from pymilvus import (
        connections,
        Collection,
        CollectionSchema,
        FieldSchema,
        DataType,
        utility,
    )
except ImportError:
    logger.warning("pymilvus not installed. Install with: pip install pymilvus")
    connections = None
    Collection = None


class MilvusClient:
    """
    High-performance Milvus client for vector operations.
    
    Features:
    - Collection lifecycle management
    - Batch insert optimization
    - Multiple index types (IVF_FLAT, HNSW, FLAT)
    - GPU-accelerated index types (GPU_CAGRA, GPU_IVF_PQ, GPU_IVF_FLAT, GPU_BRUTE_FORCE)
    - Filtered search support
    - Automatic flushing and loading
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "image_embeddings",
        embedding_dim: int = 768,
        vector_field: str = "vector",
        index_type: str = "IVF_FLAT",
        metric_type: str = "IP",
        nlist: int = 128,
        nprobe: int = 16,
        M: int = 16,
        efConstruction: int = 256,
        # GPU index parameters
        intermediate_graph_degree: int = 64,
        graph_degree: int = 32,
        itopk_size: int = 64,
        search_width: int = 4,
        min_iterations: int = 0,
        max_iterations: int = 0,
        team_size: int = 0,
        # GPU_IVF_PQ parameters
        m: int = 8,
        nbits: int = 8,
        alias: str = "default",
    ):
        """
        Initialize Milvus client.
        
        Args:
            host: Milvus server host
            port: Milvus server port
            collection_name: Default collection name
            embedding_dim: Embedding vector dimension
            vector_field: Vector field name in collection schema
            index_type: Index type (IVF_FLAT, HNSW, FLAT, GPU_CAGRA, GPU_IVF_PQ, GPU_IVF_FLAT, GPU_BRUTE_FORCE)
            metric_type: Distance metric (L2, IP, COSINE)
            nlist: Number of cluster units (for IVF_FLAT, GPU_IVF_FLAT, GPU_IVF_PQ)
            nprobe: Number of units to query (for IVF_FLAT, GPU_IVF_FLAT, GPU_IVF_PQ)
            M: Maximum degree of node (for HNSW)
            efConstruction: Construction time/accuracy tradeoff (for HNSW)
            intermediate_graph_degree: Intermediate graph degree (for GPU_CAGRA)
            graph_degree: Graph degree (for GPU_CAGRA)
            itopk_size: itopk size for search (for GPU_CAGRA)
            search_width: Search width (for GPU_CAGRA)
            min_iterations: Min iterations (for GPU_CAGRA)
            max_iterations: Max iterations (for GPU_CAGRA)
            team_size: Team size (for GPU_CAGRA)
            m: Number of subquantizers (for GPU_IVF_PQ)
            nbits: Bits per subquantizer (for GPU_IVF_PQ)
            alias: Connection alias
        """
        self.host = host
        self.port = port
        self.default_collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.vector_field = vector_field
        self.index_type = index_type
        self.metric_type = metric_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.M = M
        self.efConstruction = efConstruction
        # GPU index parameters
        self.intermediate_graph_degree = intermediate_graph_degree
        self.graph_degree = graph_degree
        self.itopk_size = itopk_size
        self.search_width = search_width
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.team_size = team_size
        # GPU_IVF_PQ parameters
        self.m = m
        self.nbits = nbits
        self.alias = alias
        
        # Connect to Milvus
        self._connect()
        
        # Cache for loaded collections
        self._collections: Dict[str, Collection] = {}
    
    def _connect(self) -> None:
        """Connect to Milvus server."""
        try:
            connections.connect(
                alias=self.alias,
                host=self.host,
                port=self.port,
            )
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def disconnect(self) -> None:
        """Disconnect from Milvus server."""
        try:
            connections.disconnect(alias=self.alias)
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.warning(f"Error disconnecting from Milvus: {e}")
    
    def list_collections(self) -> List[str]:
        """
        List all collections.
        
        Returns:
            List of collection names
        """
        try:
            collections = utility.list_collections()
            logger.debug(f"Found {len(collections)} collections")
            return collections
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise
    
    def collection_exists(self, name: str) -> bool:
        """
        Check if a collection exists.
        
        Args:
            name: Collection name
            
        Returns:
            True if collection exists, False otherwise or on error
        """
        try:
            return utility.has_collection(name)
        except Exception as e:
            logger.warning(f"Error checking if collection '{name}' exists: {e}")
            return False  # Assume collection doesn't exist on error
    
    def create_collection(
        self,
        name: Optional[str] = None,
        dim: Optional[int] = None,
        description: str = "",
        auto_id: bool = True,
    ) -> Collection:
        """
        Create a new collection.
        
        Args:
            name: Collection name (uses default if None)
            dim: Embedding dimension (uses default if None)
            description: Collection description
            auto_id: Whether to auto-generate IDs
            
        Returns:
            Created collection
        """
        name = name or self.default_collection_name
        dim = dim or self.embedding_dim
        
        if self.collection_exists(name):
            logger.warning(f"Collection '{name}' already exists")
            return self.get_collection(name)
        
        try:
            # Define schema
            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.INT64,
                    is_primary=True,
                    auto_id=auto_id,
                ),
                FieldSchema(
                    name=self.vector_field,
                    dtype=DataType.FLOAT_VECTOR,
                    dim=dim,
                ),
                FieldSchema(
                    name="metadata",
                    dtype=DataType.JSON,
                ),
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description=description,
            )
            
            # Create collection
            collection = Collection(
                name=name,
                schema=schema,
                using=self.alias,
            )
            
            logger.info(f"Created collection '{name}' with dimension {dim}")
            
            # Create index
            self._create_index(collection)
            
            # Load collection
            collection.load()
            
            # Cache collection
            self._collections[name] = collection
            
            return collection
            
        except Exception as e:
            logger.error(f"Failed to create collection '{name}': {e}")
            raise
    
    def _create_index(self, collection: Collection) -> None:
        """
        Create index on collection.
        
        Args:
            collection: Collection object
        """
        try:
            # Define index parameters based on index type
            if self.index_type == "IVF_FLAT":
                index_params = {
                    "metric_type": self.metric_type,
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": self.nlist},
                }
            elif self.index_type == "HNSW":
                index_params = {
                    "metric_type": self.metric_type,
                    "index_type": "HNSW",
                    "params": {
                        "M": self.M,
                        "efConstruction": self.efConstruction,
                    },
                }
            elif self.index_type == "FLAT":
                index_params = {
                    "metric_type": self.metric_type,
                    "index_type": "FLAT",
                    "params": {},
                }
            elif self.index_type == "GPU_CAGRA":
                # GPU_CAGRA: GPU-accelerated graph-based index
                index_params = {
                    "metric_type": self.metric_type,
                    "index_type": "GPU_CAGRA",
                    "params": {
                        "intermediate_graph_degree": self.intermediate_graph_degree,
                        "graph_degree": self.graph_degree,
                    },
                }
            elif self.index_type == "GPU_IVF_FLAT":
                # GPU_IVF_FLAT: GPU-accelerated IVF index with flat storage
                index_params = {
                    "metric_type": self.metric_type,
                    "index_type": "GPU_IVF_FLAT",
                    "params": {"nlist": self.nlist},
                }
            elif self.index_type == "GPU_IVF_PQ":
                # GPU_IVF_PQ: GPU-accelerated IVF index with product quantization
                index_params = {
                    "metric_type": self.metric_type,
                    "index_type": "GPU_IVF_PQ",
                    "params": {
                        "nlist": self.nlist,
                        "m": self.m,  # Number of subquantizers (configurable)
                        "nbits": self.nbits,  # Bits per subquantizer (configurable)
                    },
                }
            elif self.index_type == "GPU_BRUTE_FORCE":
                # GPU_BRUTE_FORCE: GPU-accelerated brute-force search
                index_params = {
                    "metric_type": self.metric_type,
                    "index_type": "GPU_BRUTE_FORCE",
                    "params": {},
                }
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
            
            # Create index
            collection.create_index(
                field_name=self.vector_field,
                index_params=index_params,
            )
            
            logger.info(f"Created {self.index_type} index on collection '{collection.name}'")
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise
    
    def drop_index(self, collection_name: Optional[str] = None) -> None:
        """
        Drop index from collection to improve insertion performance.
        
        Args:
            collection_name: Target collection name
        """
        collection_name = collection_name or self.default_collection_name
        
        # Check if collection exists first
        if not self.collection_exists(collection_name):
            logger.info(f"Collection '{collection_name}' does not exist, skipping index drop")
            return
        
        try:
            collection = self.get_collection(collection_name)
            
            # Check if index exists
            if collection.has_index():
                collection.drop_index()
                logger.info(f"Dropped index from collection '{collection_name}'")
            else:
                logger.info(f"No index to drop in collection '{collection_name}'")
                
        except Exception as e:
            logger.error(f"Failed to drop index from collection '{collection_name}': {e}")
            raise
    
    def create_index(self, collection_name: Optional[str] = None) -> None:
        """
        Create index on collection after bulk insertion.
        
        Args:
            collection_name: Target collection name
        """
        collection_name = collection_name or self.default_collection_name
        
        # Check if collection exists first
        if not self.collection_exists(collection_name):
            logger.warning(f"Collection '{collection_name}' does not exist, cannot create index")
            return
        
        try:
            collection = self.get_collection(collection_name)
            self._create_index(collection)
            logger.info(f"Created index on collection '{collection_name}'")
            
        except Exception as e:
            logger.error(f"Failed to create index on collection '{collection_name}': {e}")
            raise
    
    def flush_collection(self, collection_name: Optional[str] = None) -> None:
        """
        Manually flush collection to persist data.
        
        Args:
            collection_name: Target collection name
        """
        collection_name = collection_name or self.default_collection_name
        
        # Check if collection exists first
        if not self.collection_exists(collection_name):
            logger.warning(f"Collection '{collection_name}' does not exist, cannot flush")
            return
        
        try:
            collection = self.get_collection(collection_name)
            collection.flush()
            logger.info(f"Flushed collection '{collection_name}'")
            
        except Exception as e:
            logger.error(f"Failed to flush collection '{collection_name}': {e}")
            raise
    
    def get_collection(self, name: Optional[str] = None) -> Collection:
        """
        Get an existing collection.
        
        Args:
            name: Collection name (uses default if None)
            
        Returns:
            Collection object
        """
        name = name or self.default_collection_name
        
        # Check cache
        if name in self._collections:
            return self._collections[name]
        
        if not self.collection_exists(name):
            raise ValueError(f"Collection '{name}' does not exist")
        
        try:
            collection = Collection(name=name, using=self.alias)
            collection.load()
            
            # Cache collection
            self._collections[name] = collection
            
            return collection
            
        except Exception as e:
            logger.error(f"Failed to get collection '{name}': {e}")
            raise
    
    def drop_collection(self, name: Optional[str] = None) -> None:
        """
        Drop a collection.
        
        Args:
            name: Collection name (uses default if None)
        """
        name = name or self.default_collection_name
        
        try:
            if self.collection_exists(name):
                utility.drop_collection(name)
                logger.info(f"Dropped collection '{name}'")
                
                # Remove from cache
                if name in self._collections:
                    del self._collections[name]
            else:
                logger.warning(f"Collection '{name}' does not exist")
                
        except Exception as e:
            logger.error(f"Failed to drop collection '{name}': {e}")
            raise
    
    def insert_embeddings(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
        collection_name: Optional[str] = None,
        auto_flush: bool = True,
        _async: bool = False,
    ):
        """
        Insert embeddings into collection.
        
        Args:
            ids: List of unique IDs
            embeddings: Embedding vectors (N, D)
            metadata: Optional metadata for each embedding
            collection_name: Target collection name
            auto_flush: Whether to flush after insert (default: True)
            _async: Whether to use async insert (Milvus 2.3+, default: False)
            
        Returns:
            List of inserted IDs (or MutationFuture if _async=True)
        """
        collection_name = collection_name or self.default_collection_name
        
        # Validate inputs
        if len(ids) != embeddings.shape[0]:
            raise ValueError("Number of IDs must match number of embeddings")
        
        if metadata is None:
            metadata = [{}] * len(ids)
        elif len(metadata) != len(ids):
            raise ValueError("Number of metadata entries must match number of IDs")
        
        try:
            start_time = time.time()
            
            # Get collection
            collection = self.get_collection(collection_name)
            schema_fields = list(collection.schema.fields)
            
            # Resolve field names from actual collection schema
            vector_field = next(
                (
                    field for field in schema_fields
                    if getattr(field, "name", None) == self.vector_field
                ),
                None,
            )
            if vector_field is None:
                vector_field = next(
                    (
                        field for field in schema_fields
                        if getattr(field, "dtype", None) == DataType.FLOAT_VECTOR
                    ),
                    None,
                )
            
            if vector_field is None:
                raise ValueError(
                    f"No FLOAT_VECTOR field found in collection '{collection_name}'"
                )
            
            if vector_field.name != self.vector_field:
                logger.warning(
                    f"Configured vector field '{self.vector_field}' not found in "
                    f"collection '{collection_name}', using '{vector_field.name}' instead"
                )
            
            # Build insert payload by schema order (excluding auto_id primary field)
            entities: List[Any] = []
            vector_values = embeddings.tolist()
            metadata_field_exists = False
            
            for field in schema_fields:
                field_name = getattr(field, "name", None)
                is_primary = bool(getattr(field, "is_primary", False))
                auto_id = bool(getattr(field, "auto_id", False))
                
                if is_primary and auto_id:
                    continue
                
                if field_name == vector_field.name:
                    entities.append(vector_values)
                    continue
                
                if field_name == "metadata":
                    metadata_field_exists = True
                    entities.append(metadata)
                    continue
                
                if is_primary:
                    if getattr(field, "dtype", None) == DataType.INT64:
                        try:
                            entities.append([int(item) for item in ids])
                        except Exception as conversion_error:
                            raise ValueError(
                                f"Primary key field '{field_name}' expects INT64 values, "
                                f"but got non-integer IDs"
                            ) from conversion_error
                    else:
                        entities.append(ids)
                    continue
                
                raise ValueError(
                    f"Unsupported required field '{field_name}' in collection schema. "
                    f"Only primary key, vector, and metadata fields are currently supported."
                )
            
            if metadata is not None and not metadata_field_exists:
                logger.warning(
                    f"Collection '{collection_name}' has no 'metadata' field; provided metadata will be ignored"
                )
            
            # Insert with optional async mode
            if _async:
                # Async insert (Milvus 2.3+)
                insert_result = collection.insert(entities, _async=True)
                logger.debug(f"Async insert started for {len(ids)} embeddings")
                return insert_result  # Return MutationFuture
            else:
                # Synchronous insert
                insert_result = collection.insert(entities)
                
                # Optional flush to ensure data is written
                if auto_flush:
                    collection.flush()
                
                insert_time = time.time() - start_time
                throughput = len(ids) / insert_time
                inserted_ids = ids
                if hasattr(insert_result, "primary_keys") and insert_result.primary_keys:
                    inserted_ids = [str(primary_key) for primary_key in insert_result.primary_keys]
                
                logger.info(
                    f"Inserted {len(inserted_ids)} embeddings in {insert_time:.3f}s "
                    f"({throughput:.0f} vectors/sec){'' if auto_flush else ' (no flush)'}"
                )
                
                return inserted_ids
            
        except Exception as e:
            logger.error(f"Failed to insert embeddings: {e}")
            raise
    
    def delete_by_ids(
        self,
        ids: List[str],
        collection_name: Optional[str] = None,
    ) -> None:
        """
        Delete embeddings by IDs.
        
        Args:
            ids: List of IDs to delete
            collection_name: Target collection name
        """
        collection_name = collection_name or self.default_collection_name
        
        try:
            collection = self.get_collection(collection_name)
            
            # Build expression
            expr = f"id in {ids}"
            
            # Delete
            collection.delete(expr)
            collection.flush()
            
            logger.info(f"Deleted {len(ids)} embeddings from '{collection_name}'")
            
        except Exception as e:
            logger.error(f"Failed to delete embeddings: {e}")
            raise
    
    def delete_by_filter(
        self,
        expr: str,
        collection_name: Optional[str] = None,
    ) -> None:
        """
        Delete embeddings by filter expression.
        
        Args:
            expr: Filter expression (e.g., "metadata['source'] == 'web'")
            collection_name: Target collection name
        """
        collection_name = collection_name or self.default_collection_name
        
        try:
            collection = self.get_collection(collection_name)
            
            # Delete
            collection.delete(expr)
            collection.flush()
            
            logger.info(f"Deleted embeddings from '{collection_name}' with filter: {expr}")
            
        except Exception as e:
            logger.error(f"Failed to delete embeddings: {e}")
            raise
    
    def search_topk(
        self,
        query_embedding: np.ndarray,
        topk: int = 10,
        filter_expr: Optional[str] = None,
        collection_name: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for top-K most similar embeddings.
        
        Args:
            query_embedding: Query embedding vector (D,) or (1, D)
            topk: Number of results to return
            filter_expr: Optional filter expression
            collection_name: Target collection name
            output_fields: Fields to return in results
            
        Returns:
            List of search results with IDs, scores, and metadata
        """
        collection_name = collection_name or self.default_collection_name
        
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        if output_fields is None:
            output_fields = ["id", "metadata"]
        
        try:
            start_time = time.time()
            
            # Get collection
            collection = self.get_collection(collection_name)
            
            # Define search parameters based on index type
            search_params = {"metric_type": self.metric_type}
            
            if self.index_type == "IVF_FLAT":
                search_params["params"] = {"nprobe": self.nprobe}
            elif self.index_type == "HNSW":
                search_params["params"] = {"ef": 64}
            elif self.index_type == "GPU_CAGRA":
                # GPU_CAGRA search parameters
                search_params["params"] = {
                    "itopk_size": self.itopk_size,
                    "search_width": self.search_width,
                    "min_iterations": self.min_iterations,
                    "max_iterations": self.max_iterations,
                    "team_size": self.team_size,
                }
            elif self.index_type == "GPU_IVF_FLAT":
                # GPU_IVF_FLAT search parameters
                search_params["params"] = {"nprobe": self.nprobe}
            elif self.index_type == "GPU_IVF_PQ":
                # GPU_IVF_PQ search parameters
                search_params["params"] = {"nprobe": self.nprobe}
            elif self.index_type == "GPU_BRUTE_FORCE":
                # GPU_BRUTE_FORCE doesn't need additional params
                search_params["params"] = {}
            
            # Search
            results = collection.search(
                data=query_embedding.tolist(),
                anns_field=self.vector_field,
                param=search_params,
                limit=topk,
                expr=filter_expr,
                output_fields=output_fields,
            )
            
            search_time = time.time() - start_time
            
            logger.debug(f"Search completed in {search_time * 1000:.2f}ms")
            
            # Format results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    result = {
                        "id": hit.id,
                        "score": hit.score,
                        "distance": hit.distance,
                    }
                    
                    # Add requested fields
                    for field in output_fields:
                        if field != "id" and hasattr(hit, 'entity'):
                            result[field] = hit.entity.get(field)
                    
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def get_collection_stats(
        self,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Args:
            collection_name: Target collection name
            
        Returns:
            Dictionary with collection stats
        """
        collection_name = collection_name or self.default_collection_name
        
        try:
            collection = self.get_collection(collection_name)
            
            stats = {
                "name": collection.name,
                "num_entities": collection.num_entities,
                "description": collection.description,
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            raise
