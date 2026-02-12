#!/usr/bin/env python3
"""
Command-line interface for the image embedding service.
"""

import argparse
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import ImageEmbeddingPipeline
from src.config import ServiceConfig
from loguru import logger


def cmd_embed(args):
    """Extract embeddings from images."""
    config = ServiceConfig.from_yaml(args.config) if args.config else ServiceConfig.from_env()
    
    with ImageEmbeddingPipeline(config) as pipeline:
        embeddings = pipeline.embed_images(args.inputs, batch_size=args.batch_size)
        
        if args.output:
            import numpy as np
            np.save(args.output, embeddings)
            logger.info(f"Saved embeddings to {args.output}")
        else:
            print(f"Shape: {embeddings.shape}")
            print(f"Embeddings:\n{embeddings}")


def cmd_insert(args):
    """Insert images into Milvus."""
    config = ServiceConfig.from_yaml(args.config) if args.config else ServiceConfig.from_env()
    
    # Load metadata from file if provided
    metadata = None
    if args.metadata:
        with open(args.metadata, 'r') as f:
            metadata = json.load(f)
    
    with ImageEmbeddingPipeline(config) as pipeline:
        # Create collection if needed
        if args.create_collection:
            pipeline.create_collection(
                name=args.collection or config.milvus.collection_name,
                dim=config.milvus.embedding_dim,
            )
        
        inserted_ids = pipeline.insert_images(
            inputs=args.inputs,
            ids=args.ids,
            metadata=metadata,
            collection_name=args.collection,
            batch_size=args.batch_size,
        )
        
        logger.info(f"Inserted {len(inserted_ids)} images: {inserted_ids}")


def cmd_search(args):
    """Search for similar images."""
    config = ServiceConfig.from_yaml(args.config) if args.config else ServiceConfig.from_env()
    
    with ImageEmbeddingPipeline(config) as pipeline:
        results = pipeline.search_images(
            query_input=args.query,
            topk=args.topk,
            filter_expr=args.filter,
            collection_name=args.collection,
        )
        
        logger.info(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. ID: {result['id']}, Score: {result['score']:.4f}, Metadata: {result.get('metadata', {})}")


def cmd_collection(args):
    """Manage collections."""
    config = ServiceConfig.from_yaml(args.config) if args.config else ServiceConfig.from_env()
    
    with ImageEmbeddingPipeline(config) as pipeline:
        if args.action == "list":
            collections = pipeline.list_collections()
            logger.info(f"Collections: {collections}")
        
        elif args.action == "create":
            pipeline.create_collection(
                name=args.name,
                dim=args.dim or config.milvus.embedding_dim,
                description=args.description,
            )
            logger.info(f"Created collection '{args.name}'")
        
        elif args.action == "delete":
            pipeline.delete_collection(args.name)
            logger.info(f"Deleted collection '{args.name}'")
        
        elif args.action == "stats":
            stats = pipeline.get_collection_stats(args.name)
            logger.info(f"Collection stats: {json.dumps(stats, indent=2)}")


def cmd_health(args):
    """Check service health."""
    config = ServiceConfig.from_yaml(args.config) if args.config else ServiceConfig.from_env()
    
    with ImageEmbeddingPipeline(config) as pipeline:
        health = pipeline.health_check()
        logger.info(f"Service health:")
        for component, status in health.items():
            status_str = "✓" if status else "✗"
            logger.info(f"  {status_str} {component}: {'healthy' if status else 'unhealthy'}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Image Embedding Service CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config YAML file",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Embed command
    embed_parser = subparsers.add_parser("embed", help="Extract embeddings from images")
    embed_parser.add_argument("inputs", nargs="+", help="Image paths or URLs")
    embed_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    embed_parser.add_argument("--output", type=str, help="Output file (.npy)")
    embed_parser.set_defaults(func=cmd_embed)
    
    # Insert command
    insert_parser = subparsers.add_parser("insert", help="Insert images into Milvus")
    insert_parser.add_argument("inputs", nargs="+", help="Image paths or URLs")
    insert_parser.add_argument("--ids", nargs="+", required=True, help="Image IDs")
    insert_parser.add_argument("--metadata", type=str, help="Metadata JSON file")
    insert_parser.add_argument("--collection", type=str, help="Collection name")
    insert_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    insert_parser.add_argument("--create-collection", action="store_true", help="Create collection if not exists")
    insert_parser.set_defaults(func=cmd_insert)
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for similar images")
    search_parser.add_argument("query", help="Query image path or URL")
    search_parser.add_argument("--topk", type=int, default=10, help="Number of results")
    search_parser.add_argument("--filter", type=str, help="Filter expression")
    search_parser.add_argument("--collection", type=str, help="Collection name")
    search_parser.set_defaults(func=cmd_search)
    
    # Collection command
    collection_parser = subparsers.add_parser("collection", help="Manage collections")
    collection_parser.add_argument(
        "action",
        choices=["list", "create", "delete", "stats"],
        help="Action to perform",
    )
    collection_parser.add_argument("--name", type=str, help="Collection name")
    collection_parser.add_argument("--dim", type=int, help="Embedding dimension")
    collection_parser.add_argument("--description", type=str, default="", help="Collection description")
    collection_parser.set_defaults(func=cmd_collection)
    
    # Health command
    health_parser = subparsers.add_parser("health", help="Check service health")
    health_parser.set_defaults(func=cmd_health)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        args.func(args)
        return 0
    except Exception as e:
        logger.error(f"Command failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
