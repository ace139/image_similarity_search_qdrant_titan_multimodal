from __future__ import annotations

import os
from typing import Optional, List
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, CollectionInfo, PayloadSchemaType,
    CreateFieldIndex, FieldCondition, MatchValue
)
from qdrant_client.http.exceptions import UnexpectedResponse


def get_qdrant_client(
    url: str,
    api_key: Optional[str] = None,
    timeout: int = 60
) -> QdrantClient:
    """Create and return a Qdrant client."""
    return QdrantClient(
        url=url,
        api_key=api_key,
        timeout=timeout,
        https=url.startswith('https://'),
        prefer_grpc=False  # Use HTTP for better compatibility
    )


def ensure_collection_exists(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    distance: Distance = Distance.COSINE
) -> bool:
    """Ensure the collection exists with proper configuration.
    
    Returns True if collection was created, False if it already existed.
    """
    try:
        collection_info = client.get_collection(collection_name)
        # Validate vector size matches
        vectors_config = collection_info.config.params.vectors
        if vectors_config is None:
            raise ValueError(f"Collection '{collection_name}' has no vector configuration")
        
        if isinstance(vectors_config, dict):
            # Handle case where vectors is a dictionary (named vectors)
            if not vectors_config:
                raise ValueError(f"Collection '{collection_name}' has no vector configurations")
            # Get the first (or default) vector configuration
            vector_config = next(iter(vectors_config.values()))
            actual_size = vector_config.size
        else:
            # Handle case where vectors is a single VectorParams object
            actual_size = vectors_config.size
        
        if actual_size != vector_size:
            raise ValueError(
                f"Collection '{collection_name}' has vector size "
                f"{actual_size}, but expected {vector_size}"
            )
        return False  # Collection already exists
    except UnexpectedResponse as e:
        if e.status_code == 404:
            # Collection doesn't exist, create it
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )
            return True  # Collection was created
        raise  # Re-raise other errors


def ensure_payload_indexes(
    client: QdrantClient,
    collection_name: str,
    field_names: List[str]
) -> None:
    """Ensure payload field indexes exist for the specified fields.
    
    This is required for filtering operations in Qdrant.
    """
    for field_name in field_names:
        try:
            # Determine field type based on field name
            if field_name == "ts":
                # ts is a numeric timestamp, use INTEGER type
                field_schema = PayloadSchemaType.INTEGER
            else:
                # user_id, meal_type are strings, use KEYWORD type
                field_schema = PayloadSchemaType.KEYWORD
                
            # Try to create the index - this will succeed if it doesn't exist
            # and be ignored if it already exists
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema
            )
            print(f"[qdrant] Ensured index exists for field: {field_name} (type: {field_schema})")
        except UnexpectedResponse as e:
            # Index might already exist or there could be another issue
            if "already exists" in str(e).lower() or "index exists" in str(e).lower():
                print(f"[qdrant] Index already exists for field: {field_name}")
            else:
                print(f"[qdrant] Warning: Could not create index for {field_name}: {e}")
        except Exception as e:
            print(f"[qdrant] Warning: Could not create index for {field_name}: {e}")


def validate_collection_config(
    client: QdrantClient,
    collection_name: str,
    expected_vector_size: int
) -> CollectionInfo:
    """Validate that the collection exists and has the correct configuration."""
    try:
        collection_info = client.get_collection(collection_name)
    except UnexpectedResponse as e:
        if e.status_code == 404:
            raise ValueError(
                f"Collection '{collection_name}' does not exist. "
                f"Please create it first or use ensure_collection_exists()."
            )
        raise

    vectors_config = collection_info.config.params.vectors
    if vectors_config is None:
        raise ValueError(f"Collection '{collection_name}' has no vector configuration")
    
    if isinstance(vectors_config, dict):
        # Handle case where vectors is a dictionary (named vectors)
        if not vectors_config:
            raise ValueError(f"Collection '{collection_name}' has no vector configurations")
        # Get the first (or default) vector configuration
        vector_config = next(iter(vectors_config.values()))
        actual_size = vector_config.size
    else:
        # Handle case where vectors is a single VectorParams object
        actual_size = vectors_config.size
    
    if actual_size != expected_vector_size:
        raise ValueError(
            f"Collection '{collection_name}' has vector size {actual_size}, "
            f"but expected {expected_vector_size}"
        )
    
    return collection_info