from __future__ import annotations

import uuid
from typing import Dict, List, Optional, Any, Union
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, Filter, FieldCondition, Range, MatchValue, MatchAny
)


def upsert_vector(
    client: QdrantClient,
    collection_name: str,
    vector_id: str,
    vector: List[float],
    payload: Dict[str, Any]
) -> bool:
    """Insert or update a vector in the collection.
    
    Returns True if successful.
    """
    try:
        point = PointStruct(
            id=vector_id,
            vector=vector,
            payload=payload
        )
        
        client.upsert(
            collection_name=collection_name,
            points=[point]
        )
        return True
    except Exception as e:
        print(f"[qdrant] upsert_vector failed for {vector_id}: {e}")
        return False


def search_vectors(
    client: QdrantClient,
    collection_name: str,
    query_vector: List[float],
    limit: int = 10,
    filters: Optional[Dict[str, Union[str, Dict[str, Any]]]] = None,
    score_threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Search for similar vectors in the collection.
    
    Returns a list of search results with id, score, and payload.
    """
    try:
        # Build filter conditions
        filter_condition = None
        if filters:
            filter_condition = build_filter_conditions(filters)
        
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=filter_condition,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False
        )
        
        # Convert to our format
        results = []
        for point in search_result:
            results.append({
                "id": str(point.id),
                "score": float(point.score),
                "payload": point.payload or {}
            })
        
        return results
    except Exception as e:
        print(f"[qdrant] search_vectors failed: {e}")
        return []


def build_filter_conditions(filters: Dict[str, Union[str, Dict[str, Any]]]) -> Optional[Filter]:
    """Build Qdrant filter conditions from a dictionary.
    
    Supports:
    - {"field": {"$eq": "value"}}
    - {"field": {"$in": ["val1", "val2"]}}
    - {"field": {"$gte": 100, "$lte": 200}}
    - {"$and": [condition1, condition2]}
    """
    conditions = []
    
    for field, condition in filters.items():
        if field == "$and":
            # Handle nested AND conditions
            nested_conditions = []
            if isinstance(condition, list):
                for nested_filter in condition:
                    if isinstance(nested_filter, dict):
                        nested_result = build_filter_conditions(nested_filter)
                        if nested_result:
                            nested_conditions.extend(nested_result.must or [])
            conditions.extend(nested_conditions)
            continue
        
        if isinstance(condition, dict):
            for op, value in condition.items():
                if op == "$eq":
                    conditions.append(
                        FieldCondition(key=field, match=MatchValue(value=value))
                    )
                elif op == "$in":
                    conditions.append(
                        FieldCondition(key=field, match=MatchAny(any=value))
                    )
                elif op in ["$gte", "$lte"]:
                    range_condition = {}
                    if op == "$gte":
                        range_condition["gte"] = value
                    else:
                        range_condition["lte"] = value
                    conditions.append(
                        FieldCondition(key=field, range=Range(**range_condition))
                    )
        else:
            # Direct value match
            conditions.append(
                FieldCondition(key=field, match=MatchValue(value=condition))
            )
    
    return Filter(must=conditions) if conditions else None


def delete_vector(
    client: QdrantClient,
    collection_name: str,
    vector_id: str
) -> bool:
    """Delete a vector from the collection.
    
    Returns True if successful.
    """
    try:
        client.delete(
            collection_name=collection_name,
            points_selector=[vector_id]
        )
        return True
    except Exception as e:
        print(f"[qdrant] delete_vector failed for {vector_id}: {e}")
        return False


def get_vector_info(
    client: QdrantClient,
    collection_name: str,
    vector_id: str
) -> Optional[Dict[str, Any]]:
    """Get information about a specific vector.
    
    Returns the vector payload if found, None otherwise.
    """
    try:
        points = client.retrieve(
            collection_name=collection_name,
            ids=[vector_id],
            with_payload=True,
            with_vectors=False
        )
        
        if points:
            return {
                "id": str(points[0].id),
                "payload": points[0].payload or {}
            }
        return None
    except Exception as e:
        print(f"[qdrant] get_vector_info failed for {vector_id}: {e}")
        return None