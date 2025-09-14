"""
Service for handling search queries and vector operations.
"""
from datetime import datetime, date, time as dtime
from typing import List, Dict, Any, Optional, Tuple

from mmfood.aws.session import get_bedrock_client, get_s3_client
from mmfood.bedrock.ai import generate_mm_embedding
from mmfood.qdrant.client import get_qdrant_client, validate_collection_config, ensure_payload_indexes
from mmfood.qdrant.operations import search_vectors
from mmfood.utils.time import to_unix_ts
from mmfood.database import MetricsDatabase, MetricsTimer
from mmfood.config import AppConfig


class SearchService:
    """Service for handling search operations."""
    
    def __init__(self, config: AppConfig, metrics_db: MetricsDatabase):
        self.config = config
        self.metrics_db = metrics_db
    
    def execute_search(
        self,
        user_id: str,
        query_mode: str,
        query_text: Optional[str] = None,
        query_image_bytes: Optional[bytes] = None,
        query_image_filename: Optional[str] = None,
        date_range: Optional[Tuple[date, date]] = None,
        meal_types: Optional[List[str]] = None,
        top_k: int = 5,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a complete search operation.
        
        Returns:
            Dictionary containing search results and performance metrics
        """
        # Start metrics logging for this request
        filters_dict = {"user_id": user_id}
        if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
            start_d, end_d = date_range
            if isinstance(start_d, date) and isinstance(end_d, date):
                filters_dict["date_range"] = [start_d.isoformat(), end_d.isoformat()]
        if meal_types:
            filters_dict["meal_types"] = meal_types
            
        request_id = self.metrics_db.log_request_start(
            user_id=user_id,
            query_type=query_mode.lower(),
            query_text=query_text,
            query_image_path=query_image_filename,
            filters=filters_dict,
            top_k=top_k,
            session_id=session_id
        )
        
        total_timer = MetricsTimer()
        embedding_timer = MetricsTimer()
        search_timer = MetricsTimer()
        
        try:
            with total_timer:
                # Initialize clients
                bedrock = get_bedrock_client(self.config.region, self.config.profile)
                s3 = get_s3_client(self.config.region, self.config.profile)
                qdrant = get_qdrant_client(
                    self.config.qdrant_url, 
                    self.config.qdrant_api_key, 
                    self.config.qdrant_timeout
                )
                
                # Validate collection configuration
                validate_collection_config(qdrant, self.config.qdrant_collection_name, self.config.output_dim)
                
                # Ensure payload indexes exist for filtering
                ensure_payload_indexes(qdrant, self.config.qdrant_collection_name, ["user_id", "meal_type", "ts"])
                
                # Create query embedding
                with embedding_timer:
                    q_embedding = generate_mm_embedding(
                        bedrock_client=bedrock,
                        model_id=self.config.model_id,
                        output_dim=self.config.output_dim,
                        input_text=query_text if query_mode.lower() == "text" else None,
                        input_image_bytes=query_image_bytes if query_mode.lower() == "image" else None,
                    )
                
                # Log embedding operation
                self.metrics_db.log_embedding_operation(
                    operation_type="query",
                    model_id=self.config.model_id,
                    input_type="multimodal" if query_mode.lower() == "image" else "text",
                    duration_ms=embedding_timer.duration_ms,
                    embedding_dimension=len(q_embedding),
                    request_id=request_id
                )
                
                # Build filter conditions
                filters = self._build_filters(user_id, date_range, meal_types)
                
                # Search vectors
                with search_timer:
                    results = search_vectors(
                        qdrant,
                        self.config.qdrant_collection_name,
                        [float(x) for x in q_embedding],
                        limit=int(top_k),
                        filters=filters,
                        score_threshold=0.1
                    )
                
                # Log vector search operation
                self.metrics_db.log_vector_operation(
                    operation_type="search",
                    collection_name=self.config.qdrant_collection_name,
                    duration_ms=search_timer.duration_ms,
                    vector_count=len(results),
                    request_id=request_id
                )
            
            # Log search results
            self.metrics_db.log_search_results(request_id, results)
            
            # Log request completion
            self.metrics_db.log_request_completion(
                request_id=request_id,
                total_duration_ms=total_timer.duration_ms,
                embedding_duration_ms=embedding_timer.duration_ms,
                search_duration_ms=search_timer.duration_ms,
                results_count=len(results),
                success=True
            )
            
            return {
                "success": True,
                "results": results,
                "request_id": request_id,
                "s3_client": s3,  # Needed for result image fetching
                "performance": {
                    "total_duration_ms": total_timer.duration_ms,
                    "embedding_duration_ms": embedding_timer.duration_ms,
                    "search_duration_ms": search_timer.duration_ms
                }
            }
            
        except Exception as e:
            # Log error
            self.metrics_db.log_request_completion(
                request_id=request_id,
                total_duration_ms=total_timer.duration_ms if total_timer.duration_ms else 0,
                embedding_duration_ms=embedding_timer.duration_ms if embedding_timer.duration_ms else 0,
                search_duration_ms=search_timer.duration_ms if search_timer.duration_ms else 0,
                results_count=0,
                success=False,
                error_message=str(e)
            )
            
            return {
                "success": False,
                "error": str(e),
                "request_id": request_id,
                "performance": {
                    "total_duration_ms": total_timer.duration_ms if total_timer.duration_ms else 0,
                    "embedding_duration_ms": embedding_timer.duration_ms if embedding_timer.duration_ms else 0,
                    "search_duration_ms": search_timer.duration_ms if search_timer.duration_ms else 0
                }
            }
    
    def _build_filters(
        self,
        user_id: str,
        date_range: Optional[Tuple[date, date]],
        meal_types: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Build Qdrant filter conditions."""
        filters = {"user_id": {"$eq": user_id}}
        
        # Date range handling
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_d, end_d = date_range
            if isinstance(start_d, date) and isinstance(end_d, date):
                start_dt = datetime.combine(start_d, dtime(0, 0, 0))
                end_dt = datetime.combine(end_d, dtime(23, 59, 59))
                filters["ts"] = {
                    "$gte": to_unix_ts(start_dt), 
                    "$lte": to_unix_ts(end_dt)
                }
        
        if meal_types:
            filters["meal_type"] = {"$in": meal_types}
        
        return filters
