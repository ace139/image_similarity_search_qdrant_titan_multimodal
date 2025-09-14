"""
Service for handling image ingestion and embedding generation.
"""
import json
import uuid
from datetime import datetime, timezone
from typing import Tuple, Optional, Dict, Any

from mmfood.aws.s3 import upload_bytes_to_s3, ext_from_mime
from mmfood.aws.session import get_bedrock_client, get_s3_client
from mmfood.bedrock.ai import generate_mm_embedding, generate_image_description
from mmfood.qdrant.client import get_qdrant_client, ensure_collection_exists, validate_collection_config, ensure_payload_indexes
from mmfood.qdrant.operations import upsert_vector
from mmfood.utils.time import to_unix_ts
from mmfood.database import MetricsDatabase, MetricsTimer
from mmfood.config import AppConfig


class IngestService:
    """Service for handling image ingestion workflow."""
    
    def __init__(self, config: AppConfig, metrics_db: MetricsDatabase):
        self.config = config
        self.metrics_db = metrics_db
    
    def generate_description_and_embedding(
        self, 
        image_bytes: bytes, 
        meal_data: Dict[str, Any]
    ) -> Tuple[str, list, float, float]:
        """
        Generate image description and embedding.
        
        Returns:
            Tuple of (description, embedding, description_duration_ms, embedding_duration_ms)
        """
        bedrock = get_bedrock_client(self.config.region, self.config.profile)
        
        # Generate image description
        description_timer = MetricsTimer()
        with description_timer:
            display_text, embed_text = generate_image_description(
                image_bytes=image_bytes,
                meal_data=meal_data,
                bedrock_client=bedrock,
                claude_model_id=self.config.claude_vision_model_id,
            )
        
        # Generate multi-modal embedding
        embedding_timer = MetricsTimer()
        with embedding_timer:
            embedding = generate_mm_embedding(
                bedrock_client=bedrock,
                model_id=self.config.model_id,
                output_dim=self.config.output_dim,
                input_text=embed_text,
                input_image_bytes=image_bytes
            )
        
        # Log embedding generation for ingestion
        self.metrics_db.log_embedding_operation(
            operation_type="ingest",
            model_id=self.config.model_id,
            input_type="multimodal",
            duration_ms=embedding_timer.duration_ms,
            embedding_dimension=len(embedding)
        )
        
        return display_text, embedding, description_timer.duration_ms, embedding_timer.duration_ms
    
    def upload_to_s3_and_index(
        self,
        image_bytes: bytes,
        image_filename: str,
        content_type: Optional[str],
        embedding: list,
        description: str,
        user_id: str,
        meal_datetime: datetime,
        meal_type: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Upload image to S3 and index embedding in Qdrant.
        
        Returns:
            Tuple of (success, upload_details)
        """
        s3 = get_s3_client(self.config.region, self.config.profile)
        qdrant = get_qdrant_client(
            self.config.qdrant_url, 
            self.config.qdrant_api_key, 
            self.config.qdrant_timeout
        )
        
        # Ensure collection exists and validate configuration
        ensure_collection_exists(qdrant, self.config.qdrant_collection_name, len(embedding))
        validate_collection_config(qdrant, self.config.qdrant_collection_name, len(embedding))
        
        # Ensure payload indexes exist for filtering
        ensure_payload_indexes(qdrant, self.config.qdrant_collection_name, ["user_id", "meal_type", "ts"])
        
        # Prepare S3 keys
        image_id = str(uuid.uuid4())
        ext = ext_from_mime(content_type)
        if not ext and isinstance(image_filename, str) and "." in image_filename:
            ext = "." + image_filename.rsplit(".", 1)[-1]
        
        image_key = f"{self.config.images_prefix}{image_id}{ext}"
        vector_key = f"{self.config.embeddings_prefix}{image_id}.json"
        
        # Upload image bytes
        upload_bytes_to_s3(s3, self.config.bucket, image_key, image_bytes, content_type=content_type)
        
        # Build metadata
        ts = to_unix_ts(meal_datetime)
        
        base_record = {
            "model_id": self.config.model_id,
            "embedding_length": len(embedding),
            "s3_image_bucket": self.config.bucket,
            "s3_image_key": image_key,
            "uploaded_filename": image_filename,
            "content_type": content_type,
            "output_embedding_length": self.config.output_dim,
            "region": self.config.region,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            # domain metadata
            "user_id": user_id,
            "meal_type": meal_type,
            "meal_time": meal_datetime.isoformat(),
            "ts": ts,
            "generated_description": description,
            "embedding": embedding
        }
        
        # Upload complete embedding record JSON to S3
        upload_bytes_to_s3(
            s3, self.config.bucket, vector_key,
            json.dumps(base_record).encode("utf-8"),
            content_type="application/json"
        )
        
        # Qdrant: Prepare payload
        qdrant_payload = {
            "user_id": user_id,
            "meal_type": meal_type,
            "ts": ts,
            "s3_image_key": image_key,
            "s3_embedding_key": vector_key,
            "s3_bucket": self.config.bucket,
            "model_id": self.config.model_id,
            "uploaded_filename": image_filename,
            "content_type": content_type,
            "meal_time": meal_datetime.isoformat(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "generated_description": description,
            "embedding_length": len(embedding),
            "output_embedding_length": self.config.output_dim,
            "region": self.config.region
        }
        
        # Insert embedding into Qdrant
        vector_timer = MetricsTimer()
        with vector_timer:
            success = upsert_vector(
                qdrant,
                self.config.qdrant_collection_name,
                image_id,
                [float(x) for x in embedding],
                qdrant_payload
            )
        
        # Log vector upsert operation
        self.metrics_db.log_vector_operation(
            operation_type="upsert",
            collection_name=self.config.qdrant_collection_name,
            duration_ms=vector_timer.duration_ms,
            vector_count=1,
            success=success
        )
        
        upload_details = {
            "image_id": image_id,
            "image_key": image_key,
            "vector_key": vector_key,
            "collection": self.config.qdrant_collection_name,
            "vector_duration_ms": vector_timer.duration_ms
        }
        
        return success, upload_details
