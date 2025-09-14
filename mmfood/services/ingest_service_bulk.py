"""
Service for handling image ingestion and embedding generation (bulk-capable).
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Tuple, Optional, Dict, Any

from qdrant_client.models import PointStruct

from mmfood.aws.s3 import upload_bytes_to_s3, ext_from_mime
from mmfood.aws.session import get_bedrock_client, get_s3_client
from mmfood.bedrock.ai import generate_mm_embedding, generate_image_description
from mmfood.qdrant.client import (
    get_qdrant_client,
    ensure_collection_exists,
    validate_collection_config,
    ensure_payload_indexes,
)
from mmfood.qdrant.operations import upsert_vector, upsert_vectors_batch
from mmfood.utils.time import to_unix_ts
from mmfood.database import MetricsDatabase, MetricsTimer
from mmfood.config import AppConfig


class IngestService:
    """Service for handling image ingestion workflow (regular and bulk)."""

    def __init__(self, config: AppConfig, metrics_db: MetricsDatabase):
        self.config = config
        self.metrics_db = metrics_db

    def generate_description_and_embedding(
        self,
        image_bytes: bytes,
        meal_data: Dict[str, Any],
    ) -> Tuple[str, list, float, float]:
        """Generate image description (Claude Vision) and embedding (Titan MM).

        Returns a tuple: (description, embedding, description_ms, embedding_ms)
        """
        bedrock = get_bedrock_client(self.config.region, self.config.profile)

        # Description
        description_timer = MetricsTimer()
        with description_timer:
            display_text, embed_text = generate_image_description(
                image_bytes=image_bytes,
                meal_data=meal_data,
                bedrock_client=bedrock,
                claude_model_id=self.config.claude_vision_model_id,
            )

        # Embedding
        embedding_timer = MetricsTimer()
        with embedding_timer:
            embedding = generate_mm_embedding(
                bedrock_client=bedrock,
                model_id=self.config.model_id,
                output_dim=self.config.output_dim,
                input_text=embed_text,
                input_image_bytes=image_bytes,
            )

        # Log embed op
        self.metrics_db.log_embedding_operation(
            operation_type="ingest",
            model_id=self.config.model_id,
            input_type="multimodal",
            duration_ms=embedding_timer.duration_ms,
            embedding_dimension=len(embedding),
        )

        return display_text, embedding, description_timer.duration_ms, embedding_timer.duration_ms

    def upload_to_s3_and_prepare_point(
        self,
        image_bytes: bytes,
        image_filename: str,
        content_type: Optional[str],
        embedding: list,
        description: str,
        user_id: str,
        meal_datetime: datetime,
        meal_type: str,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Upload image + embedding JSON to S3 and build a Qdrant PointStruct (no upsert)."""
        s3 = get_s3_client(self.config.region, self.config.profile)

        # Prepare keys
        image_id = str(uuid.uuid4())
        ext = ext_from_mime(content_type)
        if not ext and isinstance(image_filename, str) and "." in image_filename:
            ext = "." + image_filename.rsplit(".", 1)[-1]

        image_key = f"{self.config.images_prefix}{image_id}{ext}"
        vector_key = f"{self.config.embeddings_prefix}{image_id}.json"

        # Upload image
        s3_img_timer = MetricsTimer()
        with s3_img_timer:
            upload_bytes_to_s3(s3, self.config.bucket, image_key, image_bytes, content_type=content_type)

        # Prepare JSON content
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
            "embedding": embedding,
        }

        emb_json_bytes = json.dumps(base_record).encode("utf-8")
        s3_emb_timer = MetricsTimer()
        with s3_emb_timer:
            upload_bytes_to_s3(
                s3,
                self.config.bucket,
                vector_key,
                emb_json_bytes,
                content_type="application/json",
            )

        # Build Qdrant point
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
            "region": self.config.region,
        }

        point = PointStruct(
            id=image_id,
            vector=[float(x) for x in embedding],
            payload=qdrant_payload,
        )

        details = {
            "image_id": image_id,
            "image_key": image_key,
            "vector_key": vector_key,
            "s3_image_upload_ms": s3_img_timer.duration_ms,
            "s3_embedding_upload_ms": s3_emb_timer.duration_ms,
            "embedding_json_size_bytes": len(emb_json_bytes),
            "point": point,
        }
        return True, details

    def upload_to_s3_and_index(
        self,
        image_bytes: bytes,
        image_filename: str,
        content_type: Optional[str],
        embedding: list,
        description: str,
        user_id: str,
        meal_datetime: datetime,
        meal_type: str,
        *,
        target_collection: Optional[str] = None,
        wait_for_qdrant: bool = True,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Upload to S3 and index a single vector into Qdrant (regular path)."""
        s3 = get_s3_client(self.config.region, self.config.profile)
        qdrant = get_qdrant_client(self.config.qdrant_url, self.config.qdrant_api_key, self.config.qdrant_timeout)

        # Resolve collection
        collection_name = target_collection or self.config.qdrant_collection_name

        # Ensure collection
        ensure_collection_exists(qdrant, collection_name, len(embedding))
        validate_collection_config(qdrant, collection_name, len(embedding))
        ensure_payload_indexes(qdrant, collection_name, ["user_id", "meal_type", "ts"])  # safe for bulk too

        # Prepare keys
        image_id = str(uuid.uuid4())
        ext = ext_from_mime(content_type)
        if not ext and isinstance(image_filename, str) and "." in image_filename:
            ext = "." + image_filename.rsplit(".", 1)[-1]
        image_key = f"{self.config.images_prefix}{image_id}{ext}"
        vector_key = f"{self.config.embeddings_prefix}{image_id}.json"

        # Upload image
        s3_img_timer = MetricsTimer()
        with s3_img_timer:
            upload_bytes_to_s3(s3, self.config.bucket, image_key, image_bytes, content_type=content_type)

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
            "embedding": embedding,
        }

        emb_json_bytes = json.dumps(base_record).encode("utf-8")
        s3_emb_timer = MetricsTimer()
        with s3_emb_timer:
            upload_bytes_to_s3(s3, self.config.bucket, vector_key, emb_json_bytes, content_type="application/json")

        # Build payload
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
            "region": self.config.region,
        }

        # Qdrant upsert
        vector_timer = MetricsTimer()
        with vector_timer:
            success = upsert_vector(
                qdrant,
                collection_name,
                image_id,
                [float(x) for x in embedding],
                qdrant_payload,
            )

        # Metrics
        self.metrics_db.log_vector_operation(
            operation_type="upsert",
            collection_name=collection_name,
            duration_ms=vector_timer.duration_ms,
            vector_count=1,
            success=success,
        )

        details = {
            "image_id": image_id,
            "image_key": image_key,
            "vector_key": vector_key,
            "collection": collection_name,
            "vector_duration_ms": vector_timer.duration_ms,
            "s3_image_upload_ms": s3_img_timer.duration_ms,
            "s3_embedding_upload_ms": s3_emb_timer.duration_ms,
            "embedding_json_size_bytes": len(emb_json_bytes),
        }
        return success, details
