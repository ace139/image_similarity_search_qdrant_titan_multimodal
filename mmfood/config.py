from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, List, Mapping

from mmfood.aws.s3 import normalize_prefix
from mmfood.bedrock.ai import DEFAULT_CLAUDE_VISION_PROFILE


@dataclass
class AppConfig:
    region: str
    profile: Optional[str]

    bucket: str
    images_prefix: str
    embeddings_prefix: str

    model_id: str
    output_dim: int

    # Qdrant configuration
    qdrant_url: str
    qdrant_api_key: Optional[str]
    qdrant_collection_name: str
    qdrant_timeout: int

    claude_vision_model_id: str

    def missing_required(self) -> List[str]:
        missing: List[str] = []
        if not self.bucket:
            missing.append("APP_S3_BUCKET")

        # Qdrant requirements
        if not self.qdrant_url:
            missing.append("QDRANT_URL")
        if not self.qdrant_collection_name:
            missing.append("QDRANT_COLLECTION_NAME")
        
        return missing

    def validate(self) -> None:
        missing = self.missing_required()
        if missing:
            raise ValueError(
                "Missing required environment variables: " + ", ".join(missing)
            )


def load_config(env: Mapping[str, str] | None = None) -> AppConfig:
    env = os.environ if env is None else env

    # Region & profile
    region = env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION") or "us-east-1"
    env_aki = env.get("AWS_ACCESS_KEY_ID")
    env_sak = env.get("AWS_SECRET_ACCESS_KEY")
    profile = None if (env_aki and env_sak) else (env.get("AWS_PROFILE") or None)

    # App S3
    bucket = env.get("APP_S3_BUCKET") or ""
    images_prefix = normalize_prefix(env.get("APP_IMAGES_PREFIX", "images/"))
    embeddings_prefix = normalize_prefix(env.get("APP_EMBEDDINGS_PREFIX", "embeddings/"))

    # Models
    model_id = env.get("MODEL_ID", "amazon.titan-embed-image-v1")
    output_dim = int(env.get("OUTPUT_EMBEDDING_LENGTH", "1024"))

    # Qdrant configuration
    qdrant_url = env.get("QDRANT_URL") or ""
    qdrant_api_key = env.get("QDRANT_API_KEY") or None
    qdrant_collection_name = env.get("QDRANT_COLLECTION_NAME", "food_embeddings")
    qdrant_timeout = int(env.get("QDRANT_TIMEOUT", "60"))

    # Claude Vision inference profile (ID/ARN)
    claude_model = env.get("CLAUDE_VISION_MODEL_ID", DEFAULT_CLAUDE_VISION_PROFILE)

    return AppConfig(
        region=region,
        profile=profile,
        bucket=bucket,
        images_prefix=images_prefix,
        embeddings_prefix=embeddings_prefix,
        model_id=model_id,
        output_dim=output_dim,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        qdrant_collection_name=qdrant_collection_name,
        qdrant_timeout=qdrant_timeout,
        claude_vision_model_id=claude_model,
    )
