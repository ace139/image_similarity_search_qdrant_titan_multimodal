from __future__ import annotations

from typing import Optional, Tuple


def presign_url(s3_client, bucket: str, key: str, expires_in: int = 3600) -> str:
    return s3_client.generate_presigned_url(
        ClientMethod="get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires_in
    )


def get_object_bytes(s3_client, bucket: str, key: str) -> bytes:
    resp = s3_client.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()


def get_object_bytes_and_meta(s3_client, bucket: str, key: str) -> Tuple[bytes, dict]:
    """Fetch object bytes and minimal metadata for debug display."""
    meta = s3_client.head_object(Bucket=bucket, Key=key)
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    return data, {
        "content_type": meta.get("ContentType"),
        "content_length": meta.get("ContentLength"),
        "etag": meta.get("ETag"),
        "last_modified": str(meta.get("LastModified")),
    }


def upload_bytes_to_s3(s3_client, bucket: str, key: str, data: bytes, content_type: Optional[str] = None):
    extra = {"ContentType": content_type} if content_type else {}
    s3_client.put_object(Bucket=bucket, Key=key, Body=data, **extra)


def normalize_prefix(prefix: str) -> str:
    prefix = prefix.strip()
    if not prefix:
        return ""
    if not prefix.endswith("/"):
        prefix += "/"
    return prefix


def ext_from_mime(mime: Optional[str]) -> str:
    if not mime:
        return ""
    mapping = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
    }
    return mapping.get(mime.lower(), "")
