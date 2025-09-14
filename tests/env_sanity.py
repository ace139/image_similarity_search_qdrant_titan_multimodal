#!/usr/bin/env python3
"""
Environment Sanity Checks (Python-only)

Runs the following checks in order using only environment variables (no AWS profile fallback):
1) AWS tokens validity (STS)
2) S3 configuration and access
3) Titan embedding model availability (and optional invoke)
4) Qdrant accessibility and collection configuration

Usage:
  python tests/env_sanity.py                 # Run with default read-only checks
  python tests/env_sanity.py --write-s3      # Also perform a tiny write+delete check in S3
  python tests/env_sanity.py --invoke-bedrock  # Attempt a minimal Titan invocation (may incur small cost)
  python tests/env_sanity.py --json          # Output JSON summary

You can also control optional checks via env vars (parsed as bools):
  SANITY_WRITE_TESTS=1
  SANITY_BEDROCK_INVOKE=1

Requires:
- boto3, botocore
- qdrant-client
- python-dotenv (optional; if available, .env will be loaded)

This script does NOT use AWS profiles; it relies solely on environment variables, matching the app's behavior.
"""
from __future__ import annotations

import os
import sys
import time
import json
import argparse
import dataclasses
from typing import Dict, Any
from pathlib import Path

# Ensure project root is on sys.path so 'mmfood' package can be imported when running as a script
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    def load_dotenv():  # type: ignore
        return False

import botocore
from botocore.exceptions import ClientError

# Import helper functions from the application modules
from mmfood.aws.session import _get_boto3_session, get_s3_client, get_bedrock_client
from mmfood.qdrant.client import get_qdrant_client, validate_collection_config


@dataclasses.dataclass(frozen=True)
class Config:
    # AWS (no fallbacks)
    region: str
    profile: str | None
    # S3 (no fallbacks)
    bucket: str  # APP_S3_BUCKET
    images_prefix: str  # APP_IMAGES_PREFIX
    embeddings_prefix: str  # APP_EMBEDDINGS_PREFIX
    # Titan Bedrock (no fallbacks)
    model_id: str  # MODEL_ID
    output_dim: int  # OUTPUT_EMBEDDING_LENGTH
    # Qdrant (no fallbacks for required values)
    qdrant_url: str  # QDRANT_URL
    qdrant_api_key: str | None  # QDRANT_API_KEY (optional)
    qdrant_collection_name: str  # QDRANT_COLLECTION_NAME
    qdrant_timeout: int  # QDRANT_TIMEOUT


def load_config() -> Config:
    """Load config strictly from environment. Exit if any required variable is missing."""
    required_vars = [
        # AWS
        "AWS_REGION",
        # S3 Storage (from .env.example)
        "APP_S3_BUCKET",
        "APP_IMAGES_PREFIX",
        "APP_EMBEDDINGS_PREFIX",
        # AI Models (from .env.example)
        "MODEL_ID",
        "OUTPUT_EMBEDDING_LENGTH",
        # Qdrant
        "QDRANT_URL",
        "QDRANT_COLLECTION_NAME",
        "QDRANT_TIMEOUT",
    ]
    missing = [k for k in required_vars if not os.getenv(k)]
    if missing:
        print("Missing required environment variables:", file=sys.stderr)
        for k in missing:
            print(f"  - {k}", file=sys.stderr)
        sys.exit(2)

    # Parse integer values with validation
    try:
        output_dim = int(os.getenv("OUTPUT_EMBEDDING_LENGTH", ""))
    except Exception:
        print("Invalid OUTPUT_EMBEDDING_LENGTH: must be an integer", file=sys.stderr)
        sys.exit(2)

    try:
        qdrant_timeout = int(os.getenv("QDRANT_TIMEOUT", ""))
    except Exception:
        print("Invalid QDRANT_TIMEOUT: must be an integer", file=sys.stderr)
        sys.exit(2)

    return Config(
        region=os.getenv("AWS_REGION", ""),
        profile=os.getenv("AWS_PROFILE") or None,  # optional
        bucket=os.getenv("APP_S3_BUCKET", ""),
        images_prefix=os.getenv("APP_IMAGES_PREFIX", ""),
        embeddings_prefix=os.getenv("APP_EMBEDDINGS_PREFIX", ""),
        model_id=os.getenv("MODEL_ID", ""),
        output_dim=output_dim,
        qdrant_url=os.getenv("QDRANT_URL", ""),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        qdrant_collection_name=os.getenv("QDRANT_COLLECTION_NAME", ""),
        qdrant_timeout=qdrant_timeout,
    )



# --- Formatting helpers ---
class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def supports_color() -> bool:
    return sys.stdout.isatty()


def ctext(text: str, color: str) -> str:
    if supports_color():
        return f"{color}{text}{Colors.END}"
    return text


def b(text: str) -> str:
    return ctext(text, Colors.BOLD)


def ok_mark(ok: bool) -> str:
    return ctext("✓" if ok else "✗", Colors.GREEN if ok else Colors.RED)


def to_bool(val: str | None, default: bool = False) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


# --- Tests ---

def test_aws_tokens(cfg) -> Dict[str, Any]:
    """Validate AWS tokens via STS GetCallerIdentity using env-only session."""
    result: Dict[str, Any] = {"ok": False}
    try:
        session = _get_boto3_session(cfg.region, cfg.profile)
        sts = session.client("sts")
        identity = sts.get_caller_identity()
        result.update(
            {
                "ok": True,
                "account": identity.get("Account"),
                "arn": identity.get("Arn"),
                "user_id": identity.get("UserId"),
                "region": cfg.region,
            }
        )
    except Exception as e:  # ValueError from session policy or AWS errors
        result.update({"error": str(e)})
    return result


def test_s3(cfg, do_write: bool = False) -> Dict[str, Any]:
    """Check S3 bucket access. Optionally perform a tiny write+delete to verify write perms."""
    result: Dict[str, Any] = {
        "ok": False,
        "bucket": cfg.bucket,
        "images_prefix": cfg.images_prefix,
        "embeddings_prefix": cfg.embeddings_prefix,
        "head_bucket_ok": False,
        "list_ok": False,
        "write_ok": None,
        "cleanup_ok": None,
    }
    try:
        s3 = get_s3_client(cfg.region, cfg.profile)
        # Head bucket
        s3.head_bucket(Bucket=cfg.bucket)
        result["head_bucket_ok"] = True
        # List a few objects under images prefix
        s3.list_objects_v2(Bucket=cfg.bucket, Prefix=cfg.images_prefix, MaxKeys=1)
        result["list_ok"] = True

        # Optional write test
        if do_write:
            key = f"{cfg.images_prefix.rstrip('/')}/env_sanity_{int(time.time())}.txt"
            try:
                s3.put_object(Bucket=cfg.bucket, Key=key, Body=b"OK", ContentType="text/plain")
                result["write_ok"] = True
            except ClientError as e:
                result["write_ok"] = False
                result["write_error"] = str(e)
            finally:
                try:
                    s3.delete_object(Bucket=cfg.bucket, Key=key)
                    result["cleanup_ok"] = True
                except ClientError as e:
                    result["cleanup_ok"] = False
                    result["cleanup_error"] = str(e)

        result["ok"] = result["head_bucket_ok"] and result["list_ok"] and (
            True if result["write_ok"] is None else bool(result["write_ok"])  # if write requested, require success
        )
    except Exception as e:
        result["error"] = str(e)
    return result


def test_titan_model(cfg, do_invoke: bool = False) -> Dict[str, Any]:
    """Check Titan embedding model accessibility.

    Strategy:
    - Try to list foundation models (control plane) and verify the configured model ID is present.
    - If listing is denied or inconclusive, optionally attempt a minimal runtime invoke.
    """
    result: Dict[str, Any] = {
        "ok": False,
        "model_id": cfg.model_id,
        "listed": False,
        "invoked": None,
    }
    # Control-plane list
    try:
        session = _get_boto3_session(cfg.region, cfg.profile)
        bedrock = session.client("bedrock")
        resp = bedrock.list_foundation_models()
        summaries = resp.get("modelSummaries", [])
        # Prefer exact match; fallback to contains for minor ID variants
        exact = any(s.get("modelId") == cfg.model_id for s in summaries)
        contains = any(cfg.model_id in s.get("modelId", "") for s in summaries)
        result["listed"] = bool(exact or contains)
    except Exception as e:
        result["list_error"] = str(e)

    # Optional minimal runtime invoke (may incur small cost)
    if do_invoke:
        try:
            rt = get_bedrock_client(cfg.region, cfg.profile)
            payload = {
                "inputText": "env sanity",
                "embeddingConfig": {"outputEmbeddingLength": min(16, max(8, int(cfg.output_dim) if cfg.output_dim else 16))},
            }
            rt.invoke_model(
                modelId=cfg.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload),
            )
            result["invoked"] = True
        except Exception as e:
            result["invoked"] = False
            result["invoke_error"] = str(e)

    # Final decision: accessible if listed OR (invoked True when attempted)
    result["ok"] = result["listed"] or (result["invoked"] is True)
    return result


def test_qdrant(cfg) -> Dict[str, Any]:
    """Check Qdrant health by listing collections and validating the configured collection."""
    result: Dict[str, Any] = {
        "ok": False,
        "url": cfg.qdrant_url,
        "collection": cfg.qdrant_collection_name,
        "healthy": False,
        "collection_exists": None,
        "vector_size_ok": None,
    }
    try:
        client = get_qdrant_client(cfg.qdrant_url, cfg.qdrant_api_key, cfg.qdrant_timeout)
        # Basic connectivity: list collections
        client.get_collections()
        result["healthy"] = True
        # Validate collection config (exists + vector size)
        try:
            info = validate_collection_config(client, cfg.qdrant_collection_name, cfg.output_dim)
            result["collection_exists"] = True
            # Derive vector size from the returned info for reporting
            try:
                vectors_config = info.config.params.vectors
                if hasattr(vectors_config, "size") and vectors_config.size == cfg.output_dim:
                    result["vector_size_ok"] = True
                else:
                    # Named vectors or mismatch
                    result["vector_size_ok"] = True  # validate_collection_config would have raised on mismatch
            except Exception:
                result["vector_size_ok"] = True
        except Exception as ve:
            # Collection missing or mismatch
            result["collection_exists"] = False
            result["vector_size_ok"] = False
            result["collection_error"] = str(ve)

        result["ok"] = bool(result["healthy"])  # Health is primary; collection can be created later
    except Exception as e:
        result["error"] = str(e)
    return result


def print_human_report(aws_res: Dict[str, Any], s3_res: Dict[str, Any], titan_res: Dict[str, Any], qdrant_res: Dict[str, Any]) -> None:
    print(b("\nEnvironment Sanity Checks"))
    print("=" * 32)

    # 1) AWS Tokens
    print(f"\n1) AWS tokens (STS): {ok_mark(aws_res.get('ok', False))}")
    if aws_res.get("ok"):
        print(f"   Account: {aws_res.get('account')} | ARN: {aws_res.get('arn')} | Region: {aws_res.get('region')}")
    else:
        print(ctext(f"   Error: {aws_res.get('error', 'Unknown error')}", Colors.RED))

    # 2) S3
    print(f"\n2) S3 setup: {ok_mark(s3_res.get('ok', False))}")
    print(f"   Bucket: {s3_res.get('bucket')} | Prefix(images): {s3_res.get('images_prefix')}")
    details = [
        ("head_bucket", s3_res.get("head_bucket_ok", False)),
        ("list", s3_res.get("list_ok", False)),
    ]
    if s3_res.get("write_ok") is not None:
        details.append(("write", bool(s3_res.get("write_ok"))))
        details.append(("cleanup", bool(s3_res.get("cleanup_ok"))))
    print("   Checks: " + ", ".join([f"{name}={ok_mark(ok)}" for name, ok in details]))
    if not s3_res.get("ok") and s3_res.get("error"):
        print(ctext(f"   Error: {s3_res['error']}", Colors.RED))

    # 3) Titan embedding model
    print(f"\n3) Titan embedding model access: {ok_mark(titan_res.get('ok', False))}")
    print(f"   Model ID: {titan_res.get('model_id')}")
    print(
        "   Checks: "
        + ", ".join(
            [
                f"listed={ok_mark(bool(titan_res.get('listed')))}",
                (
                    f"invoked={ok_mark(bool(titan_res.get('invoked')))}"
                    if titan_res.get("invoked") is not None
                    else "invoked=skipped"
                ),
            ]
        )
    )
    if not titan_res.get("ok"):
        if titan_res.get("list_error"):
            print(ctext(f"   List error: {titan_res['list_error']}", Colors.YELLOW))
        if titan_res.get("invoke_error"):
            print(ctext(f"   Invoke error: {titan_res['invoke_error']}", Colors.YELLOW))

    # 4) Qdrant
    print(f"\n4) Qdrant accessibility: {ok_mark(qdrant_res.get('ok', False))}")
    print(f"   URL: {qdrant_res.get('url')} | Collection: {qdrant_res.get('collection')}")
    print(
        "   Checks: "
        + ", ".join(
            [
                f"healthy={ok_mark(bool(qdrant_res.get('healthy')))}",
                (
                    f"collection_exists={ok_mark(bool(qdrant_res.get('collection_exists')))}"
                    if qdrant_res.get("collection_exists") is not None
                    else "collection_exists=unknown"
                ),
                (
                    f"vector_size_ok={ok_mark(bool(qdrant_res.get('vector_size_ok')))}"
                    if qdrant_res.get("vector_size_ok") is not None
                    else "vector_size_ok=unknown"
                ),
            ]
        )
    )
    if not qdrant_res.get("ok") and qdrant_res.get("error"):
        print(ctext(f"   Error: {qdrant_res['error']}", Colors.RED))
    if qdrant_res.get("collection_error"):
        print(ctext(f"   Collection error: {qdrant_res['collection_error']}", Colors.YELLOW))

    # Summary
    all_ok = all([aws_res.get("ok"), s3_res.get("ok"), titan_res.get("ok"), qdrant_res.get("ok")])
    print(b("\nSummary:"), ok_mark(all_ok))


def main():
    load_dotenv()  # best-effort

    parser = argparse.ArgumentParser(description="Environment sanity checks (Python-only)")
    parser.add_argument("--write-s3", action="store_true", default=to_bool(os.getenv("SANITY_WRITE_TESTS"), False), help="Perform a tiny S3 write+delete to verify write permissions")
    parser.add_argument("--invoke-bedrock", action="store_true", default=to_bool(os.getenv("SANITY_BEDROCK_INVOKE"), False), help="Attempt a minimal Titan embedding invocation (may incur small cost)")
    parser.add_argument("--json", action="store_true", help="Output JSON summary")
    args = parser.parse_args()

    cfg = load_config()

    aws_res = test_aws_tokens(cfg)
    s3_res = test_s3(cfg, do_write=args.write_s3)
    titan_res = test_titan_model(cfg, do_invoke=args.invoke_bedrock)
    qdrant_res = test_qdrant(cfg)

    if args.json:
        print(
            json.dumps(
                {
                    "aws_tokens": aws_res,
                    "s3": s3_res,
                    "titan_model": titan_res,
                    "qdrant": qdrant_res,
                    "all_ok": all(
                        [aws_res.get("ok"), s3_res.get("ok"), titan_res.get("ok"), qdrant_res.get("ok")]
                    ),
                },
                indent=2,
            )
        )
    else:
        print_human_report(aws_res, s3_res, titan_res, qdrant_res)

    # Exit non-zero if any critical check failed
    all_ok = all([aws_res.get("ok"), s3_res.get("ok"), titan_res.get("ok"), qdrant_res.get("ok")])
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
