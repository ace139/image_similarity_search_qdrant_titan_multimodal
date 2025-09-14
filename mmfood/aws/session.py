from __future__ import annotations

import os
import boto3
from typing import Optional
from botocore.exceptions import ProfileNotFound

DEFAULT_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"


def _get_boto3_session(region: Optional[str] = None, profile: Optional[str] = None):
    """Create a boto3 Session using ONLY explicit credentials from the environment.

    Policy (env-first and env-only unless AWS_PROFILE is explicitly set in env):
    - If any of AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/AWS_SESSION_TOKEN are present in env,
      require at least AKI+SAK, and if AKI starts with ASIA then also require STS. Do NOT fall back
      to profiles or the default chain in this case.
    - Else if AWS_PROFILE is set (either via argument or env), use that exact profile.
    - Else raise a clear error (do not use default credential chain).
    """
    region = region or (os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or DEFAULT_REGION)

    aki = os.getenv("AWS_ACCESS_KEY_ID")
    sak = os.getenv("AWS_SECRET_ACCESS_KEY")
    sts = os.getenv("AWS_SESSION_TOKEN")

    # 1) Explicit static/temporary keys from environment take absolute precedence
    if any([aki, sak, sts]):
        if not (aki and sak):
            raise ValueError(
                "AWS environment variables detected but incomplete. "
                "Set both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env (and AWS_SESSION_TOKEN for temporary keys)."
            )
        if str(aki).startswith("ASIA") and not sts:
            raise ValueError(
                "Temporary AWS credentials detected (ASIA...). Missing AWS_SESSION_TOKEN in .env."
            )
        return boto3.Session(
            aws_access_key_id=aki,
            aws_secret_access_key=sak,
            aws_session_token=sts,
            region_name=region,
        )

    # 2) Otherwise, allow an explicitly-set profile (argument or env)
    prof = profile if profile is not None else (os.getenv("AWS_PROFILE") or None)
    if prof:
        try:
            return boto3.Session(profile_name=prof, region_name=region)
        except ProfileNotFound:
            raise ValueError(
                f"AWS profile '{prof}' not found. Update AWS_PROFILE in .env or provide static credentials."
            )

    # 3) No explicit env credentials or profile: do not silently fall back
    raise ValueError(
        "No explicit AWS credentials found. Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY (and AWS_SESSION_TOKEN for temporary keys) "
        "or AWS_PROFILE in your .env."
    )


def get_bedrock_client(region: Optional[str] = None, profile: Optional[str] = None):
    session = _get_boto3_session(region, profile)
    return session.client("bedrock-runtime")


def get_s3_client(region: Optional[str] = None, profile: Optional[str] = None):
    session = _get_boto3_session(region, profile)
    return session.client("s3")
