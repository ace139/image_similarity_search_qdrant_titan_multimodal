from __future__ import annotations

import os
import boto3
from typing import Optional
from botocore.exceptions import ProfileNotFound

DEFAULT_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"


def _get_boto3_session(region: Optional[str] = None, profile: Optional[str] = None):
    """Create a boto3 Session prioritizing environment credentials over profiles.

    Behavior:
    - If AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set in the environment, use them and ignore profile.
    - Otherwise, if a profile is provided, attempt to use it; if missing, fall back to default chain.
    - Finally, fall back to the default credential chain (env vars, shared config, role).
    """
    region = region or (os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or DEFAULT_REGION)

    aki = os.getenv("AWS_ACCESS_KEY_ID")
    sak = os.getenv("AWS_SECRET_ACCESS_KEY")
    sts = os.getenv("AWS_SESSION_TOKEN")

    if aki and sak:
        return boto3.Session(
            aws_access_key_id=aki,
            aws_secret_access_key=sak,
            aws_session_token=sts,
            region_name=region,
        )

    prof = profile if profile is not None else (os.getenv("AWS_PROFILE") or None)
    if prof:
        try:
            return boto3.Session(profile_name=prof, region_name=region)
        except ProfileNotFound:
            print(f"[WARN] AWS profile '{prof}' not found. Falling back to default credentials.")

    return boto3.Session(region_name=region)


def get_bedrock_client(region: Optional[str] = None, profile: Optional[str] = None):
    session = _get_boto3_session(region, profile)
    return session.client("bedrock-runtime")


def get_s3_client(region: Optional[str] = None, profile: Optional[str] = None):
    session = _get_boto3_session(region, profile)
    return session.client("s3")
