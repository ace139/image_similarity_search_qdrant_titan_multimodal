# Environment Sanity Checks (Python-only)

This folder contains a single test script that validates your environment and infrastructure connectivity using only environment variables.

- Script: `tests/env_sanity.py`
- Checks: AWS STS, S3 access, Bedrock model availability, Qdrant health/collection configuration
- Behavior: Strict environment variable usage. If any required variable is missing or invalid, the script exits with a clear message and exit code 2.

## What it checks

1) AWS tokens (STS)
- Verifies credentials by calling STS GetCallerIdentity
- Reports Account, ARN, and Region

2) S3 setup
- Heads the bucket and lists objects under your images prefix
- Optional: performs a tiny write+delete test when enabled

3) Titan embedding model access (Bedrock)
- Lists foundation models and verifies your configured model ID is present
- Optional: performs a minimal runtime invoke (may incur small cost)

4) Qdrant accessibility
- Connects to your Qdrant endpoint and lists collections (health check)
- Validates the configured collection exists and vector size matches OUTPUT_EMBEDDING_LENGTH

## Required environment variables
These are strict—no fallbacks are used. Set them in your `.env` file (see `.env.example`) or the shell environment.

- AWS_REGION
- APP_S3_BUCKET
- APP_IMAGES_PREFIX
- APP_EMBEDDINGS_PREFIX
- MODEL_ID
- OUTPUT_EMBEDDING_LENGTH (integer)
- QDRANT_URL
- QDRANT_COLLECTION_NAME
- QDRANT_TIMEOUT (integer)

Optional:
- AWS_PROFILE (if you prefer profile-based creds instead of access keys)
- QDRANT_API_KEY (if your Qdrant requires it)

Notes on credentials:
- The script creates AWS clients using a strict session policy: explicit environment credentials (AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY[/AWS_SESSION_TOKEN]) or an explicit AWS_PROFILE. It will not silently fall back to the default credential chain.

## Installation (Python deps)
Install the required Python packages if you haven't already:

```
pip install boto3 botocore qdrant-client python-dotenv
```

## Usage
Basic human-readable run:

```
python tests/env_sanity.py
```

JSON output (for automation/CI):

```
python tests/env_sanity.py --json
```

Optional checks:

- Include a tiny write+delete to S3 to verify write perms:
```
python tests/env_sanity.py --write-s3
```

- Attempt a minimal Titan invoke (may incur a small cost):
```
python tests/env_sanity.py --invoke-bedrock
```

You can also toggle options with environment variables:
- SANITY_WRITE_TESTS=1
- SANITY_BEDROCK_INVOKE=1

## Exit codes
- 0: All checks passed
- 1: One or more checks failed (connectivity/permissions, etc.)
- 2: Required environment variables missing or invalid

## Example JSON output
```
{
  "aws_tokens": { "ok": true, "account": "...", "arn": "...", "region": "..." },
  "s3": { "ok": true, "bucket": "...", "images_prefix": "...", "list_ok": true },
  "titan_model": { "ok": true, "model_id": "amazon.titan-embed-image-v1", "listed": true },
  "qdrant": { "ok": true, "url": "...", "collection": "...", "healthy": true, "vector_size_ok": true },
  "all_ok": true
}
```

## Troubleshooting
- Missing env vars (exit code 2):
  - Ensure your `.env` includes all required variables from `.env.example` and that numeric values are valid integers.
- S3 errors (head/list):
  - Confirm `APP_S3_BUCKET` exists and you have List/Get permissions. Verify prefixes (`APP_IMAGES_PREFIX`, `APP_EMBEDDINGS_PREFIX`).
- Bedrock model not listed or invocation denied:
  - Verify `MODEL_ID` and that your account/region has been granted access to the model in Bedrock.
- Qdrant connection refused or health fails:
  - Confirm `QDRANT_URL` and `QDRANT_API_KEY` (if needed). Ensure your Qdrant service is reachable and the collection exists.
- Vector size mismatch:
  - Ensure your Qdrant collection vector size matches `OUTPUT_EMBEDDING_LENGTH`.

---
This script is safe by default (read-only). Use `--write-s3` and/or `--invoke-bedrock` for deeper verification when you are ready.
