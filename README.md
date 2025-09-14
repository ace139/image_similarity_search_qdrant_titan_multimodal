# Image Similarity Search with Titan Multimodal, Qdrant — now with Bulk Ingest/Search

A simple Streamlit app to upload images, generate embeddings using Amazon Bedrock Titan Multimodal Embeddings, and store both the image and an embeddings JSON record in Amazon S3. It also inserts the embedding into a Qdrant vector database for similarity search.

## Prerequisites
- Python environment managed with `uv` (already set up).
- AWS credentials configured via environment variables (e.g., `AWS_ACCESS_KEY_ID`) or an AWS profile (by setting `AWS_PROFILE` in `.env`).
- Proper IAM permissions (see [IAM Permissions](#-iam-permissions) section below).
- Access to Amazon Bedrock in your chosen region and model access to:
  - `amazon.titan-embed-image-v1` (Titan Multimodal Embeddings)
  - Claude Sonnet via Bedrock Converse (set `CLAUDE_VISION_MODEL_ID` to an inference profile ID/ARN)
- A running Qdrant vector database (local Docker or managed/cloud). The app connects via `QDRANT_URL` and ensures the target collection and payload indexes.
- An S3 bucket where images and embedding JSON artifacts will be stored.

## Install dependencies
```bash
uv pip install -r requirements.txt
```

## Configuration (.env)

Create a `.env` file in the project root with your settings. The app loads all configuration from environment variables (no sidebar inputs):

```dotenv
# AWS
AWS_REGION=us-east-1                 # or AWS_DEFAULT_REGION
# Either provide static credentials (preferred for CI) or use AWS_PROFILE
# AWS_ACCESS_KEY_ID=...
# AWS_SECRET_ACCESS_KEY=...
AWS_PROFILE=default                  # optional, ignored if static credentials are set

# S3 Storage for images and embeddings JSON
APP_S3_BUCKET=your-bucket
APP_IMAGES_PREFIX=images/
APP_EMBEDDINGS_PREFIX=embeddings/

# Bedrock models
MODEL_ID=amazon.titan-embed-image-v1 # Titan Multimodal Embeddings
OUTPUT_EMBEDDING_LENGTH=1024         # 256 | 384 | 1024

# Claude Vision via Bedrock Converse requires an inference profile ID/ARN
CLAUDE_VISION_MODEL_ID=us.anthropic.claude-3-5-sonnet-20241022-v2:0

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=food_embeddings            # individual (standard) collection
QDRANT_COLLECTION_NAME_BULK=food_embeddings_bulk  # dedicated bulk collection
QDRANT_TIMEOUT=60

# Bulk identity used in payloads for bulk uploads/search
BULK_USER_ID=999999

# Optional
APP_DEBUG=false
```

- `CLAUDE_VISION_MODEL_ID` must be an inference profile ID/ARN (e.g., `us.anthropic.claude-3-5-sonnet-20241022-v2:0`).

## 🧪 Testing Your Environment

This project includes a Python-based sanity check script to validate your environment and infrastructure connectivity before running the main application. It uses only environment variables, mirroring the application's configuration behavior.

- **Script**: `tests/env_sanity.py`
- **Checks**: AWS STS, S3 access, Bedrock model availability, Qdrant health/collection configuration.
- **Behavior**: Strict environment variable usage. If any required variable is missing or invalid, the script exits with a clear message and exit code 2.

### What it Checks
1.  **AWS tokens (STS)**: Verifies credentials by calling STS `GetCallerIdentity`.
2.  **S3 setup**: Checks bucket access and optionally performs a write/delete test.
3.  **Titan embedding model access (Bedrock)**: Verifies your configured model is available and optionally performs a minimal invocation.
4.  **Qdrant accessibility**: Connects to your Qdrant endpoint, checks health, and validates the collection configuration.

### Prerequisites
The script uses the same dependencies as the main application, which should already be installed from `requirements.txt`.

Ensure you have set the required environment variables in your `.env` file (see `.env.example`). The script will fail if they are missing.

### Usage

**Basic (read-only) check:**
```bash
python tests/env_sanity.py
```

**With optional write/invoke checks:**
```bash
# Verify S3 write permissions
python tests/env_sanity.py --write-s3

# Verify Bedrock model invocation (may incur small cost)
python tests/env_sanity.py --invoke-bedrock
```

**JSON output for automation:**
```bash
python tests/env_sanity.py --json
```

### Understanding Test Results
The script provides a clear, color-coded output indicating the status of each check (✓ for success, ✗ for failure). It exits with code 0 on success, 1 on failure, and 2 for configuration errors. For more details, see `tests/README.md`.

**💡 Pro tip: Run `python tests/env_sanity.py` before launching your app to ensure everything works perfectly!**

## Module Structure

```text
app.py                         # Main Streamlit application entry point
mmfood/
  config.py                    # Centralized environment config loader (AppConfig)
  aws/
    session.py                 # Boto3 client/session helpers (S3, Bedrock)
    s3.py                      # S3 helpers (presign, uploads, key utils)
  bedrock/
    ai.py                      # Titan MM embedding + Claude Vision description
  database/
    metrics.py                 # Metrics storage and retrieval logic
  qdrant/
    client.py                  # Qdrant client helpers, ensure/validate collection
    operations.py              # Upsert/search helpers for individuals and bulk
  services/
    ingest_service.py          # Core logic for the ingestion workflow
    search_service.py          # Core logic for the search workflow
  ui/
    components.py              # Shared Streamlit UI components
    ingest.py                  # UI tabs for single and bulk ingestion
    search.py                  # UI tabs for single and bulk search
    metrics.py                 # UI tab for metrics display
  utils/
    crypto.py                  # MD5 hashing utility
    time.py                    # Timestamp utility
```

## Run the app
```bash
# If not already activated
source .venv/bin/activate

streamlit run app.py
```
The app runs at http://localhost:8501

## App structure (tabs)
- Ingest: single-image ingestion (describe → embed → S3 uploads → Qdrant upsert)
- Search: text/image search in the standard collection (scoped to the individual collection)
- 📊 Metrics: standard metrics scoped to the individual collection only (bulk excluded)
- Bulk Ingest: multi-file ingest that batches the Qdrant upserts in one shot
- Bulk Search: text/image search in the bulk collection
- Bulk 📊 Metrics: bulk-only metrics scoped to the bulk collection

## How to use
1. Prepare `.env` as shown above and ensure required IAM permissions.
2. For single ingestion: open the Ingest tab
   - Upload a food image
   - Enter Metadata → User ID (optional), meal date/time, meal type
   - Click "🚀 Ingest Image"
3. For bulk ingestion: open the Bulk Ingest tab
   - Select multiple images
   - Optional: toggle "Skip description (faster)"
   - Click "Start Bulk Ingest"
   - The app uploads images + JSON to S3 and then performs a single-shot Qdrant upsert for all points
4. For search:
   - Use Search for the individual collection or Bulk Search for the bulk collection
   - Choose query type: Text or Image
   - On Search (standard), filters are available (user/date/meal)
   - On Bulk Search, filters are hidden by design
5. Metrics:
   - 📊 Metrics shows standard (individual-only) metrics — bulk activity is excluded
   - Bulk 📊 Metrics shows only bulk activity

## S3 Object Layout
- Images: `<images_prefix>/<uuid>.<ext>`
- Embeddings JSON: `<embeddings_prefix>/<uuid>.json`

## Qdrant behavior
- One vector per image, vector id = image UUID
- Standard (individual) collection: `QDRANT_COLLECTION_NAME`
- Bulk collection: `QDRANT_COLLECTION_NAME_BULK`
- Payload fields (standard): `user_id`, `meal_type`, `ts`, S3 keys/metadata, `generated_description`, lengths, region, etc.
- Payload fields (bulk): same shape, but `user_id` is set to `BULK_USER_ID` (default `999999`) for reuse/parity. Standard metrics exclude this identity automatically.
- Payload indexes ensured for filtering: `user_id`, `meal_type`, `ts` (no filters are used in Bulk Search UI)

### Batch upsert (bulk)
- Bulk Ingest collects `PointStruct` records (id, vector, payload) for all files and performs a single `client.upsert(points=[...], wait=False)`
- For very large batches, the client auto-chunks requests (e.g., 512 points per call)
- The UI displays both average and total timings per stage (description, embedding, S3 uploads, Qdrant upsert)

### Querying
- Search tab: generates a query embedding (text or image) and searches the individual collection with optional metadata filters
- Bulk Search tab: searches the bulk collection without filters

### Qdrant setup
- Local: `docker run -p 6333:6333 qdrant/qdrant:latest`
- Cloud: point `QDRANT_URL` to your managed Qdrant endpoint and set `QDRANT_API_KEY` if required
- The app ensures your target collection exists and validates vector size against `OUTPUT_EMBEDDING_LENGTH`
  "s3_image_key": "images/123e4567-e89b-12d3-a456-426614174000.png",
  "uploaded_filename": "example.png",
  "content_type": "image/png",
  "output_embedding_length": 1024,
  "region": "us-east-1",
  "timestamp": "2025-08-09T07:00:00Z"
}
```

## 📊 Metrics

### Standard Metrics (individual-only)
- Search Performance Summary (Total Requests, Success Rate, Avg Total, Avg Results)
- Performance Breakdown (Avg Embedding, Avg Search)
- Query Types
- Search Quality KPIs: Avg Top-1 Score, Avg TopK Avg, Avg Score Min/Max
- Latest Searches (10)
- Performance Range
- Top Users (excludes the bulk identity automatically)
- Note: standard metrics exclude bulk searches and ingests.

### Bulk Metrics (bulk-only)
- Mirrors the structure above but scoped to the bulk collection
- Latest Bulk Searches (10) and Latest Bulk Ingest Runs
- In Bulk Ingest UI, we show both Averages and Totals for the run

## 🔐 IAM Permissions

This project requires specific AWS IAM permissions to function correctly. You'll need to attach the following policies to your IAM user or role.

### Required AWS Managed Policies

#### 1. Amazon Bedrock Access
```
AmazonBedrockFullAccess
```
**Purpose**: Allows access to Amazon Bedrock models for generating image embeddings and text descriptions.

#### 2. Amazon S3 Access
```
AmazonS3FullAccess
```
**Purpose**: Provides read/write access to S3 buckets for storing images and embedding metadata.

⚠️ **Security Note**: For production environments, consider using more restrictive S3 policies that limit access to specific buckets only.

### Required Custom Inline Policy

#### 3. S3 Vectors Custom Policy
Create a custom inline policy with the following JSON:

```json
{
	"Version": "2012-10-17",
	"Statement": [
		{
			"Sid": "AllS3VectorsActionsAccountWideUSE1",
			"Effect": "Allow",
			"Action": "s3vectors:*",
			"Resource": "arn:aws:s3vectors:us-east-1:995133654003:bucket/*/index/*"
		},
		{
			"Sid": "S3RWForProjectBucket",
			"Effect": "Allow",
			"Action": [
				"s3:ListBucket",
				"s3:GetBucketLocation",
				"s3:GetObject",
				"s3:PutObject",
				"s3:DeleteObject"
			],
			"Resource": [
				"arn:aws:s3:::food-plate-vectors",
				"arn:aws:s3:::food-plate-vectors/*"
			]
		}
	]
}
```

**Purpose**:
- **S3 Vectors Operations**: Grants full access to S3 Vectors service for embedding indexing and similarity search
- **Project S3 Bucket**: Provides specific read/write access to the `food-plate-vectors` bucket

### How to Apply IAM Policies

1. **Via AWS Console**:
   - Go to IAM → Users/Roles → Select your user/role
   - Click "Add permissions" → "Attach policies directly"
   - Search and attach `AmazonBedrockFullAccess` and `AmazonS3FullAccess`
   - Click "Add permissions" → "Create inline policy"
   - Use JSON editor to paste the S3 Vectors custom policy above

2. **Via AWS CLI**:
   ```bash
   # Attach managed policies
   aws iam attach-user-policy --user-name YOUR_USERNAME --policy-arn arn:aws:iam::aws:policy/AmazonBedrockFullAccess
   aws iam attach-user-policy --user-name YOUR_USERNAME --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

   # Create and attach custom inline policy (save JSON above as s3-vectors-policy.json)
   aws iam put-user-policy --user-name YOUR_USERNAME --policy-name S3VectorsCustomPolicy --policy-document file://s3-vectors-policy.json
   ```

### Policy Customization

**🔧 For Your Environment**: Update the following in the custom policy:
- Replace `995133654003` with your AWS Account ID
- Replace `food-plate-vectors` with your actual S3 bucket name
- Adjust the region (`us-east-1`) if using a different region

**🛡️ Security Best Practices**:
- Use least-privilege access in production
- Consider creating dedicated IAM roles for this application
- Regularly review and audit permissions
- Use AWS IAM Access Analyzer to validate policies

### Verification

After applying these permissions, run the environment sanity check to verify everything works. This script will test your credentials, S3 access, and Bedrock model availability.
```bash
# Run the sanity check
python tests/env_sanity.py

# For a deeper check, verify S3 write permissions and Bedrock model invocation
python tests/env_sanity.py --write-s3 --invoke-bedrock
```

## Notes
- Ensure Bedrock and the Titan Multimodal Embeddings model are enabled in the selected region.
- Default model: `amazon.titan-embed-image-v1`.
- Output embedding lengths: 256, 384, or 1024.
- The environment sanity check (`tests/env_sanity.py`) will validate that your setup is working correctly.

## Troubleshooting

### Quick Debugging
1. **Run the sanity check first**: `python tests/env_sanity.py` to identify configuration issues.
2. **Check the detailed guide**: See `tests/README.md` for comprehensive troubleshooting steps.

### Common Issues
- **AWS Errors**: Verify credentials (env vars or profile) and region settings
- **Bedrock Access**: Ensure model access is enabled in your chosen region
- **S3 Permissions**: Confirm bucket exists and you have write permissions
- **Network Issues**: Check for proxy restrictions affecting AWS service connections

### Test Requirements
- `.env` file with AWS configuration
- Python packages: `pip install -r requirements.txt`
- AWS CLI installed and configured
