# Image Similarity Search with Titan Multimodal, Qdrant — now with Bulk Ingest/Search

A simple Streamlit app to upload images, generate embeddings using Amazon Bedrock Titan Multimodal Embeddings, and store both the image and an embeddings JSON record in Amazon S3. It also inserts the embedding into a Qdrant vector database for similarity search.

## Prerequisites
- Python environment managed with `uv` (already set up).
- AWS credentials configured (environment variables or `~/.aws/credentials`).
- Proper IAM permissions (see [IAM Permissions](#-iam-permissions) section below).
- Access to Amazon Bedrock in your chosen region and model access to `amazon.titan-embed-image-v1`.
- An existing S3 bucket where images and embeddings will be stored.
- An existing S3 Vectors bucket and index for similarity search.

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

Notes:
- For S3 Vectors, `S3V_INDEX_ARN` takes precedence. If not set, both `S3V_VECTOR_BUCKET` and `S3V_INDEX_NAME` are required.
- `CLAUDE_VISION_MODEL_ID` must be an inference profile ID/ARN (e.g., `us.anthropic.claude-3-5-sonnet-20241022-v2:0`).

## 🧪 Testing Your AWS Setup

This project includes comprehensive tests to verify your AWS setup before running the main application. It's recommended to run these tests first to ensure everything works correctly.

### Quick Health Check
```bash
# Fast health check (5 seconds)
./tests/run_tests.sh quick
```

### Full End-to-End Verification
```bash
# Complete workflow test (45 seconds)
./tests/run_tests.sh simulation
```

## Module Structure

```text
app.py                         # Streamlit UI and workflow orchestration (Ingest/Search)
mmfood/
  config.py                    # Centralized environment config loader (AppConfig)
  aws/
    session.py                 # boto3 client/session helpers (S3, Bedrock, S3 Vectors)
    s3.py                      # S3 helpers (presign, uploads, key utils)
  bedrock/
    ai.py                      # Titan MM embedding + Claude Vision description
  s3vectors/
    utils.py                   # S3 Vectors metadata merge and orphan cleanup
  utils/
    time.py                    # to_unix_ts
    crypto.py                  # md5_hex
```

### Run All Tests
```bash
# Comprehensive test suite (95 seconds)
./tests/run_tests.sh all
```

### Individual Test Categories
```bash
./tests/run_tests.sh cli       # CLI verification
./tests/run_tests.sh boto3     # Python SDK tests
./tests/run_tests.sh help      # Show all options
```

### Understanding Test Results
- ✅ **Success**: All services working correctly
- ⚠️ **Warning**: Non-critical issues (app will still work)
- ❌ **Error**: Critical problems that need fixing

### What Gets Tested
- ✅ AWS credentials and permissions
- ✅ S3 bucket read/write access
- ✅ Bedrock model availability and invocation
- ✅ Image processing and embeddings generation
- ✅ End-to-end similarity search workflow

**💡 Pro tip: Run `./tests/run_tests.sh quick` before launching your app to ensure everything works perfectly!**

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
- Bulk Ingest collects `PointStruct` records (id, vector, payload) for all files and performs a sin### Embeddings JSON schema (example)
```json
{
  "model_id": "amazon.titan-embed-image-v1",
  "embedding_length": 1024,
  "embedding": [0.01, -0.02, ...],
  "s3_image_bucket": "your-bucket",
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

After applying these permissions, run the test suite to verify everything works:
```bash
# Quick verification of all permissions
./tests/run_tests.sh quick

# Full end-to-end test including S3 Vectors
./tests/run_tests.sh simulation
```

## Notes
- Ensure Bedrock and the Titan Multimodal Embeddings model are enabled in the selected region.
- Default model: `amazon.titan-embed-image-v1`.
- Output embedding lengths: 256, 384, or 1024.
- The test suite will validate all required permissions are working correctly.

## Troubleshooting

### Quick Debugging
1. **Run the test suite first**: `./tests/run_tests.sh quick` to identify configuration issues
2. **Check the detailed guide**: See `tests/README.md` for comprehensive troubleshooting steps

### Common Issues
- **AWS Errors**: Verify credentials (env vars or profile) and region settings
- **Bedrock Access**: Ensure model access is enabled in your chosen region
- **S3 Permissions**: Confirm bucket exists and you have write permissions
- **Network Issues**: Check for proxy restrictions affecting AWS service connections

### Test Requirements
- `.env` file with AWS configuration
- Python packages: `pip install -r requirements.txt`
- AWS CLI installed and configured
