# AWS Access Verification Tests

This folder contains comprehensive tests to verify your AWS setup for the image similarity search application.

## 📁 Test Files Overview

### 1. `quick_aws_tests.sh` - Quick Status Check ⚡
**Purpose**: Fast verification of basic AWS service access  
**Duration**: ~5 seconds  
**Use when**: You want a quick health check

```bash
./tests/quick_aws_tests.sh
```

**What it tests:**
- AWS identity verification
- S3 bucket basic access
- Bedrock model availability
- S3 Vectors configuration display

---

### 2. `test_aws_access.sh` - Comprehensive CLI Tests 🔧
**Purpose**: Thorough AWS CLI-based verification  
**Duration**: ~30 seconds  
**Use when**: You want detailed CLI access verification

```bash
./tests/test_aws_access.sh
```

**What it tests:**
- AWS credentials validation
- S3 bucket read/write permissions
- S3 prefixes (`images/`, `embeddings/`)
- S3 Vectors CLI availability
- Bedrock service access and model invocation
- Comprehensive error reporting

---

### 3. `test_boto3_access.py` - Python SDK Verification 🐍
**Purpose**: Verify boto3 (Python SDK) access to all services  
**Duration**: ~15 seconds  
**Use when**: You want to test the exact SDK your app uses

```bash
python tests/test_boto3_access.py
```

**What it tests:**
- boto3 client initialization
- S3 operations via Python
- Bedrock model access via Python
- Image processing libraries (PIL, numpy)
- Full Python environment verification

---

### 4. `test_app_simulation.py` - Full Application Workflow 🖼️
**Purpose**: Complete end-to-end simulation of your app's workflow  
**Duration**: ~45 seconds  
**Use when**: You want to test the entire image similarity pipeline

```bash
python tests/test_app_simulation.py
```

**What it tests:**
- Image creation and processing
- S3 image upload
- Bedrock embedding generation (actual API calls)
- Embedding storage and retrieval
- Similarity calculations
- Vector collection management
- Complete cleanup

---

## 🚀 Quick Start

### Run All Tests in Order:
```bash
# 1. Quick health check
./tests/quick_aws_tests.sh

# 2. Comprehensive CLI verification
./tests/test_aws_access.sh

# 3. Python SDK verification
python tests/test_boto3_access.py

# 4. Full workflow simulation
python tests/test_app_simulation.py
```

### Run Only What You Need:
```bash
# Just check if everything is working
./tests/quick_aws_tests.sh

# Full end-to-end test
python tests/test_app_simulation.py
```

## 📋 Prerequisites

### Required Files:
- `.env` file in the root directory with AWS configuration
- All environment variables properly set

### Required Python Packages:
```bash
pip install -r requirements.txt
```
- boto3
- python-dotenv
- Pillow
- numpy (for similarity calculations)

### AWS Permissions Required:
- **S3**: ListBucket, GetObject, PutObject, DeleteObject
- **Bedrock**: ListFoundationModels, InvokeModel
- **STS**: GetCallerIdentity

## 🎯 Expected Results

### ✅ Success Indicators:
- Green checkmarks (✓) for all major operations
- No red X marks (✗) in critical areas
- Successful embedding generation with 1024-dimensional vectors
- Similarity scores between 0.0 and 1.0

### ⚠️ Warning Indicators:
- Yellow exclamation marks (!) indicate non-critical issues
- These are usually informational and won't break your app

### ❌ Error Indicators:
- Red X marks (✗) indicate problems that need fixing
- Check AWS credentials, permissions, or service availability

## 🔧 Troubleshooting

### Common Issues:

1. **"AWS credentials not found"**
   - Check your `.env` file exists and has `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
   - Ensure credentials are not commented out

2. **"S3 bucket access denied"**
   - Verify bucket name in `.env` is correct
   - Check IAM permissions for S3 operations

3. **"Bedrock model access denied"**
   - Request access to the Titan models in AWS Bedrock console
   - Ensure your region supports Bedrock (us-east-1 recommended)

4. **"Python packages missing"**
   - Run: `pip install -r requirements.txt`

### Debug Mode:
For verbose output, you can modify the scripts to show detailed error messages.

## 📊 Test Results Interpretation

### `test_app_simulation.py` Results:
- **Similarity Scores**: Values close to 1.0 indicate high similarity
- **Embedding Length**: Should consistently be 1024 for Titan model
- **Performance**: Embedding generation typically takes 1-3 seconds per image

### Cost Information:
- `test_app_simulation.py` makes real API calls to Bedrock (~$0.01-0.02)
- Other tests are free (only metadata operations)

## 🏗️ Project Structure After Testing:
```
image_similarity_search_s3_vectors/
├── .env
├── requirements.txt
├── app.py (your main application)
└── tests/
    ├── README.md (this file)
    ├── quick_aws_tests.sh
    ├── test_aws_access.sh
    ├── test_boto3_access.py
    └── test_app_simulation.py
```

## 🎉 Success Criteria

Your AWS setup is ready when:
- ✅ All test files run without critical errors
- ✅ `test_app_simulation.py` completes the full workflow
- ✅ Embedding generation produces consistent 1024-dimensional vectors
- ✅ Similarity calculations return reasonable values (0.8-1.0 for similar images)

---

**Ready to launch your image similarity search application!** 🚀
