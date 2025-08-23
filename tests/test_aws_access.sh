#!/bin/bash

# AWS Access Verification Script
# This script tests access to AWS services used in the project

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== AWS Access Verification Script ===${NC}"
echo

# Load environment variables from .env file
if [ -f .env ]; then
    echo -e "${YELLOW}Loading environment variables from .env file...${NC}"
    export $(grep -v '^#' .env | xargs)
    echo -e "${GREEN}✓ Environment variables loaded${NC}"
else
    echo -e "${RED}✗ .env file not found${NC}"
    exit 1
fi

echo

# Test 1: Basic AWS CLI connectivity and credentials
echo -e "${BLUE}1. Testing AWS CLI connectivity and credentials...${NC}"
if aws sts get-caller-identity > /dev/null 2>&1; then
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    USER_ARN=$(aws sts get-caller-identity --query Arn --output text)
    echo -e "${GREEN}✓ AWS credentials are valid${NC}"
    echo -e "  Account ID: ${ACCOUNT_ID}"
    echo -e "  User/Role: ${USER_ARN}"
    echo -e "  Region: ${AWS_REGION}"
else
    echo -e "${RED}✗ Failed to authenticate with AWS${NC}"
    echo "Please check your AWS credentials"
    exit 1
fi

echo

# Test 2: S3 Bucket Access
echo -e "${BLUE}2. Testing S3 bucket access...${NC}"
BUCKET_NAME="${APP_S3_BUCKET:-food-plate-vectors}"

# Check if bucket exists and is accessible
if aws s3 ls "s3://${BUCKET_NAME}" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Can access S3 bucket: ${BUCKET_NAME}${NC}"
    
    # Test read access
    OBJECT_COUNT=$(aws s3 ls "s3://${BUCKET_NAME}" --recursive | wc -l)
    echo -e "  Objects in bucket: ${OBJECT_COUNT}"
    
    # Test write access (create a test file)
    echo "AWS CLI test" > /tmp/aws_test_file.txt
    if aws s3 cp /tmp/aws_test_file.txt "s3://${BUCKET_NAME}/test/aws_cli_test.txt" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Write access confirmed${NC}"
        # Clean up test file
        aws s3 rm "s3://${BUCKET_NAME}/test/aws_cli_test.txt" > /dev/null 2>&1
        rm /tmp/aws_test_file.txt
    else
        echo -e "${YELLOW}! Write access may be limited${NC}"
    fi
    
    # Check specific prefixes
    if [ -n "${APP_IMAGES_PREFIX}" ]; then
        IMAGES_COUNT=$(aws s3 ls "s3://${BUCKET_NAME}/${APP_IMAGES_PREFIX}" --recursive 2>/dev/null | wc -l || echo "0")
        echo -e "  Images in ${APP_IMAGES_PREFIX}: ${IMAGES_COUNT}"
    fi
    
    if [ -n "${APP_EMBEDDINGS_PREFIX}" ]; then
        EMBEDDINGS_COUNT=$(aws s3 ls "s3://${BUCKET_NAME}/${APP_EMBEDDINGS_PREFIX}" --recursive 2>/dev/null | wc -l || echo "0")
        echo -e "  Embeddings in ${APP_EMBEDDINGS_PREFIX}: ${EMBEDDINGS_COUNT}"
    fi
else
    echo -e "${RED}✗ Cannot access S3 bucket: ${BUCKET_NAME}${NC}"
    echo "Please check bucket permissions and name"
fi

echo

# Test 3: S3 Vectors Access  
echo -e "${BLUE}3. Testing S3 Vectors access...${NC}"
S3V_BUCKET="${S3V_VECTOR_BUCKET:-food-plate-vectors}"
S3V_INDEX="${S3V_INDEX_NAME:-test}"
S3V_ARN="${S3V_INDEX_ARN}"

# Check if S3 Vectors service is available by testing a basic operation
echo -e "  Checking S3 Vectors availability..."
if aws s3vectors 2>&1 | grep -q "operation"; then
    echo -e "${GREEN}✓ S3 Vectors CLI is available${NC}"
    
    # Try to check if the specific index exists by its ARN
    if [ -n "${S3V_ARN}" ]; then
        echo -e "  Testing index ARN: ${S3V_ARN}"
        # Since direct CLI commands may not be fully available, we test via S3 API
        # Check if the bucket has any vector configurations
        if aws s3api head-bucket --bucket "${S3V_BUCKET}" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Vector bucket is accessible${NC}"
            echo -e "  Vector bucket: ${S3V_BUCKET}"
            echo -e "  Index name: ${S3V_INDEX}"
            echo -e "${YELLOW}! Note: S3 Vectors CLI operations may require additional setup${NC}"
        else
            echo -e "${RED}✗ Cannot access vector bucket${NC}"
        fi
    else
        echo -e "${YELLOW}! S3V_INDEX_ARN not found in .env file${NC}"
    fi
else
    echo -e "${RED}✗ S3 Vectors CLI not available${NC}"
    echo "This might be because:"
    echo "  - S3 Vectors is not available in your region (${AWS_REGION})"
    echo "  - Your AWS CLI version doesn't support S3 Vectors yet"
    echo "  - The service requires additional setup"
    echo "  - You need special permissions for S3 Vectors"
fi

echo

# Test 4: Bedrock Model Access
echo -e "${BLUE}4. Testing Bedrock model access...${NC}"
MODEL="${MODEL_ID:-amazon.titan-embed-image-v1}"

# Check if Bedrock service is accessible
if aws bedrock list-foundation-models --region "${AWS_REGION}" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Bedrock service is accessible${NC}"
    
    # Check if our specific model is available
    if aws bedrock list-foundation-models --region "${AWS_REGION}" --query "modelSummaries[?modelId=='${MODEL}']" --output text | grep -q "${MODEL}"; then
        echo -e "${GREEN}✓ Model '${MODEL}' is available${NC}"
        
        # Get model details
        MODEL_STATUS=$(aws bedrock get-foundation-model --region "${AWS_REGION}" --model-identifier "${MODEL}" --query 'modelDetails.modelLifecycle.status' --output text 2>/dev/null || echo "UNKNOWN")
        echo -e "  Model status: ${MODEL_STATUS}"
        
        # Test if we can actually invoke the model (this tests real access)
        echo -e "  Testing model invocation..."
        TEST_PAYLOAD='{"inputText": "Hello world"}'
        if aws bedrock-runtime invoke-model --region "${AWS_REGION}" --model-id "${MODEL}" --body "${TEST_PAYLOAD}" --content-type "application/json" /tmp/bedrock_test_output.json > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Model invocation successful${NC}"
            rm -f /tmp/bedrock_test_output.json
        else
            echo -e "${YELLOW}! Model listed but invocation failed - may need access request${NC}"
        fi
    else
        echo -e "${RED}✗ Model '${MODEL}' not found or not accessible${NC}"
        echo "Available embedding models:"
        aws bedrock list-foundation-models --region "${AWS_REGION}" --query "modelSummaries[?contains(outputModalities, 'EMBEDDING')].modelId" --output text 2>/dev/null | head -5 || echo "  None found"
    fi
else
    echo -e "${RED}✗ Bedrock service not accessible${NC}"
    echo "This might be because:"
    echo "  - Bedrock is not available in your region (${AWS_REGION})"
    echo "  - You don't have the necessary permissions"
    echo "  - You need to request access to Bedrock models"
fi

echo

# Summary
echo -e "${BLUE}=== Summary ===${NC}"
echo -e "✓ = Working correctly"
echo -e "! = Working but with warnings"
echo -e "✗ = Not working or accessible"
echo
echo -e "${YELLOW}If you see any ✗ or ! symbols, please check:${NC}"
echo "1. Your AWS credentials and permissions"
echo "2. Service availability in your region (${AWS_REGION})"
echo "3. Bucket names and ARNs in your .env file"
echo "4. Bedrock model access requests if needed"
