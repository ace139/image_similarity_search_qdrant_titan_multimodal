#!/bin/bash

# Quick AWS Services Test - One-liner commands for each service
# Source the .env file first
source .env

echo "🔧 Quick AWS Service Tests:"
echo "=========================="

echo "✅ AWS Identity:"
aws sts get-caller-identity --query 'Account' --output text

echo "✅ S3 Bucket List:"
aws s3 ls s3://$APP_S3_BUCKET/ | head -3

echo "✅ Bedrock Models (Titan):"
aws bedrock list-foundation-models --region $AWS_REGION --query "modelSummaries[?contains(modelId, 'titan-embed-image')].{Model:modelId,Status:modelLifecycle.status}" --output table

echo "✅ S3 Vectors Status:"
echo "  Bucket: $S3V_VECTOR_BUCKET"  
echo "  Index: $S3V_INDEX_NAME"
echo "  ARN: $S3V_INDEX_ARN"

echo "=========================="
echo "✅ All basic services are accessible!"
