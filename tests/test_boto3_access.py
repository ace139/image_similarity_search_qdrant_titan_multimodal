#!/usr/bin/env python3

import boto3
import json
import os
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
from dotenv import load_dotenv
import base64

# Colors for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BLUE}{Colors.BOLD}=== {text} ==={Colors.END}")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}! {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_info(text):
    print(f"  {text}")

def main():
    print_header("Boto3 AWS Access Verification")
    
    # Load environment variables
    if not os.path.exists('.env'):
        print_error(".env file not found")
        return False
        
    load_dotenv()
    print_success("Environment variables loaded from .env")
    
    # Get configuration from environment
    aws_region = os.getenv('AWS_REGION', 'us-east-1')
    bucket_name = os.getenv('APP_S3_BUCKET', 'food-plate-vectors')
    s3v_bucket = os.getenv('S3V_VECTOR_BUCKET', 'food-plate-vectors')
    s3v_index = os.getenv('S3V_INDEX_NAME', 'test')
    s3v_arn = os.getenv('S3V_INDEX_ARN', '')
    model_id = os.getenv('MODEL_ID', 'amazon.titan-embed-image-v1')
    images_prefix = os.getenv('APP_IMAGES_PREFIX', 'images/')
    embeddings_prefix = os.getenv('APP_EMBEDDINGS_PREFIX', 'embeddings/')
    
    all_tests_passed = True
    
    # Test 1: AWS Credentials and STS
    print_header("1. Testing AWS Credentials (STS)")
    try:
        sts_client = boto3.client('sts', region_name=aws_region)
        identity = sts_client.get_caller_identity()
        
        print_success("AWS credentials are valid")
        print_info(f"Account ID: {identity['Account']}")
        print_info(f"User/Role: {identity['Arn']}")
        print_info(f"Region: {aws_region}")
        
    except (NoCredentialsError, PartialCredentialsError) as e:
        print_error(f"AWS credentials not found or incomplete: {e}")
        all_tests_passed = False
        return False
    except ClientError as e:
        print_error(f"Failed to validate AWS credentials: {e}")
        all_tests_passed = False
        return False
    
    # Test 2: S3 Access
    print_header("2. Testing S3 Access")
    try:
        s3_client = boto3.client('s3', region_name=aws_region)
        s3_resource = boto3.resource('s3', region_name=aws_region)
        
        # Test bucket access
        s3_client.head_bucket(Bucket=bucket_name)
        print_success(f"Can access S3 bucket: {bucket_name}")
        
        # List objects in bucket
        response = s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=10)
        object_count = response.get('KeyCount', 0)
        print_info(f"Objects in bucket: {object_count}")
        
        # Test write access
        test_key = 'test/boto3_test_file.txt'
        test_content = 'Boto3 access test'
        
        try:
            s3_client.put_object(Bucket=bucket_name, Key=test_key, Body=test_content)
            print_success("Write access confirmed")
            
            # Clean up test file
            s3_client.delete_object(Bucket=bucket_name, Key=test_key)
            print_info("Test file cleaned up")
        except ClientError as e:
            print_warning(f"Write access may be limited: {e}")
        
        # Check specific prefixes
        for prefix_name, prefix_path in [("Images", images_prefix), ("Embeddings", embeddings_prefix)]:
            try:
                response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix_path, MaxKeys=10)
                count = response.get('KeyCount', 0)
                print_info(f"{prefix_name} in {prefix_path}: {count} objects")
            except ClientError as e:
                print_warning(f"Could not check {prefix_name} prefix: {e}")
                
    except ClientError as e:
        print_error(f"S3 access failed: {e}")
        all_tests_passed = False
    
    # Test 3: Bedrock Access
    print_header("3. Testing Amazon Bedrock Access")
    try:
        bedrock_client = boto3.client('bedrock', region_name=aws_region)
        bedrock_runtime = boto3.client('bedrock-runtime', region_name=aws_region)
        
        print_success("Bedrock client initialized successfully")
        
        # List foundation models
        try:
            models_response = bedrock_client.list_foundation_models()
            print_success("Can list foundation models")
            
            # Check if our specific model is available
            titan_models = [model for model in models_response['modelSummaries'] 
                          if model_id in model['modelId']]
            
            if titan_models:
                model = titan_models[0]
                print_success(f"Model '{model_id}' is available")
                print_info(f"Model name: {model['modelName']}")
                print_info(f"Status: {model['modelLifecycle']['status']}")
                print_info(f"Input modalities: {', '.join(model.get('inputModalities', []))}")
                print_info(f"Output modalities: {', '.join(model.get('outputModalities', []))}")
                
                # Test model invocation
                print_info("Testing model invocation...")
                try:
                    # Test with a simple text input for the multimodal model
                    test_payload = {
                        "inputText": "Hello world test"
                    }
                    
                    response = bedrock_runtime.invoke_model(
                        modelId=model_id,
                        contentType='application/json',
                        body=json.dumps(test_payload)
                    )
                    
                    result = json.loads(response['body'].read())
                    print_success("Model invocation successful!")
                    print_info(f"Response embedding length: {len(result.get('embedding', []))}")
                    
                except ClientError as e:
                    error_code = e.response['Error']['Code']
                    if error_code == 'AccessDeniedException':
                        print_warning("Model listed but access denied - you may need to request access to this model")
                    else:
                        print_warning(f"Model invocation failed: {e}")
                        
            else:
                print_error(f"Model '{model_id}' not found")
                # Show available embedding models
                embedding_models = [model['modelId'] for model in models_response['modelSummaries']
                                  if 'EMBEDDING' in model.get('outputModalities', [])]
                print_info(f"Available embedding models: {embedding_models[:5]}")
                
        except ClientError as e:
            print_error(f"Failed to list foundation models: {e}")
            all_tests_passed = False
            
    except Exception as e:
        print_error(f"Bedrock initialization failed: {e}")
        all_tests_passed = False
    
    # Test 4: S3 with boto3 (for vector operations)
    print_header("4. Testing S3 for Vector Operations")
    try:
        # Test if we can access the vector bucket (same as regular S3 but different usage)
        s3_client.head_bucket(Bucket=s3v_bucket)
        print_success(f"Vector bucket '{s3v_bucket}' is accessible via boto3")
        print_info(f"Vector index name: {s3v_index}")
        print_info(f"Vector index ARN: {s3v_arn}")
        
        # Note: S3 Vectors operations through boto3 may require special SDK or API calls
        # For now, we confirm the bucket is accessible
        print_warning("Note: S3 Vectors operations may require specialized boto3 calls or AWS SDK extensions")
        
    except ClientError as e:
        print_error(f"Vector bucket access failed: {e}")
        all_tests_passed = False
    
    # Test 5: Test loading a simple image (simulate app functionality)
    print_header("5. Testing Image Processing Simulation")
    try:
        import io
        from PIL import Image
        import numpy as np
        
        # Create a simple test image
        test_image = Image.new('RGB', (100, 100), color='red')
        img_buffer = io.BytesIO()
        test_image.save(img_buffer, format='JPEG')
        img_bytes = img_buffer.getvalue()
        
        print_success("Test image created successfully")
        print_info(f"Image size: {len(img_bytes)} bytes")
        
        # Test encoding image for Bedrock (base64)
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        print_success("Image base64 encoding successful")
        print_info(f"Base64 length: {len(img_b64)} characters")
        
        # Simulate the multimodal embedding call (without actually calling due to cost)
        multimodal_payload = {
            "inputImage": img_b64,
            "embeddingConfig": {
                "outputEmbeddingLength": int(os.getenv('OUTPUT_EMBEDDING_LENGTH', 1024))
            }
        }
        print_success("Multimodal payload prepared successfully")
        print_info(f"Expected embedding length: {multimodal_payload['embeddingConfig']['outputEmbeddingLength']}")
        
    except ImportError as e:
        print_warning(f"Image processing libraries not available: {e}")
        print_info("You may need to install: pip install Pillow numpy")
    except Exception as e:
        print_error(f"Image processing simulation failed: {e}")
    
    # Summary
    print_header("Summary")
    if all_tests_passed:
        print_success("All critical AWS services are accessible via boto3!")
        print_info("Your application should work correctly with the current AWS setup")
    else:
        print_warning("Some tests failed - check the issues above")
        print_info("You may need to fix credentials or permissions before running your app")
    
    # Configuration summary
    print(f"\n{Colors.BLUE}Configuration Summary:{Colors.END}")
    print_info(f"AWS Region: {aws_region}")
    print_info(f"S3 Bucket: {bucket_name}")
    print_info(f"S3 Vector Bucket: {s3v_bucket}")
    print_info(f"Vector Index: {s3v_index}")
    print_info(f"Bedrock Model: {model_id}")
    print_info(f"Expected Embedding Length: {os.getenv('OUTPUT_EMBEDDING_LENGTH', '1024')}")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
