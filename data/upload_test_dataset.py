#!/usr/bin/env python3

"""
Test Dataset Batch Upload Script

This script reads the test_dataset.json and uploads all images in the data/test_images/ directory
to S3 and S3 Vectors using the same logic as the main app.

Usage:
    python data/upload_test_dataset.py [--dry-run]
"""

# Standard library imports
import argparse
import json
import os
import sys
import uuid
import time
import random
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Optional

# Third-party imports
from dotenv import load_dotenv
from botocore.exceptions import BotoCoreError, ClientError

# Add parent directory to path to import app modules before local imports
sys.path.append(str(Path(__file__).parent.parent))

# Local application imports
from app import (
    _s3v_merge_metadata,
    generate_image_description,  # uses Claude Vision and returns (display_text, embed_text)
    generate_mm_embedding,
    get_bedrock_client,
    get_s3_client,
    get_s3vectors_client,
    normalize_prefix,
    to_unix_ts,
    upload_bytes_to_s3,
)

# Load environment
load_dotenv()

# ----------------------------
# Rate limiting & backoff
# ----------------------------

def _get_float_env(possible_keys, default_val: float) -> float:
    for k in possible_keys:
        v = os.getenv(k)
        if v is not None and str(v).strip() != "":
            try:
                return float(v)
            except Exception:
                pass
    return float(default_val)

# Accept multiple casings and a common misspelling ("claud")
_CLAUDE_MIN_INTERVAL = _get_float_env([
    "BATCH_CLAUDE_MIN_INTERVAL_SEC",
    "batch_claude_mininterval_sec",
    "batch_claud_mininterval_sec",
], 1.5)

_TITAN_MIN_INTERVAL = _get_float_env([
    "BATCH_TITAN_MIN_INTERVAL_SEC",
    "batch_titan_mininterval_sec",
], 0.5)

_last_call_ts = {"claude": 0.0, "titan": 0.0}


def _rate_limit_wait(kind: str):
    now = time.perf_counter()
    minimum = _CLAUDE_MIN_INTERVAL if kind == "claude" else _TITAN_MIN_INTERVAL
    elapsed = now - _last_call_ts.get(kind, 0.0)
    if elapsed < minimum:
        time.sleep(minimum - elapsed)
    _last_call_ts[kind] = time.perf_counter()


def _call_with_backoff(fn, *, max_attempts: int = 6, base_delay: float = 1.0, jitter: float = 0.25):
    """Call fn() with exponential backoff on throttling/throughput errors."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except (ClientError, BotoCoreError) as e:
            code = None
            if isinstance(e, ClientError):
                code = (e.response.get("Error", {}).get("Code") if hasattr(e, "response") else None)
            msg = str(e)
            throttle = any(x in (code or msg) for x in [
                "ThrottlingException",
                "TooManyRequestsException",
                "ProvisionedThroughputExceededException",
                "Rate exceeded",
                "SlowDown",
            ])
            if not throttle or attempt == max_attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, jitter)
            print(f"    ⏳ Throttled (attempt {attempt}/{max_attempts}); retrying in {delay:.2f}s...")
            time.sleep(delay)

def load_test_dataset():
    """Load the test dataset configuration"""
    # Try complete dataset first, then simplified, then original
    for filename in ["test_dataset_complete.json", "test_dataset_simple.json", "test_dataset.json"]:
        dataset_path = Path(__file__).parent / filename
        if dataset_path.exists():
            with open(dataset_path, 'r') as f:
                return json.load(f)
    raise FileNotFoundError("No test dataset file found")

# Map folder day names to a date in the current ISO week
DAY_IDX = {
    "monday": 0, "tuesday": 1, "wednesday": 2,
    "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
}

def date_for_day_name(day_name: str) -> date:
    today = date.today()
    # Start of current ISO week (Monday=0)
    start = today - timedelta(days=today.weekday())
    idx = DAY_IDX.get(day_name.lower())
    if idx is None:
        return today
    return start + timedelta(days=idx)

def find_image_files(base_path: Path):
    """Find all image files in the test dataset directory (with day-based structure)"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff'}
    images = {}
    
    for user_dir in base_path.iterdir():
        if not user_dir.is_dir():
            continue
            
        user_id = user_dir.name
        images[user_id] = {}
        
        for day_dir in user_dir.iterdir():
            if not day_dir.is_dir():
                continue
                
            day = day_dir.name
            images[user_id][day] = {}
            
            for meal_dir in day_dir.iterdir():
                if not meal_dir.is_dir():
                    continue
                    
                meal_type = meal_dir.name
                meal_images = []
                
                for img_file in meal_dir.iterdir():
                    if img_file.suffix.lower() in image_extensions:
                        meal_images.append(img_file)
                
                images[user_id][day][meal_type] = meal_images
    
    return images

def upload_image(
    image_path: Path,
    user_id: str,
    day_name: Optional[str],
    meal_type: str,
    meal_data: dict,
    bedrock_client,
    s3_client,
    s3v_client,
    config: dict,
    dry_run: bool = False,
):
    """Upload a single image with its metadata"""
    
    print(f"Processing: {user_id}/{day_name or 'day'}/{meal_type}/{image_path.name}")
    
    if dry_run:
        print(f"  [DRY-RUN] Would upload {image_path}")
        return True
    
    try:
        # Read image bytes
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        # Prepare simplified meal data (no tags, no protein)
        simplified_meal_data = {
            'meal_type': meal_type,
            'notes': meal_data.get('notes', ''),
        }

        # Generate nutrition-focused JSON + compact text using the app's function
        print("  Generating image description (Claude Vision)...")
        _rate_limit_wait("claude")
        display_text, embed_text = _call_with_backoff(lambda: generate_image_description(
            image_bytes=image_bytes,
            meal_data=simplified_meal_data,
            bedrock_client=bedrock_client,
        ))
        print(f"    Generated (display): {display_text[:100]}...")

        # Generate multi-modal embedding (image + compact content tags)
        print("  Generating multi-modal embedding (Titan MM)...")
        _rate_limit_wait("titan")
        embedding = _call_with_backoff(lambda: generate_mm_embedding(
            bedrock_client=bedrock_client,
            model_id=config['model_id'],
            output_dim=config['output_dim'],
            input_text=embed_text,
            input_image_bytes=image_bytes
        ), base_delay=0.5)
        
        # Prepare keys
        image_id = str(uuid.uuid4())
        ext = image_path.suffix.lower()
        content_type = f"image/{ext[1:]}" if ext else "image/jpeg"
        
        image_key = f"{config['images_prefix']}{image_id}{ext}"
        vector_key = f"{config['embeddings_prefix']}{image_id}.json"
        
        # Upload image to S3
        print("  Uploading image to S3...")
        upload_bytes_to_s3(s3_client, config['bucket'], image_key, image_bytes, content_type)
        
        # Prepare metadata
        # Create meal datetime using folder day if available, otherwise today
        base_day = date_for_day_name(day_name) if day_name else date.today()
        meal_time_str = meal_data['meal_time']  # e.g., "07:30"
        hour, minute = map(int, meal_time_str.split(':'))
        meal_time = dtime(hour, minute)
        meal_dt = datetime.combine(base_day, meal_time)
        ts = to_unix_ts(meal_dt)
        
        # Build complete record for S3 storage (simplified)
        base_record = {
            "model_id": config['model_id'],
            "embedding_length": len(embedding),
            "s3_image_bucket": config['bucket'],
            "s3_image_key": image_key,
            "uploaded_filename": image_path.name,
            "content_type": content_type,
            "output_embedding_length": config['output_dim'],
            "region": config['region'],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            # Simplified user and meal metadata
            "user_id": user_id,
            "meal_type": meal_type,
            "meal_time": meal_dt.isoformat(),
            "ts": ts,
            "generated_description": display_text,
            "embedding": embedding
        }
        
        # Upload complete record to S3
        print("  Uploading metadata to S3...")
        upload_bytes_to_s3(
            s3_client, config['bucket'], vector_key, 
            json.dumps(base_record).encode('utf-8'), 
            content_type="application/json"
        )
        
        # Prepare simplified metadata for S3 Vectors (matching main app)
        searchable_metadata = {
            "user_id": user_id,
            "meal_type": meal_type,
            "ts": ts,
        }
        
        non_searchable_metadata = {
            "s3_image_key": image_key,
            "s3_embedding_key": vector_key,
            "s3_bucket": config['bucket'],
            "model_id": config['model_id'],
            "uploaded_filename": image_path.name,
            "content_type": content_type,
            "meal_time": meal_dt.isoformat(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Upload to S3 Vectors
        print("  Indexing in S3 Vectors...")
        put_kwargs = {
            "vectors": [{
                "key": image_id,
                "data": {"float32": [float(x) for x in embedding]},
                "metadata": _s3v_merge_metadata(searchable_metadata, non_searchable_metadata, limit=10)
            }]
        }
        
        if config['index_arn']:
            put_kwargs["indexArn"] = config['index_arn']
        else:
            put_kwargs["vectorBucketName"] = config['vector_bucket']
            put_kwargs["indexName"] = config['index_name']
        
        s3v_client.put_vectors(**put_kwargs)
        
        print(f"  ✅ Successfully uploaded {image_path.name}")
        print(f"     Image: s3://{config['bucket']}/{image_key}")
        print(f"     Metadata: s3://{config['bucket']}/{vector_key}")
        print(f"     Vector ID: {image_id}")
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to upload {image_path.name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Upload test dataset to S3 and S3 Vectors')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be uploaded without actually uploading')
    args = parser.parse_args()
    
    # Load configuration
    config = {
        'region': os.getenv('AWS_REGION', 'us-east-1'),
        'profile': os.getenv('AWS_PROFILE'),
        'bucket': os.getenv('APP_S3_BUCKET'),
        'images_prefix': normalize_prefix(os.getenv('APP_IMAGES_PREFIX', 'images/')),
        'embeddings_prefix': normalize_prefix(os.getenv('APP_EMBEDDINGS_PREFIX', 'embeddings/')),
        'model_id': os.getenv('MODEL_ID', 'amazon.titan-embed-image-v1'),
        'output_dim': int(os.getenv('OUTPUT_EMBEDDING_LENGTH', '1024')),
        'vector_bucket': os.getenv('S3V_VECTOR_BUCKET'),
        'index_name': os.getenv('S3V_INDEX_NAME'),
        'index_arn': os.getenv('S3V_INDEX_ARN')
    }
    
    if not config['bucket']:
        print("❌ Error: APP_S3_BUCKET not configured in .env file")
        return 1
    
    print(f"🚀 {'DRY RUN: ' if args.dry_run else ''}Test Dataset Upload")
    print(f"Target bucket: {config['bucket']}")
    print(f"Model: {config['model_id']}")
    print()
    print(f"Pacing: Claude every ~{_CLAUDE_MIN_INTERVAL}s, Titan every ~{_TITAN_MIN_INTERVAL}s\n")
    
    # Load dataset and find images
    dataset = load_test_dataset()
    images_base_path = Path(__file__).parent / "test_images"
    image_files = find_image_files(images_base_path)
    
    # Initialize AWS clients
    if not args.dry_run:
        bedrock_client = get_bedrock_client(config['region'], config['profile'])
        s3_client = get_s3_client(config['region'], config['profile'])
        s3v_client = get_s3vectors_client(config['region'], config['profile'])
    else:
        bedrock_client = s3_client = s3v_client = None
    
    # Upload images
    total_images = 0
    successful_uploads = 0
    
    for user_id, user_data in dataset['users'].items():
        print(f"\n👤 Processing user: {user_data['name']} ({user_id})")
        
        # Handle both old structure (user_data['meals']) and new structure (user_data['days'])
        if 'days' in user_data:
            # New day-based structure
            for day, day_data in user_data['days'].items():
                print(f"  📅 Processing {day}...")
                
                for meal_type, meal_data in day_data.items():
                    if (user_id in image_files and 
                        day in image_files[user_id] and 
                        meal_type in image_files[user_id][day]):
                        
                        meal_images = image_files[user_id][day][meal_type]
                        
                        if not meal_images:
                            print(f"    ⚠️ No images found for {user_id}/{day}/{meal_type}")
                            continue
                        
                        # Upload **all** images for this meal
                        for image_path in meal_images:
                            total_images += 1
                            success = upload_image(
                                image_path, user_id, day, meal_type, meal_data,
                                bedrock_client, s3_client, s3v_client, config, args.dry_run
                            )
                            if success:
                                successful_uploads += 1
                    else:
                        print(f"    ⚠️ No images found for {user_id}/{day}/{meal_type}")
        
        elif 'meals' in user_data:
            # Old simplified structure
            for meal_type, meal_data in user_data['meals'].items():
                if user_id in image_files and meal_type in image_files[user_id]:
                    meal_images = image_files[user_id][meal_type]
                    
                    if not meal_images:
                        print(f"  ⚠️ No images found for {user_id}/{meal_type}")
                        continue
                    
                    # Upload **all** images found for this meal
                    for image_path in meal_images:
                        total_images += 1
                        success = upload_image(
                            image_path, user_id, None, meal_type, meal_data,
                            bedrock_client, s3_client, s3v_client, config, args.dry_run
                        )
                        if success:
                            successful_uploads += 1
                else:
                    print(f"  ⚠️ No images found for {user_id}/{meal_type}")
    
    # Summary
    print("\n📊 Upload Summary:")
    print(f"Total images: {total_images}")
    print(f"Successful uploads: {successful_uploads}")
    print(f"Failed uploads: {total_images - successful_uploads}")
    
    if args.dry_run:
        print("\n💡 This was a dry run. Run without --dry-run to actually upload.")
    else:
        print("\n🎉 Upload complete! Your test dataset is ready for similarity search.")
    
    return 0 if successful_uploads == total_images else 1

if __name__ == "__main__":
    sys.exit(main())
