#!/usr/bin/env python3
"""
Utility script to create payload field indexes for existing Qdrant collections.

This script is useful if you already have data in your Qdrant collection
and are getting "Index required but not found" errors when filtering.

Usage:
    python create_qdrant_indexes.py
"""

import os
from dotenv import load_dotenv
from mmfood.config import load_config
from mmfood.qdrant.client import get_qdrant_client, ensure_payload_indexes

def main():
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    try:
        cfg = load_config()
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        print("Please check your .env file and ensure all required variables are set.")
        return 1
    
    # Connect to Qdrant
    try:
        qdrant = get_qdrant_client(
            url=cfg.qdrant_url,
            api_key=cfg.qdrant_api_key,
            timeout=cfg.qdrant_timeout
        )
        print(f"✅ Connected to Qdrant at {cfg.qdrant_url}")
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}")
        return 1
    
    # Check if collection exists
    try:
        collection_info = qdrant.get_collection(cfg.qdrant_collection_name)
        print(f"✅ Collection '{cfg.qdrant_collection_name}' found")
        
        # Get collection stats
        if hasattr(collection_info, 'points_count'):
            print(f"📊 Collection has {collection_info.points_count} points")
    except Exception as e:
        print(f"❌ Collection '{cfg.qdrant_collection_name}' not found: {e}")
        print("Please ensure your collection exists before creating indexes.")
        return 1
    
    # Create indexes for the fields used in filtering
    print("\n🔧 Creating payload field indexes...")
    try:
        ensure_payload_indexes(
            qdrant, 
            cfg.qdrant_collection_name, 
            ["user_id", "meal_type", "ts"]
        )
        print("✅ All indexes created successfully!")
        print("\n📝 Created indexes for:")
        print("  • user_id (keyword) - for user-specific filtering")
        print("  • meal_type (keyword) - for meal type filtering")
        print("  • ts (integer) - for date range filtering")
        print("\n🎉 You should now be able to use filters in your searches without errors.")
        
    except Exception as e:
        print(f"❌ Failed to create indexes: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())