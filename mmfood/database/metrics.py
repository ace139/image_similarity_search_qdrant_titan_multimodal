"""
SQLite database module for logging RAG retrieval metrics.
"""
import sqlite3
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import contextmanager


class MetricsDatabase:
    """SQLite database handler for RAG metrics logging."""
    
    def __init__(self, db_path: str = "rag_metrics.db"):
        self.db_path = Path(db_path)
        self._init_database()
    
    def _init_database(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rag_requests (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT,
                    query_type TEXT,  -- 'text' or 'image'
                    query_text TEXT,
                    query_image_path TEXT,
                    filters JSON,
                    top_k INTEGER,
                    total_duration_ms REAL,
                    embedding_duration_ms REAL,
                    search_duration_ms REAL,
                    results_count INTEGER,
                    success BOOLEAN DEFAULT 1,
                    error_message TEXT,
                    session_id TEXT,
                    client_info JSON
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS search_results (
                    id TEXT PRIMARY KEY,
                    request_id TEXT,
                    vector_id TEXT,
                    score REAL,
                    rank INTEGER,
                    s3_image_key TEXT,
                    s3_bucket TEXT,
                    meal_type TEXT,
                    meal_time TEXT,
                    fetch_duration_ms REAL,
                    fetch_success BOOLEAN DEFAULT 1,
                    FOREIGN KEY (request_id) REFERENCES rag_requests (id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embedding_operations (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    operation_type TEXT,  -- 'ingest' or 'query'
                    model_id TEXT,
                    input_type TEXT,  -- 'text', 'image', 'multimodal'
                    duration_ms REAL,
                    embedding_dimension INTEGER,
                    success BOOLEAN DEFAULT 1,
                    error_message TEXT,
                    request_id TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vector_operations (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    operation_type TEXT,  -- 'upsert', 'search', 'delete'
                    collection_name TEXT,
                    vector_count INTEGER,
                    duration_ms REAL,
                    success BOOLEAN DEFAULT 1,
                    error_message TEXT,
                    request_id TEXT
                )
            """)
            
            # Create indexes for better query performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rag_requests_timestamp ON rag_requests(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rag_requests_user_id ON rag_requests(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_search_results_request_id ON search_results(request_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_embedding_operations_timestamp ON embedding_operations(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vector_operations_timestamp ON vector_operations(timestamp)")

    @contextmanager
    def get_connection(self):
        """Get a database connection with proper cleanup."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()

    def log_request_start(self, user_id: str, query_type: str, query_text: str = None, 
                         query_image_path: str = None, filters: Dict = None, 
                         top_k: int = 5, session_id: str = None) -> str:
        """Log the start of a RAG request and return request ID."""
        request_id = str(uuid.uuid4())
        
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO rag_requests (
                    id, user_id, query_type, query_text, query_image_path, 
                    filters, top_k, session_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                request_id, user_id, query_type, query_text, query_image_path,
                str(filters) if filters else None, top_k, session_id
            ))
            conn.commit()
        
        return request_id

    def log_request_completion(self, request_id: str, total_duration_ms: float,
                              embedding_duration_ms: float, search_duration_ms: float,
                              results_count: int, success: bool = True, 
                              error_message: str = None):
        """Log the completion of a RAG request."""
        with self.get_connection() as conn:
            conn.execute("""
                UPDATE rag_requests SET 
                    total_duration_ms = ?, 
                    embedding_duration_ms = ?, 
                    search_duration_ms = ?,
                    results_count = ?,
                    success = ?,
                    error_message = ?
                WHERE id = ?
            """, (
                total_duration_ms, embedding_duration_ms, search_duration_ms,
                results_count, success, error_message, request_id
            ))
            conn.commit()

    def log_search_results(self, request_id: str, results: List[Dict[str, Any]]):
        """Log individual search results."""
        with self.get_connection() as conn:
            for rank, result in enumerate(results, 1):
                result_id = str(uuid.uuid4())
                payload = result.get("payload", {})
                
                conn.execute("""
                    INSERT INTO search_results (
                        id, request_id, vector_id, score, rank, s3_image_key,
                        s3_bucket, meal_type, meal_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result_id, request_id, result.get("id"), result.get("score"),
                    rank, payload.get("s3_image_key"), payload.get("s3_bucket"),
                    payload.get("meal_type"), payload.get("meal_time")
                ))
            conn.commit()

    def log_embedding_operation(self, operation_type: str, model_id: str,
                               input_type: str, duration_ms: float,
                               embedding_dimension: int = None, success: bool = True,
                               error_message: str = None, request_id: str = None) -> str:
        """Log an embedding generation operation."""
        op_id = str(uuid.uuid4())
        
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO embedding_operations (
                    id, operation_type, model_id, input_type, duration_ms,
                    embedding_dimension, success, error_message, request_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                op_id, operation_type, model_id, input_type, duration_ms,
                embedding_dimension, success, error_message, request_id
            ))
            conn.commit()
        
        return op_id

    def log_vector_operation(self, operation_type: str, collection_name: str,
                            duration_ms: float, vector_count: int = 1,
                            success: bool = True, error_message: str = None,
                            request_id: str = None) -> str:
        """Log a vector database operation."""
        op_id = str(uuid.uuid4())
        
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO vector_operations (
                    id, operation_type, collection_name, vector_count, duration_ms,
                    success, error_message, request_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                op_id, operation_type, collection_name, vector_count, duration_ms,
                success, error_message, request_id
            ))
            conn.commit()
        
        return op_id

    def get_metrics_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get a summary of metrics for the last N days."""
        with self.get_connection() as conn:
            # Basic request metrics
            request_stats = conn.execute("""
                SELECT 
                    COUNT(*) as total_requests,
                    AVG(total_duration_ms) as avg_total_duration,
                    AVG(embedding_duration_ms) as avg_embedding_duration,
                    AVG(search_duration_ms) as avg_search_duration,
                    AVG(results_count) as avg_results_count,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
                FROM rag_requests 
                WHERE timestamp >= datetime('now', '-{} days')
            """.format(days)).fetchone()
            
            # Query type distribution
            query_types = conn.execute("""
                SELECT query_type, COUNT(*) as count
                FROM rag_requests 
                WHERE timestamp >= datetime('now', '-{} days')
                GROUP BY query_type
            """.format(days)).fetchall()
            
            # Top users by request count
            top_users = conn.execute("""
                SELECT user_id, COUNT(*) as request_count
                FROM rag_requests 
                WHERE timestamp >= datetime('now', '-{} days')
                GROUP BY user_id
                ORDER BY request_count DESC
                LIMIT 10
            """.format(days)).fetchall()
            
            # Performance percentiles
            performance = conn.execute("""
                SELECT 
                    MIN(total_duration_ms) as min_duration,
                    MAX(total_duration_ms) as max_duration
                FROM rag_requests 
                WHERE timestamp >= datetime('now', '-{} days') AND success = 1
            """.format(days)).fetchone()
            
            return {
                "period_days": days,
                "request_stats": dict(request_stats) if request_stats else {},
                "query_types": [dict(row) for row in query_types],
                "top_users": [dict(row) for row in top_users],
                "performance": dict(performance) if performance else {}
            }

    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent error entries for debugging."""
        with self.get_connection() as conn:
            errors = conn.execute("""
                SELECT timestamp, user_id, query_type, error_message, total_duration_ms
                FROM rag_requests 
                WHERE success = 0 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,)).fetchall()
            
            return [dict(row) for row in errors]

    def cleanup_old_records(self, days_to_keep: int = 30):
        """Clean up old records to manage database size."""
        with self.get_connection() as conn:
            # Clean up old requests and related data
            conn.execute("""
                DELETE FROM search_results 
                WHERE request_id IN (
                    SELECT id FROM rag_requests 
                    WHERE timestamp < datetime('now', '-{} days')
                )
            """.format(days_to_keep))
            
            conn.execute("""
                DELETE FROM rag_requests 
                WHERE timestamp < datetime('now', '-{} days')
            """.format(days_to_keep))
            
            conn.execute("""
                DELETE FROM embedding_operations 
                WHERE timestamp < datetime('now', '-{} days')
            """.format(days_to_keep))
            
            conn.execute("""
                DELETE FROM vector_operations 
                WHERE timestamp < datetime('now', '-{} days')
            """.format(days_to_keep))
            
            conn.commit()


class MetricsTimer:
    """Context manager for timing operations."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.duration_ms = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
