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

            # Ingestion metrics (one row per ingest operation)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingest_requests (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    image_id TEXT,
                    content_type TEXT,
                    model_id TEXT,
                    output_dim INTEGER,
                    qdrant_collection_name TEXT,
                    s3_bucket TEXT,
                    original_width INTEGER,
                    original_height INTEGER,
                    resized_width INTEGER,
                    resized_height INTEGER,
                    resized_applied BOOLEAN,
                    image_size_bytes INTEGER,
                    embedding_json_size_bytes INTEGER,
                    description_ms REAL,
                    embedding_ms REAL,
                    s3_image_upload_ms REAL,
                    s3_embedding_upload_ms REAL,
                    qdrant_upsert_ms REAL,
                    total_duration_ms REAL,
                    success BOOLEAN,
                    error_step TEXT,
                    error_message TEXT
                )
            """)
            
            # Bulk ingest batch runs (one row per bulk ingest session)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bulk_ingest_runs (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    collection_name TEXT,
                    images_total INTEGER,
                    succeeded INTEGER,
                    failed INTEGER,
                    duration_ms_total REAL,
                    duration_ms_qdrant_upsert REAL,
                    avg_description_ms REAL,
                    avg_embedding_ms REAL,
                    avg_s3_image_upload_ms REAL,
                    avg_s3_embedding_upload_ms REAL,
                    notes TEXT,
                    error_message TEXT
                )
            """)
            
            # Bulk search requests (one row per bulk search)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bulk_search_requests (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    query_type TEXT,
                    top_k INTEGER,
                    score_threshold REAL,
                    duration_ms_total REAL,
                    duration_ms_embedding REAL,
                    duration_ms_search REAL,
                    results_count INTEGER,
                    success BOOLEAN,
                    error_message TEXT
                )
            """)
            
            # Create indexes for better query performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rag_requests_timestamp ON rag_requests(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rag_requests_user_id ON rag_requests(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_search_results_request_id ON search_results(request_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_embedding_operations_timestamp ON embedding_operations(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vector_operations_timestamp ON vector_operations(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ingest_requests_timestamp ON ingest_requests(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ingest_success ON ingest_requests(success)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ingest_image_id ON ingest_requests(image_id)")

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

    # --- Ingestion metrics (demo-friendly) ---
    def log_ingest_record(self, record: Dict[str, Any]):
        """Insert a single ingestion record. Missing keys are stored as NULL."""
        cols = [
            "id", "image_id", "content_type", "model_id", "output_dim",
            "qdrant_collection_name", "s3_bucket", "original_width", "original_height",
            "resized_width", "resized_height", "resized_applied", "image_size_bytes",
            "embedding_json_size_bytes", "description_ms", "embedding_ms",
            "s3_image_upload_ms", "s3_embedding_upload_ms", "qdrant_upsert_ms",
            "total_duration_ms", "success", "error_step", "error_message",
        ]
        values = [record.get(k) for k in cols]
        placeholders = ",".join(["?"] * len(cols))
        with self.get_connection() as conn:
            conn.execute(
                f"INSERT INTO ingest_requests ({','.join(cols)}) VALUES ({placeholders})",
                values,
            )
            conn.commit()

    def get_ingest_summary(self, days: int = 7) -> Dict[str, Any]:
        with self.get_connection() as conn:
            summary = conn.execute(
                f"""
                SELECT
                    COUNT(*) AS total_ingests,
                    CASE WHEN COUNT(*)>0 THEN SUM(CASE WHEN success=1 THEN 1 ELSE 0 END)*100.0/COUNT(*) ELSE 0 END AS success_rate,
                    AVG(total_duration_ms) AS avg_total,
                    AVG(description_ms) AS avg_description,
                    AVG(embedding_ms) AS avg_embedding,
                    AVG(s3_image_upload_ms) AS avg_s3_image,
                    AVG(s3_embedding_upload_ms) AS avg_s3_embedding,
                    AVG(qdrant_upsert_ms) AS avg_qdrant
                FROM ingest_requests
                WHERE timestamp >= datetime('now', '-{days} days')
                """
            ).fetchone()

            perf = conn.execute(
                f"""
                SELECT MIN(total_duration_ms) AS min_total, MAX(total_duration_ms) AS max_total
                FROM ingest_requests
                WHERE timestamp >= datetime('now', '-{days} days') AND success = 1
                """
            ).fetchone()

            return {
                "period_days": days,
                "ingest_stats": dict(summary) if summary else {},
                "ingest_performance": dict(perf) if perf else {},
            }

    def get_recent_ingest_errors(self, limit: int = 5) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, image_id, error_step, error_message
                FROM ingest_requests
                WHERE success = 0
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_recent_ingest_rows(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return the latest N ingestion rows for tabular display."""
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT 
                    timestamp,
                    image_id,
                    content_type,
                    original_width,
                    original_height,
                    resized_applied,
                    resized_width,
                    resized_height,
                    image_size_bytes,
                    embedding_json_size_bytes,
                    description_ms,
                    embedding_ms,
                    s3_image_upload_ms,
                    s3_embedding_upload_ms,
                    qdrant_upsert_ms,
                    total_duration_ms,
                    success,
                    error_step
                FROM ingest_requests
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def log_bulk_ingest_run(self, record: Dict[str, Any]) -> str:
        """Insert a bulk ingest run summary and return its ID."""
        run_id = record.get("id") or str(uuid.uuid4())
        cols = [
            "id", "collection_name", "images_total", "succeeded", "failed",
            "duration_ms_total", "duration_ms_qdrant_upsert",
            "avg_description_ms", "avg_embedding_ms",
            "avg_s3_image_upload_ms", "avg_s3_embedding_upload_ms",
            "notes", "error_message",
        ]
        values = [run_id] + [record.get(k) for k in cols[1:]]
        with self.get_connection() as conn:
            placeholders = ",".join(["?"] * len(cols))
            conn.execute(
                f"INSERT INTO bulk_ingest_runs ({','.join(cols)}) VALUES ({placeholders})",
                values,
            )
            conn.commit()
        return run_id

    def get_bulk_ingest_summary(self, days: int = 7) -> Dict[str, Any]:
        with self.get_connection() as conn:
            row = conn.execute(
                f"""
                SELECT
                    COUNT(*) AS total_runs,
                    COALESCE(SUM(images_total),0) AS total_images,
                    COALESCE(SUM(succeeded),0) AS total_succeeded,
                    COALESCE(SUM(failed),0) AS total_failed,
                    AVG(duration_ms_total) AS avg_total_ms,
                    AVG(duration_ms_qdrant_upsert) AS avg_qdrant_ms
                FROM bulk_ingest_runs
                WHERE timestamp >= datetime('now', '-{days} days')
                """
            ).fetchone()
            return dict(row) if row else {}

    def get_recent_bulk_ingest_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, collection_name, images_total, succeeded, failed,
                       duration_ms_total, duration_ms_qdrant_upsert, error_message
                FROM bulk_ingest_runs
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def log_bulk_search_request(self, record: Dict[str, Any]) -> str:
        req_id = record.get("id") or str(uuid.uuid4())
        cols = [
            "id", "query_type", "top_k", "score_threshold",
            "duration_ms_total", "duration_ms_embedding", "duration_ms_search",
            "results_count", "success", "error_message",
        ]
        values = [req_id] + [record.get(k) for k in cols[1:]]
        with self.get_connection() as conn:
            placeholders = ",".join(["?"] * len(cols))
            conn.execute(
                f"INSERT INTO bulk_search_requests ({','.join(cols)}) VALUES ({placeholders})",
                values,
            )
            conn.commit()
        return req_id

    def get_bulk_search_summary(self, days: int = 7) -> Dict[str, Any]:
        with self.get_connection() as conn:
            row = conn.execute(
                f"""
                SELECT
                    COUNT(*) AS total_requests,
                    AVG(duration_ms_total) AS avg_total_ms,
                    AVG(duration_ms_embedding) AS avg_embed_ms,
                    AVG(duration_ms_search) AS avg_search_ms,
                    AVG(results_count) AS avg_results,
                    SUM(CASE WHEN success=1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS success_rate
                FROM bulk_search_requests
                WHERE timestamp >= datetime('now', '-{days} days')
                """
            ).fetchone()
            return dict(row) if row else {}

    def get_recent_bulk_search_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, query_type, error_message, duration_ms_total
                FROM bulk_search_requests
                WHERE success = 0
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

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
            
            conn.execute("""
                DELETE FROM ingest_requests 
                WHERE timestamp < datetime('now', '-{} days')
            """.format(days_to_keep))

            conn.execute("""
                DELETE FROM bulk_ingest_runs
                WHERE timestamp < datetime('now', '-{} days')
            """.format(days_to_keep))

            conn.execute("""
                DELETE FROM bulk_search_requests
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
