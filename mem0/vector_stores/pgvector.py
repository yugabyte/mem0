import json
import logging
from contextlib import contextmanager
from typing import Any, List, Optional
from langfuse import observe
from pydantic import BaseModel

# Try to import psycopg (psycopg3) first, then fall back to psycopg2
try:
    from psycopg.types.json import Json
    from psycopg_pool import ConnectionPool
    PSYCOPG_VERSION = 3
    logger = logging.getLogger(__name__)
    logger.info("Using psycopg (psycopg3) with ConnectionPool for PostgreSQL connections")
except ImportError:
    try:
        from psycopg2.extras import Json, execute_values
        from psycopg2.pool import ThreadedConnectionPool as ConnectionPool
        PSYCOPG_VERSION = 2
        logger = logging.getLogger(__name__)
        logger.info("Using psycopg2 with ThreadedConnectionPool for PostgreSQL connections")
    except ImportError:
        raise ImportError(
            "Neither 'psycopg' nor 'psycopg2' library is available. "
            "Please install one of them using 'pip install psycopg[pool]' or 'pip install psycopg2'"
        )

from mem0.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class OutputData(BaseModel):
    id: Optional[str]
    score: Optional[float]
    payload: Optional[dict]


class PGVector(VectorStoreBase):
    def __init__(
        self,
        dbname,
        collection_name,
        embedding_model_dims,
        user,
        password,
        host,
        port,
        diskann,
        hnsw,
        minconn=1,
        maxconn=5,
        sslmode=None,
        connection_string=None,
        connection_pool=None,
    ):
        self.collection_name = collection_name
        self.use_diskann = diskann
        self.use_hnsw = hnsw
        self.embedding_model_dims = embedding_model_dims
        self.connection_pool = None

        if connection_pool is not None:
            self.connection_pool = connection_pool
        elif connection_string:
            if sslmode:
                if 'sslmode=' in connection_string:
                    import re
                    connection_string = re.sub(r'sslmode=[^ ]*', f'sslmode={sslmode}', connection_string)
                else:
                    connection_string = f"{connection_string} sslmode={sslmode}"
        else:
            connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
            if sslmode:
                connection_string = f"{connection_string} sslmode={sslmode}"
        
        if self.connection_pool is None:
            if PSYCOPG_VERSION == 3:
                self.connection_pool = ConnectionPool(conninfo=connection_string, min_size=minconn, max_size=maxconn, open=True)
            else:
                self.connection_pool = ConnectionPool(minconn=minconn, maxconn=maxconn, dsn=connection_string)

        collections = self.list_cols()
        if collection_name not in collections:
            self.create_col()

    # --- NEW TRACING WRAPPERS ---
    @observe(name="Execute Query (yugabytedb / vector)", as_type="span")
    def _execute(self, cur, query: str, params=None):
        """Wrapper to trace individual execute calls."""
        if params:
            cur.execute(query, params)
        else:
            cur.execute(query)
        return cur

    @observe(name="Execute Query Many (yugabytedb / vector)", as_type="span")
    def _executemany(self, cur, query: str, vars_list):
        """Wrapper to trace executemany calls."""
        cur.executemany(query, vars_list)
        return cur

    @observe(name="Execute Query Values (yugabytedb / vector)", as_type="span")
    def _execute_values(self, cur, query: str, argslist):
        """Wrapper to trace execute_values calls (psycopg2)."""
        execute_values(cur, query, argslist)
        return cur
    # ----------------------------

    @contextmanager
    def _get_cursor(self, commit: bool = False):
        if PSYCOPG_VERSION == 3:
            with self.connection_pool.connection() as conn:
                with conn.cursor() as cur:
                    try:
                        yield cur
                        if commit:
                            conn.commit()
                    except Exception:
                        conn.rollback()
                        logger.error("Error in cursor context (psycopg3)", exc_info=True)
                        raise
        else:
            conn = self.connection_pool.getconn()
            cur = conn.cursor()
            try:
                yield cur
                if commit:
                    conn.commit()
            except Exception as exc:
                conn.rollback()
                logger.error(f"Error occurred: {exc}")
                raise exc
            finally:
                cur.close()
                self.connection_pool.putconn(conn)

    @observe(name="Collection Creation (yugabytedb / vector)", as_type="span")
    def create_col(self) -> None:
        with self._get_cursor(commit=True) as cur:
            self._execute(cur, "CREATE EXTENSION IF NOT EXISTS vector")
            self._execute(
                cur,
                f"""
                CREATE TABLE IF NOT EXISTS {self.collection_name} (
                    id UUID PRIMARY KEY,
                    vector vector({self.embedding_model_dims}),
                    payload JSONB
                );
                """
            )
            if self.use_diskann and self.embedding_model_dims < 2000:
                self._execute(cur, "SELECT * FROM pg_extension WHERE extname = 'vectorscale'")
                if cur.fetchone():
                    self._execute(
                        cur,
                        f"""
                        CREATE INDEX IF NOT EXISTS {self.collection_name}_diskann_idx
                        ON {self.collection_name}
                        USING diskann (vector);
                        """
                    )
            elif self.use_hnsw:
                self._execute(
                    cur,
                    f"""
                    CREATE INDEX IF NOT EXISTS {self.collection_name}_hnsw_idx
                    ON {self.collection_name}
                    USING hnsw (vector vector_cosine_ops)
                    """
                )

    @observe(name="Insertion (yugabytedb / vector)", as_type="span")
    def insert(self, vectors: list[list[float]], payloads=None, ids=None) -> None:
        logger.info(f"Inserting {len(vectors)} vectors into collection {self.collection_name}")
        json_payloads = [json.dumps(payload) for payload in payloads]

        data = [(id, vector, payload) for id, vector, payload in zip(ids, vectors, json_payloads)]
        if PSYCOPG_VERSION == 3:
            with self._get_cursor(commit=True) as cur:
                self._executemany(
                    cur,
                    f"INSERT INTO {self.collection_name} (id, vector, payload) VALUES (%s, %s, %s)",
                    data,
                )
        else:
            with self._get_cursor(commit=True) as cur:
                self._execute_values(
                    cur,
                    f"INSERT INTO {self.collection_name} (id, vector, payload) VALUES %s",
                    data,
                )

    @observe(name="Searching (yugabytedb / vector)", as_type="retriever")
    def search(
        self,
        query: str,
        vectors: list[float],
        limit: Optional[int] = 5,
        filters: Optional[dict] = None,
    ) -> List[OutputData]:
        filter_conditions = []
        filter_params = []

        if filters:
            for k, v in filters.items():
                filter_conditions.append("payload->>%s = %s")
                filter_params.extend([k, str(v)])

        filter_clause = "WHERE " + " AND ".join(filter_conditions) if filter_conditions else ""

        with self._get_cursor() as cur:
            self._execute(
                cur,
                f"""
                SELECT id, vector <=> %s::vector AS distance, payload
                FROM {self.collection_name}
                {filter_clause}
                ORDER BY distance
                LIMIT %s
                """,
                (vectors, *filter_params, limit),
            )

            results = cur.fetchall()
        return [OutputData(id=str(r[0]), score=float(r[1]), payload=r[2]) for r in results]

    @observe(name="Deletion (yugabytedb / vector)", as_type="span")
    def delete(self, vector_id: str) -> None:
        with self._get_cursor(commit=True) as cur:
            self._execute(cur, f"DELETE FROM {self.collection_name} WHERE id = %s", (vector_id,))

    @observe(name="Updation (yugabytedb / vector)", as_type="span")
    def update(
        self,
        vector_id: str,
        vector: Optional[list[float]] = None,
        payload: Optional[dict] = None,
    ) -> None:
        with self._get_cursor(commit=True) as cur:
            if vector:
               self._execute(
                    cur,
                    f"UPDATE {self.collection_name} SET vector = %s WHERE id = %s",
                    (vector, vector_id),
                )
            if payload:
                if PSYCOPG_VERSION == 3:
                    self._execute(
                        cur,
                        f"UPDATE {self.collection_name} SET payload = %s WHERE id = %s",
                        (Json(payload), vector_id),
                    )
                else:
                    self._execute(
                        cur,
                        f"UPDATE {self.collection_name} SET payload = %s WHERE id = %s",
                        (Json(payload), vector_id),
                    )

    @observe(name="Retrieval (yugabytedb / vector)", as_type="retriever")
    def get(self, vector_id: str) -> OutputData:
        with self._get_cursor() as cur:
            self._execute(
                cur,
                f"SELECT id, vector, payload FROM {self.collection_name} WHERE id = %s",
                (vector_id,),
            )
            result = cur.fetchone()
            if not result:
                return None
            return OutputData(id=str(result[0]), score=None, payload=result[2])

    @observe(name="Listing Collections (yugabytedb / vector)", as_type="retriever")
    def list_cols(self) -> List[str]:
        with self._get_cursor() as cur:
            self._execute(cur, "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
            return [row[0] for row in cur.fetchall()]

    @observe(name="Collection Deletion (yugabytedb / vector)", as_type="span")
    def delete_col(self) -> None:
        with self._get_cursor(commit=True) as cur:
            self._execute(cur, f"DROP TABLE IF EXISTS {self.collection_name}")

    @observe(name="Collection Information (yugabytedb / vector)", as_type="retriever")
    def col_info(self) -> dict[str, Any]:
        with self._get_cursor() as cur:
            self._execute(
                cur,
                f"""
                SELECT
                    table_name,
                    (SELECT COUNT(*) FROM {self.collection_name}) as row_count,
                    (SELECT pg_size_pretty(pg_total_relation_size('{self.collection_name}'))) as total_size
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = %s
            """,
                (self.collection_name,),
            )
            result = cur.fetchone()
        return {"name": result[0], "count": result[1], "size": result[2]}

    @observe(name="Listing Vectors (yugabytedb / vector)", as_type="retriever")
    def list(
        self,
        filters: Optional[dict] = None,
        limit: Optional[int] = 100
    ) -> List[OutputData]:
        filter_conditions = []
        filter_params = []

        if filters:
            for k, v in filters.items():
                filter_conditions.append("payload->>%s = %s")
                filter_params.extend([k, str(v)])

        filter_clause = "WHERE " + " AND ".join(filter_conditions) if filter_conditions else ""

        query = f"""
            SELECT id, vector, payload
            FROM {self.collection_name}
            {filter_clause}
            LIMIT %s
        """

        with self._get_cursor() as cur:
            self._execute(cur, query, (*filter_params, limit))
            results = cur.fetchall()
        return [[OutputData(id=str(r[0]), score=None, payload=r[2]) for r in results]]

    def __del__(self) -> None:
        try:
            if PSYCOPG_VERSION == 3:
                self.connection_pool.close()
            else:
                self.connection_pool.closeall()
        except Exception:
            pass

    @observe(name="Resetting Index (yugabytedb / vector)", as_type="span")
    def reset(self) -> None:
        logger.warning(f"Resetting index {self.collection_name}...")
        self.delete_col()
        self.create_col()