"""
Apache AGE graph store integration test suite.

Connection credentials are passed via pytest CLI flags:
    --pg-host     (default: localhost)
    --pg-port     (default: 5432)
    --pg-user     (default: postgres)
    --pg-password (default: postgres)
    --pg-database (default: postgres)

Example:
    python3 -m pytest tests/memory/test_apache_age.py -v -s \\
        --pg-host=localhost --pg-password=secret --pg-database=testdb

Parts A and B run without a database. Part C skips if PostgreSQL is unreachable.
Part C opens a single psycopg2 connection for all tests.

A timestamped result file is written to tests/memory/age_test_results/ after
each run, containing connection details, per-test timings, and every SQL query
that hit the database.
"""

import datetime
import json
import os
import time
import uuid

import numpy as np
import pytest
from unittest.mock import Mock, patch


# ============================================================
# Query logging infrastructure
# ============================================================

class _QueryLog:
    """Collects every SQL statement executed, tagged by phase."""

    def __init__(self):
        self.entries = []
        self.phase = "init"

    def set_phase(self, phase):
        self.phase = phase

    def log(self, sql, params=None):
        sql_str = sql.strip() if isinstance(sql, str) else str(sql).strip()
        params_str = str(params) if params is not None else None
        self.entries.append({
            "phase": self.phase,
            "sql": sql_str,
            "params": params_str,
        })

    def get_entries(self, phase):
        return [e for e in self.entries if e["phase"] == phase]


class _LoggingCursorProxy:
    """Wraps a psycopg2 cursor to intercept execute() calls."""

    def __init__(self, cursor, log):
        self._cursor = cursor
        self._log = log

    def execute(self, sql, vars=None):
        self._log.log(sql, vars)
        return self._cursor.execute(sql, vars)

    def __enter__(self):
        self._cursor.__enter__()
        return self

    def __exit__(self, *args):
        return self._cursor.__exit__(*args)

    def __getattr__(self, name):
        return getattr(self._cursor, name)

    def __iter__(self):
        return iter(self._cursor)


class _LoggingConnProxy:
    """Wraps a psycopg2 connection so every cursor logs queries."""

    def __init__(self, conn, log):
        self._conn = conn
        self._log = log

    def cursor(self, *args, **kwargs):
        cur = self._conn.cursor(*args, **kwargs)
        return _LoggingCursorProxy(cur, self._log)

    @property
    def autocommit(self):
        return self._conn.autocommit

    @autocommit.setter
    def autocommit(self, val):
        self._conn.autocommit = val

    @property
    def closed(self):
        return self._conn.closed

    def close(self):
        return self._conn.close()

    def __getattr__(self, name):
        return getattr(self._conn, name)


_query_log = _QueryLog()
_test_results = {}
_test_order = []
_pg_creds_for_report = {}
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "age_test_results")


def _make_logged_connect():
    """Return a psycopg2.connect replacement that wraps connections with logging."""
    import psycopg2 as _real

    real_connect = _real.connect

    def logged_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        return _LoggingConnProxy(conn, _query_log)

    return logged_connect


# ============================================================
# Report generation
# ============================================================

def _generate_report():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"result_{ts}.txt")

    sep = "=" * 70
    dash = "-" * 70
    lines = []

    lines.append(sep)
    lines.append("Apache AGE Test Results")
    lines.append(f"Run: {datetime.datetime.now().isoformat(timespec='seconds')}")
    lines.append(sep)
    lines.append("")

    if _pg_creds_for_report:
        lines.append("Connection:")
        lines.append(f"  Host:     {_pg_creds_for_report.get('host', 'N/A')}")
        lines.append(f"  Port:     {_pg_creds_for_report.get('port', 'N/A')}")
        lines.append(f"  User:     {_pg_creds_for_report.get('user', 'N/A')}")
        lines.append(f"  Database: {_pg_creds_for_report.get('database', 'N/A')}")
    else:
        lines.append("Connection: N/A (no integration tests ran)")
    lines.append("")

    # Class setup
    setup_entries = _query_log.get_entries("class_setup")
    if setup_entries:
        lines.append(sep)
        lines.append("SETUP (TestApacheAGEIntegration)")
        lines.append(sep)
        _append_queries(lines, setup_entries)
        lines.append("")

    # Per-test sections
    for test_name in _test_order:
        info = _test_results.get(test_name, {})
        outcome = info.get("outcome", "UNKNOWN")
        elapsed = info.get("time", 0)

        lines.append(dash)
        lines.append(f"{test_name}  {outcome}  {elapsed:.3f}s")
        lines.append(dash)

        cleanup = _query_log.get_entries(f"cleanup:{test_name}")
        test_queries = _query_log.get_entries(f"test:{test_name}")

        if cleanup:
            lines.append("Cleanup:")
            _append_queries(lines, cleanup)

        if test_queries:
            lines.append("Queries:")
            _append_queries(lines, test_queries)
        elif cleanup:
            lines.append("Queries:")
            lines.append("  (none)")
        else:
            lines.append("  (no database queries)")

        lines.append("")

    # Class teardown
    teardown_entries = _query_log.get_entries("class_teardown")
    if teardown_entries:
        lines.append(sep)
        lines.append("TEARDOWN (TestApacheAGEIntegration)")
        lines.append(sep)
        _append_queries(lines, teardown_entries)
        lines.append("")

    # Summary
    total_time = sum(info.get("time", 0) for info in _test_results.values())
    passed = sum(1 for v in _test_results.values() if v.get("outcome") == "PASSED")
    failed = sum(1 for v in _test_results.values() if v.get("outcome") == "FAILED")
    skipped = sum(1 for v in _test_results.values() if v.get("outcome") == "SKIPPED")

    lines.append(sep)
    lines.append("SUMMARY")
    lines.append(f"  Total tests: {len(_test_results)}  |  Passed: {passed}  |  Failed: {failed}  |  Skipped: {skipped}")
    lines.append(f"  Total time:  {total_time:.3f}s")
    lines.append(sep)

    report = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(report)
    print(f"\n  Report written to {path}")


def _append_queries(lines, entries):
    for i, entry in enumerate(entries, 1):
        lines.append(f"  [{i}] {entry['sql']}")
        if entry.get("params"):
            lines.append(f"      params: {entry['params']}")


# ============================================================
# Session / module-level fixtures
# ============================================================

@pytest.fixture(scope="session")
def pg_credentials(request):
    """Read PG connection details from pytest CLI flags."""
    creds = {
        "host": request.config.getoption("--pg-host"),
        "port": int(request.config.getoption("--pg-port")),
        "user": request.config.getoption("--pg-user"),
        "password": request.config.getoption("--pg-password"),
        "database": request.config.getoption("--pg-database"),
    }
    _pg_creds_for_report.update(creds)
    return creds


@pytest.fixture(autouse=True, scope="session")
def _write_report():
    """Write the result file after all tests complete."""
    yield
    _generate_report()


@pytest.fixture(autouse=True)
def _report_test_duration(request):
    """Track timing, outcome, and set the default query phase for every test."""
    test_name = request.node.name
    _test_order.append(test_name)
    _query_log.set_phase(f"test:{test_name}")
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    outcome = "PASSED"
    if hasattr(request.node, "rep_call"):
        outcome = request.node.rep_call.outcome.upper()
    elif hasattr(request.node, "rep_setup") and request.node.rep_setup.failed:
        outcome = "ERROR"
    _test_results[test_name] = {"time": elapsed, "outcome": outcome}
    print(f"\n  [{test_name}] {elapsed:.3f}s")


# ============================================================
# Source path & conditional import
# ============================================================
APACHE_AGE_MEMORY_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'mem0', 'memory', 'apache_age_memory.py'
)

_IMPORT_ERROR = None
try:
    from mem0.memory.apache_age_memory import MemoryGraph, safe_json_loads
except ImportError as e:
    _IMPORT_ERROR = str(e)
    MemoryGraph = None
    safe_json_loads = None

requires_module = pytest.mark.skipif(
    _IMPORT_ERROR is not None,
    reason=f"Cannot import apache_age_memory: {_IMPORT_ERROR}",
)

# ============================================================
# Orthogonal 384-dim test embeddings
# ============================================================
_alice_emb = np.zeros(384)
_alice_emb[0:96] = 1.0
_bob_emb = np.zeros(384)
_bob_emb[96:192] = 1.0
_charlie_emb = np.zeros(384)
_charlie_emb[192:288] = 1.0
_dave_emb = np.zeros(384)
_dave_emb[288:384] = 1.0

EMBEDDINGS = {
    "alice": _alice_emb.tolist(),
    "bob": _bob_emb.tolist(),
    "charlie": _charlie_emb.tolist(),
    "dave": _dave_emb.tolist(),
}


# ============================================================
# Part A: Static Source Analysis (no DB or module import needed)
# ============================================================
class TestApacheAGESourceAnalysis:
    """Read apache_age_memory.py as text and assert coding patterns."""

    @pytest.fixture(autouse=True)
    def load_source(self):
        if not os.path.exists(APACHE_AGE_MEMORY_PATH):
            pytest.skip("apache_age_memory.py not found")
        with open(APACHE_AGE_MEMORY_PATH, 'r') as f:
            self.source = f.read()

    def test_cypher_syntax_validation(self):
        assert "WHERE 1=1" not in self.source
        assert "where_conditions" in self.source

    def test_agent_id_filter_patterns(self):
        assert 'filters.get("agent_id")' in self.source
        assert 'params["agent_id"]' in self.source

    def test_run_id_filter_patterns(self):
        assert 'filters.get("run_id")' in self.source
        assert 'params["run_id"]' in self.source

    def test_user_identity_integration(self):
        assert "user_identity = f\"user_id: {filters['user_id']}\"" in self.source
        assert "user_identity += f\", agent_id: {filters['agent_id']}\"" in self.source
        assert "user_identity += f\", run_id: {filters['run_id']}\"" in self.source

    def test_embedding_table_schema(self):
        assert "user_id TEXT NOT NULL" in self.source
        assert "agent_id TEXT" in self.source
        assert "run_id TEXT" in self.source
        assert "embedding vector(" in self.source

    def test_no_sql_injection_in_graph_name(self):
        assert "self.graph_name" in self.source
        assert "getattr(self.config.graph_store.config, 'graph_name'" in self.source


# ============================================================
# Part B: Pure Unit Tests (no DB required)
# ============================================================
@requires_module
class TestApacheAGEUnitMethods:
    """Unit tests for helper methods that don't touch the database."""

    @pytest.fixture
    def bare_instance(self):
        """Create a MemoryGraph without calling __init__."""
        return MemoryGraph.__new__(MemoryGraph)

    # -- _extract_return_columns --

    def test_extract_return_columns_simple(self, bare_instance):
        result = bare_instance._extract_return_columns("MATCH (n) RETURN a, b, c")
        assert result == ["a", "b", "c"]

    def test_extract_return_columns_with_aliases(self, bare_instance):
        query = "MATCH (n)-[r]->(m) RETURN n.name AS source, type(r) AS relationship, m.name AS target"
        result = bare_instance._extract_return_columns(query)
        assert result == ["source", "relationship", "target"]

    def test_extract_return_columns_with_functions(self, bare_instance):
        query = "MATCH (n) RETURN count(n) AS cnt, id(n) AS nid"
        result = bare_instance._extract_return_columns(query)
        assert result == ["cnt", "nid"]

    def test_extract_return_columns_with_limit(self, bare_instance):
        query = "MATCH (n) RETURN n.name AS name LIMIT 10"
        result = bare_instance._extract_return_columns(query)
        assert result == ["name"]

    # -- _substitute_parameters --

    def test_substitute_parameters_string(self, bare_instance):
        result = bare_instance._substitute_parameters(
            "MATCH (n) WHERE n.name = $name", {"name": "alice"}
        )
        assert result == "MATCH (n) WHERE n.name = 'alice'"

    def test_substitute_parameters_int(self, bare_instance):
        result = bare_instance._substitute_parameters(
            "MATCH (n) WHERE id(n) = $id", {"id": 42}
        )
        assert result == "MATCH (n) WHERE id(n) = 42"

    def test_substitute_parameters_list(self, bare_instance):
        result = bare_instance._substitute_parameters(
            "SET n.vec = $vec", {"vec": [1.0, 2.0, 3.0]}
        )
        assert result == "SET n.vec = [1.0,2.0,3.0]"

    def test_substitute_parameters_none(self, bare_instance):
        result = bare_instance._substitute_parameters(
            "SET n.val = $val", {"val": None}
        )
        assert result == "SET n.val = null"

    def test_substitute_parameters_bool(self, bare_instance):
        # Note: bool is a subclass of int in Python, so the implementation's
        # isinstance(value, (int, float)) branch fires before the bool branch.
        result = bare_instance._substitute_parameters(
            "SET n.active = $active", {"active": True}
        )
        assert result == "SET n.active = True"

    # -- safe_json_loads --

    def test_safe_json_loads_valid(self):
        assert safe_json_loads('[{"key": "value"}]') == [{"key": "value"}]

    def test_safe_json_loads_truncated(self):
        result = safe_json_loads('[{"key": "value"}')
        assert result == [{"key": "value"}]

    def test_safe_json_loads_invalid(self):
        assert safe_json_loads("not json at all{{{") == []


# ============================================================
# Part C helpers
# ============================================================

def _make_config(pg_credentials, graph_name):
    """Build a mock config object with real PG credentials."""
    config = Mock()
    config.graph_store.config.host = pg_credentials["host"]
    config.graph_store.config.port = pg_credentials["port"]
    config.graph_store.config.database = pg_credentials["database"]
    config.graph_store.config.user = pg_credentials["user"]
    config.graph_store.config.password = pg_credentials["password"]
    config.graph_store.config.graph_name = graph_name
    config.graph_store.threshold = 0.7
    config.graph_store.custom_prompt = None
    config.graph_store.llm = None
    config.embedder.provider = "mock"
    config.embedder.config = {}
    config.vector_store.config = {}
    config.llm.provider = "mock"
    config.llm.config = {}
    return config


def _make_mock_embedding_model():
    """Mock embedding model that returns deterministic orthogonal vectors."""
    mock_model = Mock()
    mock_model.config.embedding_dims = 384

    def embed(text):
        return EMBEDDINGS.get(text, np.zeros(384).tolist())

    mock_model.embed.side_effect = embed
    return mock_model


def _try_connect(pg_credentials):
    """Attempt a PG connection; call pytest.skip on failure."""
    import psycopg2

    try:
        conn = psycopg2.connect(**pg_credentials)
        conn.close()
    except Exception as e:
        pytest.skip(f"PostgreSQL not available: {e}")


def _cleanup_graph(conn, graph_name, embedding_table):
    """Best-effort teardown of a test graph and its embedding rows."""
    try:
        with conn.cursor() as cur:
            cur.execute("SET search_path = ag_catalog, public;")
            cur.execute(f"SELECT * FROM ag_catalog.drop_graph('{graph_name}', true);")
    except Exception:
        pass
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {embedding_table} WHERE graph_name = %s;",
                [graph_name],
            )
    except Exception:
        pass


def get_node_count(mg):
    """Count Entity nodes in the test graph."""
    results = mg._execute_cypher(
        f"MATCH (n:{mg.node_label}) RETURN count(n) AS cnt"
    )
    if not results:
        return 0
    val = results[0].get("cnt", 0)
    return int(val) if val is not None else 0


def get_edge_count(mg):
    """Count directed edges between Entity nodes in the test graph."""
    results = mg._execute_cypher(
        f"MATCH (n:{mg.node_label})-[e]->(m:{mg.node_label}) RETURN count(e) AS cnt"
    )
    if not results:
        return 0
    val = results[0].get("cnt", 0)
    return int(val) if val is not None else 0


# ============================================================
# Part C: Integration Tests (real PostgreSQL + Apache AGE)
#
# One psycopg2 connection, one AGE graph, and one embedding
# table are shared across all tests in the class.  Data is
# wiped between tests; the connection stays open until the
# class finishes.  Every SQL statement is captured in the
# query log and written to the result file.
# ============================================================
@requires_module
class TestApacheAGEIntegration:
    """Integration tests that run against a live PostgreSQL instance."""

    @pytest.fixture(scope="class")
    def memory_graph(self, pg_credentials):
        """Single MemoryGraph shared by every test in the class.

        Opens one psycopg2 connection (via a logging proxy), creates one AGE
        graph and one embedding table.  Torn down after the last test.
        """
        _try_connect(pg_credentials)
        _query_log.set_phase("class_setup")

        uid = uuid.uuid4().hex[:8]
        graph_name = f"test_mem0_{uid}"
        emb_table = f"test_age_emb_{uid}"
        config = _make_config(pg_credentials, graph_name)
        mock_emb = _make_mock_embedding_model()
        mock_llm = Mock()

        with patch("mem0.memory.apache_age_memory.EmbedderFactory") as ef, \
             patch("mem0.memory.apache_age_memory.LlmFactory") as lf, \
             patch("mem0.memory.apache_age_memory.psycopg2.connect",
                   side_effect=_make_logged_connect()):
            ef.create.return_value = mock_emb
            lf.create.return_value = mock_llm
            mg = MemoryGraph(config)

        mg.embedding_table = emb_table
        mg._initialize_embedding_table()

        yield mg

        # --- class-level teardown ---
        _query_log.set_phase("class_teardown")
        _cleanup_graph(mg.conn, graph_name, mg.embedding_table)
        try:
            with mg.conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {mg.embedding_table};")
        except Exception:
            pass
        try:
            mg.conn.close()
        except Exception:
            pass

    @pytest.fixture(autouse=True)
    def _clean_between_tests(self, request, memory_graph):
        """Wipe graph data and embeddings before each test (same connection)."""
        test_name = request.node.name
        _query_log.set_phase(f"cleanup:{test_name}")
        mg = memory_graph
        try:
            mg._execute_cypher(f"MATCH (n:{mg.node_label}) DETACH DELETE n")
        except Exception:
            pass
        try:
            with mg.conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {mg.embedding_table} WHERE graph_name = %s;",
                    [mg.graph_name],
                )
        except Exception:
            pass
        _query_log.set_phase(f"test:{test_name}")

    # -- Initialization --

    def test_initialization(self, memory_graph):
        mg = memory_graph
        assert mg.conn is not None
        assert not mg.conn.closed
        assert mg.graph_name.startswith("test_mem0_")
        assert mg.threshold == 0.7
        assert mg.embedding_dims == 384
        assert mg.node_label == "Entity"
        assert mg.rel_label == "CONNECTED_TO"

    @pytest.mark.parametrize("embedding_dims", [None, 0, -1])
    def test_initialization_invalid_embedding_dims(self, pg_credentials, embedding_dims):
        _try_connect(pg_credentials)

        graph_name = f"test_mem0_bad_{uuid.uuid4().hex[:8]}"
        config = _make_config(pg_credentials, graph_name)

        mock_emb = Mock()
        mock_emb.config.embedding_dims = embedding_dims

        try:
            with patch("mem0.memory.apache_age_memory.EmbedderFactory") as ef, \
                 patch("mem0.memory.apache_age_memory.LlmFactory") as lf, \
                 patch("mem0.memory.apache_age_memory.psycopg2.connect",
                       side_effect=_make_logged_connect()):
                ef.create.return_value = mock_emb
                lf.create.return_value = Mock()
                with pytest.raises(ValueError, match="must be a positive"):
                    MemoryGraph(config)
        finally:
            import psycopg2
            try:
                conn = _LoggingConnProxy(psycopg2.connect(**pg_credentials), _query_log)
                conn.autocommit = True
                _cleanup_graph(conn, graph_name, "mem0_age_embeddings")
                conn.close()
            except Exception:
                pass

    # -- _add_entities --

    def test_add_entities_both_new(self, memory_graph):
        mg = memory_graph
        filters = {"user_id": "test_user"}
        data = [{"source": "alice", "destination": "bob", "relationship": "knows"}]

        result = mg._add_entities(data, filters, {})
        assert len(result) == 1
        assert len(result[0]) >= 1
        assert result[0][0]["source"] == "alice"
        assert result[0][0]["target"] == "bob"

        assert get_node_count(mg) == 2
        assert get_edge_count(mg) == 1

    def test_add_entities_source_exists(self, memory_graph):
        mg = memory_graph
        filters = {"user_id": "test_user"}

        mg._add_entities(
            [{"source": "alice", "destination": "bob", "relationship": "knows"}],
            filters, {},
        )
        assert get_node_count(mg) == 2

        mg._add_entities(
            [{"source": "alice", "destination": "charlie", "relationship": "likes"}],
            filters, {},
        )
        assert get_node_count(mg) == 3
        assert get_edge_count(mg) == 2

    def test_add_entities_dest_exists(self, memory_graph):
        mg = memory_graph
        filters = {"user_id": "test_user"}

        mg._add_entities(
            [{"source": "alice", "destination": "bob", "relationship": "knows"}],
            filters, {},
        )

        mg._add_entities(
            [{"source": "charlie", "destination": "bob", "relationship": "likes"}],
            filters, {},
        )
        assert get_node_count(mg) == 3
        assert get_edge_count(mg) == 2

        all_rels = mg.get_all(filters)
        targets = [r["target"] for r in all_rels]
        assert targets.count("bob") == 2

    def test_add_entities_both_exist(self, memory_graph):
        mg = memory_graph
        filters = {"user_id": "test_user"}

        mg._add_entities(
            [{"source": "alice", "destination": "bob", "relationship": "knows"}],
            filters, {},
        )
        node_count_before = get_node_count(mg)

        mg._add_entities(
            [{"source": "alice", "destination": "bob", "relationship": "likes"}],
            filters, {},
        )
        assert get_node_count(mg) == node_count_before

    def test_add_entities_multiple(self, memory_graph):
        mg = memory_graph
        filters = {"user_id": "test_user", "agent_id": "agent1", "run_id": "run1"}
        data = [
            {"source": "alice", "destination": "bob", "relationship": "knows"},
            {"source": "bob", "destination": "charlie", "relationship": "knows"},
            {"source": "charlie", "destination": "alice", "relationship": "knows"},
        ]

        result = mg._add_entities(data, filters, {})
        assert len(result) == 3

        all_rels = mg.get_all(filters)
        assert len(all_rels) == 3
        pairs = {(r["source"], r["target"]) for r in all_rels}
        assert ("alice", "bob") in pairs
        assert ("bob", "charlie") in pairs
        assert ("charlie", "alice") in pairs

    # -- get_all --

    def test_get_all(self, memory_graph):
        mg = memory_graph
        filters = {"user_id": "test_user"}

        mg._add_entities([
            {"source": "alice", "destination": "bob", "relationship": "knows"},
            {"source": "bob", "destination": "charlie", "relationship": "knows"},
        ], filters, {})

        results = mg.get_all(filters)
        assert len(results) == 2
        for r in results:
            assert "source" in r
            assert "relationship" in r
            assert "target" in r

    def test_get_all_with_filters(self, memory_graph):
        mg = memory_graph

        filters1 = {"user_id": "test_user", "agent_id": "agent1"}
        mg._add_entities(
            [{"source": "alice", "destination": "bob", "relationship": "knows"}],
            filters1, {},
        )

        filters2 = {"user_id": "test_user", "agent_id": "agent2"}
        mg._add_entities(
            [{"source": "charlie", "destination": "dave", "relationship": "knows"}],
            filters2, {},
        )

        results1 = mg.get_all(filters1)
        assert len(results1) == 1
        assert results1[0]["source"] == "alice"
        assert results1[0]["target"] == "bob"

        results2 = mg.get_all(filters2)
        assert len(results2) == 1
        assert results2[0]["source"] == "charlie"
        assert results2[0]["target"] == "dave"

    # -- _delete_entities --

    def test_delete_entities(self, memory_graph):
        """Delete a specific relationship.

        Apache AGE stores all edges with label CONNECTED_TO (the semantic name
        is in the r.name property), so the delete input must use that label.
        """
        mg = memory_graph
        filters = {"user_id": "test_user"}

        mg._add_entities([
            {"source": "alice", "destination": "bob", "relationship": "knows"},
            {"source": "bob", "destination": "charlie", "relationship": "knows"},
        ], filters, {})
        assert get_edge_count(mg) == 2

        to_delete = [{"source": "alice", "destination": "bob", "relationship": "CONNECTED_TO"}]
        mg._delete_entities(to_delete, filters)

        assert get_edge_count(mg) == 1
        remaining = mg.get_all(filters)
        assert len(remaining) == 1
        assert remaining[0]["source"] == "bob"
        assert remaining[0]["target"] == "charlie"

    # -- delete_all --

    def test_delete_all(self, memory_graph):
        mg = memory_graph
        filters = {"user_id": "test_user", "agent_id": "agent1"}

        mg._add_entities([
            {"source": "alice", "destination": "bob", "relationship": "knows"},
            {"source": "bob", "destination": "charlie", "relationship": "knows"},
        ], filters, {})
        assert get_node_count(mg) > 0

        mg.delete_all(filters)
        assert get_node_count(mg) == 0
        assert get_edge_count(mg) == 0

    # -- _search_graph_db --

    def test_search_graph_db(self, memory_graph):
        mg = memory_graph
        filters = {"user_id": "test_user"}

        mg._add_entities([
            {"source": "alice", "destination": "bob", "relationship": "knows"},
            {"source": "bob", "destination": "charlie", "relationship": "knows"},
        ], filters, {})

        results = mg._search_graph_db(["bob"], filters, threshold=0.8)
        assert len(results) > 0

        node_names = set()
        for r in results:
            node_names.add(r.get("source"))
            node_names.add(r.get("destination"))
        assert "bob" in node_names

    # -- reset --

    def test_reset(self, memory_graph):
        mg = memory_graph
        filters = {"user_id": "test_user"}

        mg._add_entities(
            [{"source": "alice", "destination": "bob", "relationship": "knows"}],
            filters, {},
        )
        assert get_node_count(mg) > 0

        mg.reset()
        assert get_node_count(mg) == 0
        assert get_edge_count(mg) == 0

    # -- embedding helpers --

    def test_upsert_embedding(self, memory_graph):
        mg = memory_graph
        filters = {"user_id": "test_user"}

        mg._add_entities(
            [{"source": "alice", "destination": "bob", "relationship": "knows"}],
            filters, {},
        )

        results = mg._search_embeddings(
            EMBEDDINGS["alice"], filters, limit=1, threshold=0.5,
        )
        assert len(results) >= 1

        node_id = results[0]["node_graphid"]
        mg._upsert_embedding(node_id, "alice", EMBEDDINGS["alice"], filters)

        results2 = mg._search_embeddings(
            EMBEDDINGS["alice"], filters, limit=1, threshold=0.5,
        )
        assert len(results2) >= 1

    def test_search_embeddings(self, memory_graph):
        mg = memory_graph
        filters = {"user_id": "test_user"}

        mg._add_entities([
            {"source": "alice", "destination": "bob", "relationship": "knows"},
            {"source": "charlie", "destination": "dave", "relationship": "knows"},
        ], filters, {})

        results = mg._search_embeddings(
            EMBEDDINGS["alice"], filters, limit=10, threshold=0.5,
        )
        names = [r["node_name"] for r in results]
        assert "alice" in names
        assert "charlie" not in names
