import json
import logging
import time

from mem0.memory.utils import format_entities, sanitize_relationship_for_cypher

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    raise ImportError("psycopg2 is not installed. Please install it using pip install psycopg2-binary")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("rank_bm25 is not installed. Please install it using pip install rank-bm25")

from mem0.graphs.tools import (
    DELETE_MEMORY_STRUCT_TOOL_GRAPH,
    DELETE_MEMORY_TOOL_GRAPH,
    EXTRACT_ENTITIES_STRUCT_TOOL,
    EXTRACT_ENTITIES_TOOL,
    RELATIONS_STRUCT_TOOL,
    RELATIONS_TOOL,
)
from mem0.graphs.utils import EXTRACT_RELATIONS_PROMPT, get_delete_messages
from mem0.utils.factory import EmbedderFactory, LlmFactory

logger = logging.getLogger(__name__)

def safe_json_loads(s):
    if not isinstance(s, str):
        return s

    # Try normal parse first
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # ---- Repair common truncation: missing closing brackets ----
    open_brackets = s.count("[")
    close_brackets = s.count("]")

    if close_brackets < open_brackets:
        s = s + ("]" * (open_brackets - close_brackets))

    # Try again
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return []

class MemoryGraph:
    def __init__(self, config):
        self.config = config
        
        # Initialize PostgreSQL connection for Apache AGE
        self.conn = psycopg2.connect(
            host=self.config.graph_store.config.host,
            port=self.config.graph_store.config.port,
            database=self.config.graph_store.config.database,
            user=self.config.graph_store.config.user,
            password=self.config.graph_store.config.password,
        )
        self.conn.autocommit = True
        
        # Get or create graph name
        self.graph_name = getattr(self.config.graph_store.config, 'graph_name', 'mem0_graph')
        
        # Initialize Apache AGE
        self._initialize_age()
        
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )
        self.embedding_dims = self.embedding_model.config.embedding_dims

        if self.embedding_dims is None or self.embedding_dims <= 0:
            raise ValueError(f"embedding_dims must be a positive integer. Given: {self.embedding_dims}")

        self.embedding_table = "mem0_age_embeddings"
        self._initialize_embedding_table()

        self.node_label = "Entity"
        self.rel_label = "CONNECTED_TO"

        # Default to openai if no specific provider is configured
        self.llm_provider = "openai"
        if self.config.llm and self.config.llm.provider:
            self.llm_provider = self.config.llm.provider
        if self.config.graph_store and self.config.graph_store.llm and self.config.graph_store.llm.provider:
            self.llm_provider = self.config.graph_store.llm.provider
        
        # Get LLM config with proper null checks
        llm_config = None
        if self.config.graph_store and self.config.graph_store.llm and hasattr(self.config.graph_store.llm, "config"):
            llm_config = self.config.graph_store.llm.config
        elif hasattr(self.config.llm, "config"):
            llm_config = self.config.llm.config
        self.llm = LlmFactory.create(self.llm_provider, llm_config)

        self.user_id = None
        # Use threshold from graph_store config, default to 0.7 for backward compatibility
        self.threshold = self.config.graph_store.threshold if hasattr(self.config.graph_store, 'threshold') else 0.7

    def _initialize_age(self):
        """Initialize Apache AGE extension and create graph if not exists."""
        with self.conn.cursor() as cursor:
            # Load AGE extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS age;")
            
            # Load ag_catalog to the search path
            cursor.execute("SET search_path = ag_catalog, public;")
            
            # Check if graph already exists
            cursor.execute(f"SELECT * FROM ag_catalog.ag_graph WHERE name = '{self.graph_name}';")
            if not cursor.fetchone():
                # Create graph only if it doesn't exist
                cursor.execute(f"SELECT * FROM ag_catalog.create_graph('{self.graph_name}');")
                logger.info(f"Created graph '{self.graph_name}'")
            else:
                logger.info(f"Graph '{self.graph_name}' already exists")

    def _initialize_embedding_table(self):
        """Initialize pgvector storage for embeddings."""
        with self.conn.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.embedding_table} (
                    graph_name TEXT NOT NULL,
                    node_graphid BIGINT NOT NULL,
                    node_name TEXT,
                    embedding vector({self.embedding_dims}) NOT NULL,
                    user_id TEXT NOT NULL,
                    agent_id TEXT,
                    run_id TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (graph_name, node_graphid)
                );
                """
            )
            cursor.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self.embedding_table}_filters_idx
                ON {self.embedding_table} (graph_name, user_id, agent_id, run_id);
                """
            )

    def _execute_sql(self, sql, params=None, fetch=False):
        with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(sql, params or [])
            if fetch:
                return cursor.fetchall()
            return []

    def _upsert_embedding(self, node_graphid, node_name, embedding, filters):
        sql = f"""
        INSERT INTO {self.embedding_table}
            (graph_name, node_graphid, node_name, embedding, user_id, agent_id, run_id)
        VALUES
            (%s, %s, %s, %s::vector, %s, %s, %s)
        ON CONFLICT (graph_name, node_graphid)
        DO UPDATE SET
            node_name = EXCLUDED.node_name,
            embedding = EXCLUDED.embedding,
            user_id = EXCLUDED.user_id,
            agent_id = EXCLUDED.agent_id,
            run_id = EXCLUDED.run_id,
            updated_at = NOW();
        """
        params = [
            self.graph_name,
            int(node_graphid),
            node_name,
            embedding,
            filters["user_id"],
            filters.get("agent_id"),
            filters.get("run_id"),
        ]
        self._execute_sql(sql, params=params)

    def _delete_embeddings(self, filters=None, graph_only=False):
        where_conditions = ["graph_name = %s"]
        params = [self.graph_name]

        if not graph_only and filters:
            where_conditions.append("user_id = %s")
            params.append(filters["user_id"])
            if filters.get("agent_id"):
                where_conditions.append("agent_id = %s")
                params.append(filters["agent_id"])
            if filters.get("run_id"):
                where_conditions.append("run_id = %s")
                params.append(filters["run_id"])

        where_clause = " AND ".join(where_conditions)
        sql = f"DELETE FROM {self.embedding_table} WHERE {where_clause};"
        self._execute_sql(sql, params=params)

    def _search_embeddings(self, embedding, filters, limit=100, threshold=None):
        threshold_val = threshold if threshold is not None else self.threshold

        where_conditions = ["graph_name = %s", "user_id = %s"]
        params = [self.graph_name, filters["user_id"]]
        if filters.get("agent_id"):
            where_conditions.append("agent_id = %s")
            params.append(filters["agent_id"])
        if filters.get("run_id"):
            where_conditions.append("run_id = %s")
            params.append(filters["run_id"])

        where_clause = " AND ".join(where_conditions)
        sql = f"""
        SELECT node_graphid, node_name, 1 - distance AS similarity
        FROM (
            SELECT node_graphid, node_name, embedding <=> %s::vector AS distance
            FROM {self.embedding_table}
            WHERE {where_clause}
        ) AS ranked
        WHERE (1 - distance) >= %s
        ORDER BY distance ASC
        LIMIT %s;
        """
        query_params = [embedding, *params, threshold_val, limit]
        return self._execute_sql(sql, params=query_params, fetch=True)

    def _execute_cypher(self, cypher_query, parameters=None):
        """Execute a Cypher query using Apache AGE.
        
        Note: Apache AGE doesn't support parameterized queries in the same way as Neo4j.
        We need to substitute parameter values directly into the Cypher query string.
        """
        import json
        import re
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Set search path for AGE
            cursor.execute("SET search_path = ag_catalog, public;")
            
            # Substitute parameters into the query if provided
            if parameters:
                processed_query = self._substitute_parameters(cypher_query, parameters)
            else:
                processed_query = cypher_query
            
            # Extract column names from the RETURN clause
            column_names = self._extract_return_columns(processed_query)
            
            # Build column definitions - use the actual column names from RETURN clause
            if column_names:
                column_defs = ", ".join([f"{col} agtype" for col in column_names])
            else:
                # Fallback to single result column
                column_defs = "result agtype"
            
            # Build the cypher function call
            age_query = f"SELECT * FROM cypher('{self.graph_name}', $$ {processed_query} $$) as ({column_defs});"
            logger.debug(f"Executing AGE query: {age_query}")
            
            try:
                cursor.execute(age_query)
                results = cursor.fetchall()
                
                # Debug: log raw results to understand agtype format
                if results:
                    logger.debug(f"Raw AGE results (first row): {results[0]}")
                    logger.debug(f"Raw AGE result types: {[(k, type(v).__name__, repr(v)[:100]) for k, v in results[0].items()]}")
                else:
                    logger.debug(f"Query returned NO results")
                    logger.debug(f"Query was: {age_query[:500]}...")
                
                # Parse agtype results - Apache AGE returns data in agtype format
                # which needs to be converted from JSON strings to Python objects
                parsed_results = []
                for row in results:
                    parsed_row = {}
                    for key, value in row.items():
                        # agtype values are stored as strings, parse them
                        if value is not None:
                            try:
                                # Try to parse as JSON (agtype format)
                                if isinstance(value, str):
                                    # Remove the agtype type annotation if present
                                    # Format: "string_value"::agtype or just "string_value"
                                    parsed_value = json.loads(value) if value.startswith('{') or value.startswith('[') or value.startswith('"') else value
                                else:
                                    parsed_value = value
                                parsed_row[key] = parsed_value
                            except (json.JSONDecodeError, TypeError):
                                # If parsing fails, use the raw value
                                parsed_row[key] = value
                        else:
                            parsed_row[key] = value
                    parsed_results.append(parsed_row)
                
                logger.debug(f"Parsed results (first row): {parsed_results[0] if parsed_results else 'None'}")
                return parsed_results
            except Exception as e:
                logger.error(f"Error executing Cypher query: {e}")
                logger.error(f"Query: {age_query}")
                raise
    
    def _extract_return_columns(self, cypher_query):
        """Extract column names from the RETURN clause of a Cypher query.
        
        Returns a list of column names (aliases) from the RETURN statement.
        """
        import re
        
        # Find the last RETURN statement (in case of UNION queries)
        # Match RETURN followed by column definitions, stopping at ORDER BY, LIMIT, or end
        return_pattern = r'RETURN\s+(.*?)(?:ORDER\s+BY|LIMIT|$)'
        matches = list(re.finditer(return_pattern, cypher_query, re.IGNORECASE | re.DOTALL))
        
        if not matches:
            return []
        
        # Get the last RETURN clause
        return_clause = matches[-1].group(1).strip()
        
        # Split by commas (but not commas inside function calls)
        columns = []
        depth = 0
        current_col = ""
        
        for char in return_clause:
            if char == '(':
                depth += 1
                current_col += char
            elif char == ')':
                depth -= 1
                current_col += char
            elif char == ',' and depth == 0:
                columns.append(current_col.strip())
                current_col = ""
            else:
                current_col += char
        
        # Don't forget the last column
        if current_col.strip():
            columns.append(current_col.strip())
        
        # Extract aliases (the part after AS, or the whole expression if no AS)
        column_names = []
        for col in columns:
            # Look for ' AS alias' pattern
            as_match = re.search(r'\s+AS\s+(\w+)', col, re.IGNORECASE)
            if as_match:
                column_names.append(as_match.group(1))
            else:
                # No AS clause - try to extract a simple identifier
                # Remove whitespace and get the last word
                simple_name = col.strip().split()[-1]
                # Clean up any special characters
                simple_name = re.sub(r'[^\w]', '', simple_name)
                if simple_name:
                    column_names.append(simple_name)
        
        return column_names
    
    def _substitute_parameters(self, cypher_query, parameters):
        """Substitute parameter placeholders with actual values for Apache AGE.
        
        Apache AGE doesn't support parameterized queries, so we need to inline values.
        This method replaces $param_name with the actual value.
        """
        import json
        import re
        
        query = cypher_query
        for key, value in parameters.items():
            # Check if this parameter is used with a ::vector cast
            vector_pattern = f"\\${key}::vector"
            has_vector_cast = vector_pattern in query
            
            placeholder = f"${key}"
            
            # Convert value to appropriate Cypher literal
            if isinstance(value, str):
                # Escape single quotes in strings
                escaped_value = value.replace("'", "\\'")
                cypher_value = f"'{escaped_value}'"
            elif isinstance(value, (int, float)):
                cypher_value = str(value)
            elif isinstance(value, bool):
                cypher_value = "true" if value else "false"
            elif isinstance(value, list):
                # For arrays/vectors, convert to JSON array format for Apache AGE
                if value and isinstance(value[0], (int, float)):
                    # This is a vector embedding - format as JSON array
                    # Apache AGE will cast [1.0, 2.0, 3.0]::vector to vector type
                    array_elements = ",".join(str(v) for v in value)
                    cypher_value = f"[{array_elements}]"
                else:
                    # Regular list - store as JSON
                    cypher_value = json.dumps(value)
            elif value is None:
                cypher_value = "null"
            else:
                # For other types, try JSON serialization
                cypher_value = json.dumps(value)
            
            # Replace the placeholder (with or without ::vector cast)
            if has_vector_cast:
                query = query.replace(vector_pattern, f"{cypher_value}::vector")
            else:
                query = query.replace(placeholder, cypher_value)
        
        return query

    def add(self, data, filters):
        """
        Adds data to the graph.

        Args:
            data (str): The data to add to the graph.
            filters (dict): A dictionary containing filters to be applied during the addition.
        """
        entity_type_map = self._retrieve_nodes_from_data(data, filters)
        to_be_added = self._establish_nodes_relations_from_data(data, filters, entity_type_map)
        search_output = self._search_graph_db(node_list=list(entity_type_map.keys()), filters=filters)
        to_be_deleted = self._get_delete_entities_from_search_output(search_output, data, filters)
        deleted_entities = self._delete_entities(to_be_deleted, filters)
        added_entities = self._add_entities(to_be_added, filters, entity_type_map)
        return {"deleted_entities": deleted_entities, "added_entities": added_entities}

    def search(self, query, filters, limit=5):
        """
        Search for memories and related graph data.

        Args:
            query (str): Query to search for.
            filters (dict): A dictionary containing filters to be applied during the search.
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 5.

        Returns:
            dict: A dictionary containing:
                - "contexts": List of search results from the base data store.
                - "entities": List of related graph data based on the query.
        """
        entity_type_map = self._retrieve_nodes_from_data(query, filters)
        search_output = self._search_graph_db(node_list=list(entity_type_map.keys()), filters=filters)

        if not search_output:
            return []

        search_outputs_sequence = [
            [item["source"], item["relationship"], item["destination"]] for item in search_output
        ]
        bm25 = BM25Okapi(search_outputs_sequence)

        tokenized_query = query.split(" ")
        reranked_results = bm25.get_top_n(tokenized_query, search_outputs_sequence, n=limit)

        search_results = []
        for item in reranked_results:
            search_results.append({"source": item[0], "relationship": item[1], "destination": item[2]})

        logger.info(f"Returned {len(search_results)} search results")

        return search_results

    def delete_all(self, filters):
        """Delete all nodes and relationships for a user or specific agent."""
        # Build node properties for filtering
        where_conditions = ["n.user_id = $user_id"]
        if filters.get("agent_id"):
            where_conditions.append("n.agent_id = $agent_id")
        if filters.get("run_id"):
            where_conditions.append("n.run_id = $run_id")
        where_clause = " AND ".join(where_conditions)

        cypher = f"""
        MATCH (n:{self.node_label})
        WHERE {where_clause}
        DETACH DELETE n
        """
        
        params = {"user_id": filters["user_id"]}
        if filters.get("agent_id"):
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            params["run_id"] = filters["run_id"]
            
        self._execute_cypher(cypher, parameters=params)
        self._delete_embeddings(filters=filters)

    def get_all(self, filters, limit=100):
        """
        Retrieves all nodes and relationships from the graph database based on optional filtering criteria.
        
        Args:
            filters (dict): A dictionary containing filters to be applied during the retrieval.
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.
        Returns:
            list: A list of dictionaries, each containing:
                - 'source': The source node name.
                - 'relationship': The relationship type.
                - 'target': The target node name.
        """
        # Build WHERE conditions
        where_conditions = ["n.user_id = $user_id", "m.user_id = $user_id"]
        if filters.get("agent_id"):
            where_conditions.append("n.agent_id = $agent_id")
            where_conditions.append("m.agent_id = $agent_id")
        if filters.get("run_id"):
            where_conditions.append("n.run_id = $run_id")
            where_conditions.append("m.run_id = $run_id")
        where_clause = " AND ".join(where_conditions)

        query = f"""
        MATCH (n:{self.node_label})-[r]->(m:{self.node_label})
        WHERE {where_clause}
        RETURN n.name AS source, type(r) AS relationship, m.name AS target
        LIMIT $limit
        """
        
        params = {"user_id": filters["user_id"], "limit": limit}
        if filters.get("agent_id"):
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            params["run_id"] = filters["run_id"]

        results = self._execute_cypher(query, parameters=params)

        final_results = []
        for result in results:
            final_results.append(
                {
                    "source": result["source"],
                    "relationship": result["relationship"],
                    "target": result["target"],
                }
            )

        logger.info(f"Retrieved {len(final_results)} relationships")

        return final_results

    def _retrieve_nodes_from_data(self, data, filters):
        """Extracts all the entities mentioned in the query."""
        _tools = [EXTRACT_ENTITIES_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [EXTRACT_ENTITIES_STRUCT_TOOL]
        search_results = self.llm.generate_response(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a smart assistant who understands entities and their types in a given text. If user message contains self reference such as 'I', 'me', 'my' etc. then use {filters['user_id']} as the source entity. Extract all the entities from the text. ***DO NOT*** answer the question itself if the given text is a question.",
                },
                {"role": "user", "content": data},
            ],
            tools=_tools,
        )

        entity_type_map = {}

        try:
            for tool_call in search_results["tool_calls"]:
                if tool_call["name"] != "extract_entities":
                    continue
                entities = tool_call["arguments"]["entities"]
                if isinstance(entities, str):
                    entities = safe_json_loads(entities)
                for item in entities:
                    entity_type_map[item["entity"]] = item["entity_type"]
        except Exception as e:
            logger.exception(
                f"Error in search tool: {e}, llm_provider={self.llm_provider}, search_results={search_results}, entities={entities}"
            )

        entity_type_map = {k.lower().replace(" ", "_"): v.lower().replace(" ", "_") for k, v in entity_type_map.items()}
        logger.debug(f"Entity type map: {entity_type_map}\n search_results={search_results}")
        return entity_type_map

    def _establish_nodes_relations_from_data(self, data, filters, entity_type_map):
        """Establish relations among the extracted nodes."""

        # Compose user identification string for prompt
        user_identity = f"user_id: {filters['user_id']}"
        if filters.get("agent_id"):
            user_identity += f", agent_id: {filters['agent_id']}"
        if filters.get("run_id"):
            user_identity += f", run_id: {filters['run_id']}"

        if self.config.graph_store.custom_prompt:
            system_content = EXTRACT_RELATIONS_PROMPT.replace("USER_ID", user_identity)
            # Add the custom prompt line if configured
            system_content = system_content.replace("CUSTOM_PROMPT", f"4. {self.config.graph_store.custom_prompt}")
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": data},
            ]
        else:
            system_content = EXTRACT_RELATIONS_PROMPT.replace("USER_ID", user_identity)
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"List of entities: {list(entity_type_map.keys())}. \n\nText: {data}"},
            ]

        _tools = [RELATIONS_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [RELATIONS_STRUCT_TOOL]

        extracted_entities = self.llm.generate_response(
            messages=messages,
            tools=_tools,
        )

        entities = []
        if extracted_entities.get("tool_calls"):
            entities = extracted_entities["tool_calls"][0].get("arguments", {}).get("entities", [])
        
        if isinstance(entities, str):
            entities = safe_json_loads(entities)

        try:
            entities = self._remove_spaces_from_entities(entities)
        except Exception as e:
            logger.exception(f"Error removing spaces from entities: {e}, entities={entities}, extracted_entities={extracted_entities}")
            return []

        logger.debug(f"Extracted entities: {entities}")
        return entities

    def _search_graph_db(self, node_list, filters, limit=100, threshold=None):
        """Search similar nodes among and their respective incoming and outgoing relations."""
        result_relations = []
        
        threshold_val = threshold if threshold else self.threshold

        # Build WHERE conditions
        where_conditions = ["n.user_id = $user_id"]
        if filters.get("agent_id"):
            where_conditions.append("n.agent_id = $agent_id")
        if filters.get("run_id"):
            where_conditions.append("n.run_id = $run_id")
        where_clause = " AND ".join(where_conditions)
        
        # Build WHERE conditions for related nodes
        related_where = ["m.user_id = $user_id"]
        if filters.get("agent_id"):
            related_where.append("m.agent_id = $agent_id")
        if filters.get("run_id"):
            related_where.append("m.run_id = $run_id")
        related_where_clause = " AND ".join(related_where)

        for node in node_list:
            n_embedding = self.embedding_model.embed(node)

            matches = self._search_embeddings(
                n_embedding,
                filters=filters,
                limit=limit,
                threshold=threshold_val,
            )
            if not matches:
                continue

            node_names = [row.get("node_name") for row in matches if row.get("node_name")]
            similarity_map = {row.get("node_name"): float(row["similarity"]) for row in matches if row.get("node_name")}

            # Search each matched node individually to avoid IN list issues
            for node_name in node_names:
                params = {"user_id": filters["user_id"], "node_name": node_name}
                if filters.get("agent_id"):
                    params["agent_id"] = filters["agent_id"]
                if filters.get("run_id"):
                    params["run_id"] = filters["run_id"]

                # Search for outgoing relationships
                cypher_outgoing = f"""
                MATCH (n:{self.node_label})
                WHERE {where_clause} AND n.name = $node_name
                MATCH (n)-[r]->(m:{self.node_label})
                WHERE {related_where_clause}
                RETURN 
                    n.name AS source,
                    id(n) AS source_id,
                    type(r) AS relationship,
                    id(r) AS relation_id,
                    m.name AS destination,
                    id(m) AS destination_id,
                    0.0 AS similarity
                """
                
                # Search for incoming relationships
                cypher_incoming = f"""
                MATCH (n:{self.node_label})
                WHERE {where_clause} AND n.name = $node_name
                MATCH (m:{self.node_label})-[r]->(n)
                WHERE {related_where_clause}
                RETURN 
                    m.name AS source,
                    id(m) AS source_id,
                    type(r) AS relationship,
                    id(r) AS relation_id,
                    n.name AS destination,
                    id(n) AS destination_id,
                    0.0 AS similarity
                """

                try:
                    outgoing_results = self._execute_cypher(cypher_outgoing, parameters=params)
                    incoming_results = self._execute_cypher(cypher_incoming, parameters=params)

                    for item in outgoing_results:
                        item["similarity"] = similarity_map.get(node_name, 0.0)
                    for item in incoming_results:
                        item["similarity"] = similarity_map.get(node_name, 0.0)
                    
                    all_results = outgoing_results + incoming_results
                    # Sort by similarity and limit
                    all_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
                    result_relations.extend(all_results[:limit])
                except Exception as e:
                    logger.warning(f"Error searching graph db for node {node}: {e}")
                    continue

        return result_relations

    def _get_delete_entities_from_search_output(self, search_output, data, filters):
        """Get the entities to be deleted from the search output."""
        search_output_string = format_entities(search_output)

        # Compose user identification string for prompt
        user_identity = f"user_id: {filters['user_id']}"
        if filters.get("agent_id"):
            user_identity += f", agent_id: {filters['agent_id']}"
        if filters.get("run_id"):
            user_identity += f", run_id: {filters['run_id']}"

        system_prompt, user_prompt = get_delete_messages(search_output_string, data, user_identity)

        _tools = [DELETE_MEMORY_TOOL_GRAPH]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [
                DELETE_MEMORY_STRUCT_TOOL_GRAPH,
            ]

        memory_updates = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=_tools,
        )

        to_be_deleted = []
        for item in memory_updates.get("tool_calls", []):
            if item.get("name") == "delete_graph_memory":
                to_be_deleted.append(item.get("arguments"))
        # Clean entities formatting
        to_be_deleted = self._remove_spaces_from_entities(to_be_deleted)
        logger.debug(f"Deleted relationships: {to_be_deleted}")
        return to_be_deleted

    def _delete_entities(self, to_be_deleted, filters):
        """Delete the entities from the graph."""
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)
        run_id = filters.get("run_id", None)
        results = []

        for item in to_be_deleted:
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            params = {
                "source_name": source,
                "dest_name": destination,
                "user_id": user_id,
            }
            
            # Build WHERE conditions
            source_where = ["n.name = $source_name", "n.user_id = $user_id"]
            dest_where = ["m.name = $dest_name", "m.user_id = $user_id"]
            
            if agent_id:
                source_where.append("n.agent_id = $agent_id")
                dest_where.append("m.agent_id = $agent_id")
                params["agent_id"] = agent_id
            if run_id:
                source_where.append("n.run_id = $run_id")
                dest_where.append("m.run_id = $run_id")
                params["run_id"] = run_id
                
            source_where_clause = " AND ".join(source_where)
            dest_where_clause = " AND ".join(dest_where)

            # Delete the specific relationship between nodes
            # Note: We need to capture properties before DELETE
            cypher = f"""
            MATCH (n:{self.node_label})-[r:{relationship}]->(m:{self.node_label})
            WHERE {source_where_clause} AND {dest_where_clause}
            WITH n.name AS source_name, type(r) AS rel_type, m.name AS target_name, r
            DELETE r
            RETURN source_name AS source, rel_type AS relationship, target_name AS target
            """

            try:
                result = self._execute_cypher(cypher, parameters=params)
                results.append(result)
            except Exception as e:
                logger.warning(f"Error deleting entity: {e}")
                results.append([])

        return results

    def _add_entities(self, to_be_added, filters, entity_type_map):
        """Add the new entities to the graph. Merge the nodes if they already exist."""
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)
        run_id = filters.get("run_id", None)
        results = []
        
        for item in to_be_added:
            # entities
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            # embeddings
            source_embedding = self.embedding_model.embed(source)
            dest_embedding = self.embedding_model.embed(destination)

            # search for the nodes with the closest embeddings
            source_node_search_result = self._search_source_node(source_embedding, filters, threshold=self.threshold)
            destination_node_search_result = self._search_destination_node(dest_embedding, filters, threshold=self.threshold)

            # Build base properties
            base_props = {"user_id": user_id}
            if agent_id:
                base_props["agent_id"] = agent_id
            if run_id:
                base_props["run_id"] = run_id

            params = {
                "source_name": source,
                "dest_name": destination,
                "relationship_name": relationship,
                "user_id": user_id,
            }
            if agent_id:
                params["agent_id"] = agent_id
            if run_id:
                params["run_id"] = run_id

            # Build property strings for MERGE
            source_merge_props = ["name: $source_name", "user_id: $user_id"]
            dest_merge_props = ["name: $dest_name", "user_id: $user_id"]
            if agent_id:
                source_merge_props.append("agent_id: $agent_id")
                dest_merge_props.append("agent_id: $agent_id")
            if run_id:
                source_merge_props.append("run_id: $run_id")
                dest_merge_props.append("run_id: $run_id")
            
            source_merge_str = ", ".join(source_merge_props)
            dest_merge_str = ", ".join(dest_merge_props)

            if not destination_node_search_result and source_node_search_result:
                # Convert string id to int for Apache AGE graphid comparison
                source_id = int(source_node_search_result[0].get("id"))
                params["source_id"] = source_id
                
                # Apache AGE: MERGE nodes/relationships without properties, then SET them
                cypher = f"""
                MATCH (source:{self.node_label})
                WHERE id(source) = $source_id
                SET source.mentions = COALESCE(source.mentions, 0) + 1
                WITH source
                MERGE (destination:{self.node_label} {{{dest_merge_str}}})
                SET destination.mentions = COALESCE(destination.mentions, 0) + 1,
                    destination.created = COALESCE(destination.created, timestamp())
                WITH source, destination
                MERGE (source)-[r:{self.rel_label}]->(destination)
                SET r.name = $relationship_name,
                    r.created = COALESCE(r.created, timestamp()),
                    r.mentions = COALESCE(r.mentions, 0) + 1
                RETURN
                    source.name AS source,
                    id(source) AS source_id,
                    type(r) AS relationship,
                    destination.name AS target,
                    id(destination) AS destination_id
                """
            elif destination_node_search_result and not source_node_search_result:
                # Convert string id to int for Apache AGE graphid comparison
                destination_id = int(destination_node_search_result[0].get("id"))
                params["destination_id"] = destination_id
                
                # Apache AGE: MERGE nodes/relationships without properties, then SET them
                cypher = f"""
                MATCH (destination:{self.node_label})
                WHERE id(destination) = $destination_id
                SET destination.mentions = COALESCE(destination.mentions, 0) + 1
                WITH destination
                MERGE (source:{self.node_label} {{{source_merge_str}}})
                SET source.mentions = COALESCE(source.mentions, 0) + 1,
                    source.created = COALESCE(source.created, timestamp())
                WITH source, destination
                MERGE (source)-[r:{self.rel_label}]->(destination)
                SET r.name = $relationship_name,
                    r.created = COALESCE(r.created, timestamp()),
                    r.mentions = COALESCE(r.mentions, 0) + 1
                RETURN
                    source.name AS source,
                    id(source) AS source_id,
                    type(r) AS relationship,
                    destination.name AS target,
                    id(destination) AS destination_id
                """
            elif source_node_search_result and destination_node_search_result:
                # Convert string ids to int for Apache AGE graphid comparison
                source_id = int(source_node_search_result[0].get("id"))
                destination_id = int(destination_node_search_result[0].get("id"))
                params["source_id"] = source_id
                params["destination_id"] = destination_id
                
                # Apache AGE: MERGE relationship without properties, then SET them
                cypher = f"""
                MATCH (source:{self.node_label})
                WHERE id(source) = $source_id
                SET source.mentions = COALESCE(source.mentions, 0) + 1
                WITH source
                MATCH (destination:{self.node_label})
                WHERE id(destination) = $destination_id
                SET destination.mentions = COALESCE(destination.mentions, 0) + 1
                WITH source, destination
                MERGE (source)-[r:{self.rel_label}]->(destination)
                SET r.name = $relationship_name,
                    r.created = COALESCE(r.created, timestamp()),
                    r.updated = timestamp(),
                    r.mentions = COALESCE(r.mentions, 0) + 1
                RETURN
                    source.name AS source,
                    id(source) AS source_id,
                    type(r) AS relationship,
                    destination.name AS target,
                    id(destination) AS destination_id
                """
            else:
                # Apache AGE: MERGE nodes/relationships without properties, then SET them
                cypher = f"""
                MERGE (source:{self.node_label} {{{source_merge_str}}})
                SET source.mentions = COALESCE(source.mentions, 0) + 1,
                    source.created = COALESCE(source.created, timestamp())
                WITH source
                MERGE (destination:{self.node_label} {{{dest_merge_str}}})
                SET destination.mentions = COALESCE(destination.mentions, 0) + 1,
                    destination.created = COALESCE(destination.created, timestamp())
                WITH source, destination
                MERGE (source)-[rel:{self.rel_label}]->(destination)
                SET rel.name = $relationship_name,
                    rel.created = COALESCE(rel.created, timestamp()),
                    rel.mentions = COALESCE(rel.mentions, 0) + 1
                RETURN
                    source.name AS source,
                    id(source) AS source_id,
                    type(rel) AS relationship,
                    destination.name AS target,
                    id(destination) AS destination_id
                """

            try:
                logger.debug(f"Executing add entity query for {source} -> {destination}")
                result = self._execute_cypher(cypher, parameters=params)
                logger.debug(f"Add entity result: {result}")
                source_ids = set()
                dest_ids = set()

                for row in result or []:
                    if row.get("source_id") is not None:
                        source_ids.add(int(row["source_id"]))
                    if row.get("destination_id") is not None:
                        dest_ids.add(int(row["destination_id"]))

                if not source_ids and source_node_search_result:
                    source_ids.add(int(source_node_search_result[0].get("id")))
                if not dest_ids and destination_node_search_result:
                    dest_ids.add(int(destination_node_search_result[0].get("id")))

                for source_id in source_ids:
                    self._upsert_embedding(
                        node_graphid=source_id,
                        node_name=source,
                        embedding=source_embedding,
                        filters=filters,
                    )
                for dest_id in dest_ids:
                    self._upsert_embedding(
                        node_graphid=dest_id,
                        node_name=destination,
                        embedding=dest_embedding,
                        filters=filters,
                    )
                results.append(result)
            except Exception as e:
                logger.warning(f"Error adding entity {source} -> {destination}: {e}")
                results.append([])

        return results

    def _remove_spaces_from_entities(self, entity_list):
        for item in entity_list:
            item["source"] = item["source"].lower().replace(" ", "_")
            item["relationship"] = sanitize_relationship_for_cypher(item["relationship"].lower().replace(" ", "_"))
            item["destination"] = item["destination"].lower().replace(" ", "_")
        return entity_list

    def _search_source_node(self, source_embedding, filters, threshold=0.9):
        """Search for source nodes with similar embeddings."""
        try:
            matches = self._search_embeddings(
                source_embedding,
                filters=filters,
                limit=1,
                threshold=threshold,
            )
            if not matches:
                return []
            match = matches[0]
            return [
                {
                    "id": str(int(match["node_graphid"])),
                    "source_similarity": float(match["similarity"]),
                }
            ]
        except Exception as e:
            logger.warning(f"Error searching source node: {e}")
            return []

    def _search_destination_node(self, destination_embedding, filters, threshold=0.9):
        """Search for destination nodes with similar embeddings."""
        try:
            matches = self._search_embeddings(
                destination_embedding,
                filters=filters,
                limit=1,
                threshold=threshold,
            )
            if not matches:
                return []
            match = matches[0]
            return [
                {
                    "id": str(int(match["node_graphid"])),
                    "destination_similarity": float(match["similarity"]),
                }
            ]
        except Exception as e:
            logger.warning(f"Error searching destination node: {e}")
            return []

    def reset(self):
        """Reset the graph by clearing all nodes and relationships."""
        logger.warning("Clearing graph...")
        cypher_query = f"""
        MATCH (n:{self.node_label})
        DETACH DELETE n
        """
        result = self._execute_cypher(cypher_query)
        self._delete_embeddings(graph_only=True)
        return result
    
    def __del__(self):
        """Close the database connection when the object is destroyed."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
