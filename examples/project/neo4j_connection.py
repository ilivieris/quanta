import re
import time
import logging
from typing import List, Dict, Any
from pathlib import Path
from neo4j import GraphDatabase

class Neo4jConnection:
    def __init__(self, uri: str, user: str, pwd: str) -> None:
        '''
            Connection with Neo4j

            Parameters
            ----------
            uri: (str) 
                URL
            user: (str)
                username 
            pwd: (str)
                password
        '''
        self._uri = uri
        self._user = user
        self._pwd = pwd
        self._driver = None

        self.node_properties_query = """
            MATCH (n)
            UNWIND labels(n) AS nodeLabel
            WITH nodeLabel, keys(n) AS props
            UNWIND (CASE WHEN size(props) = 0 THEN [null] ELSE props END) AS property
            WITH nodeLabel, collect(DISTINCT property) AS collectedProps
            WITH nodeLabel AS nodeLabels,
                 [p IN collectedProps WHERE p IS NOT NULL | {property: p, type: 'String'}] AS properties
            RETURN {labels: nodeLabels, properties: properties} AS output
            """

        self.rel_properties_query = """
            MATCH ()-[r]->()
            WITH type(r) AS relType, keys(r) AS props
            UNWIND (CASE WHEN size(props) = 0 THEN [null] ELSE props END) AS property
            WITH relType, collect(DISTINCT property) AS collectedProps
            WITH relType,
                 [p IN collectedProps WHERE p IS NOT NULL | {property: p, type: 'String'}] AS properties
            RETURN {type: relType, properties: properties} AS output
            """

        self.rel_query = """
            MATCH (a)-[r]->(b)
            WITH labels(a)[0] AS startLabel, type(r) AS relType, labels(b)[0] AS endLabel
            WHERE startLabel IS NOT NULL AND endLabel IS NOT NULL
            RETURN DISTINCT {start: startLabel, type: relType, end: endLabel} AS output
            """

        try:
            self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._pwd))
            print("[INFO] Connection to Neo4j established")
        except Exception as e:
            print("[ERROR] Connection to Neo4j not established")
            print(" > ", e)
            raise e


    def close(self)->None:
        '''
            Close connection with Neo4j
        '''
        if self._driver is not None:
            self._driver.close()
            print("[INFO] Connection with Neo4j is terminated")
            
    def query(self, query=None, parameters=None, db=None)->str:
        '''
            Conduct a query to the database

            Parameters
            ----------
            query: (str)
                user query
            parameters: (str)
                parameters

            Returns
            -------
            Response from the query (str)
        '''
        assert self._driver is not None, "Driver not initialized!"
        session = None
        response = None
        success = False
        for i in range(1):
            try: 
                session = self._driver.session(database=db) if db is not None else self._driver.session() 
                response = list(session.run(query, parameters))
                if i > 0: 
                    print("[INFO] Query successfully conducted")
                success = True
                break
            except Exception as e:
                # pass
                print("[ERROR] Query failed")
                print(f" > Query: {query}\n -> {e}")
            finally: 
                if session is not None:
                    session.close()
        if not success:
            print("[WARNING] Query was not executed") 
        return response

    def clean_base(self)->None:
        '''
            Remove all nodes/relationships from the database
        '''
        self.query("MATCH (n) DETACH DELETE n")
        print('[INFO] All items in Neo4j were deleted')


    def get_schema(self) -> str:
        """
        Refreshes the Neo4j graph schema information, excluding the VIEWER node and its relationships.

        Returns
        -------
        DB schema (str)
        """
        # Fetch raw data from queries
        node_properties = [el["output"] for el in (self.query(self.node_properties_query) or [])]
        rel_properties = [el["output"] for el in (self.query(self.rel_properties_query) or [])]
        relationships = [el["output"] for el in (self.query(self.rel_query) or [])]

        # Exclude VIEWER node properties
        node_properties = [el for el in node_properties if el["labels"] != "VIEWER"]
        # Exclude relationships involving VIEWER
        relationships = [el for el in relationships if el["start"] != "VIEWER" and el["end"] != "VIEWER"]

        self.schema = f"""
        Node properties are the following:
        {node_properties}
        Relationship properties are the following:
        {rel_properties}
        The relationships are the following:
        {[f"(:{el['start']})-[:{el['type']}]->(:{el['end']})" for el in relationships]}
        """

        return self.schema
    
    
    
def neo4j_connection(neo4j_settings: dict = None, clean_graph: bool = False):
    """
    Establishes a connection to a Neo4j database and optionally cleans the graph.

    Parameters:
    neo4j_settings (dict): A dictionary containing the connection settings for Neo4j.
                           Expected keys: "connection_url", "username", "password".
    clean_graph (bool): A flag indicating whether to clean the graph after establishing the connection.

    Returns:
    Neo4jConnection: An instance of the Neo4jConnection class.

    Raises:
    Exception: If there is an error while connecting to Neo4j.
    """
    try:
        graph = Neo4jConnection(
            uri=neo4j_settings["connection_url"],
            user=neo4j_settings["username"],
            pwd=neo4j_settings["password"],
        )
        
        if clean_graph: 
            graph.clean_base()
        
        return graph
    except Exception as e:
        raise e
    
    

class Neo4jExtractor:
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        """
        Initialize Neo4j connection
        
        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            username: Neo4j username
            password: Neo4j password
            database: Database name (default: "neo4j")
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        self.cypher_queries = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)

    def close(self):
        """Close the Neo4j driver connection"""
        if self.driver:
            self.driver.close()

    def extract_constraints(self) -> List[str]:
        """Extract all constraints from the database"""
        with self.driver.session(database=self.database) as session:
            result = session.run("SHOW CONSTRAINTS")
            constraints = []
            
            for record in result:
                constraint_type = record.get("type", "")
                entity_type = record.get("entityType", "")
                labels = record.get("labelsOrTypes", [])
                properties = record.get("properties", [])
                name = record.get("name", "")
                
                if constraint_type == "UNIQUENESS":
                    if entity_type == "NODE":
                        label = labels[0] if labels else ""
                        prop = properties[0] if properties else ""
                        cypher = f"CREATE CONSTRAINT {name} FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
                        constraints.append(cypher)
                elif constraint_type == "NODE_PROPERTY_EXISTENCE":
                    label = labels[0] if labels else ""
                    prop = properties[0] if properties else ""
                    cypher = f"CREATE CONSTRAINT {name} FOR (n:{label}) REQUIRE n.{prop} IS NOT NULL"
                    constraints.append(cypher)
                elif constraint_type == "RELATIONSHIP_PROPERTY_EXISTENCE":
                    rel_type = labels[0] if labels else ""
                    prop = properties[0] if properties else ""
                    cypher = f"CREATE CONSTRAINT {name} FOR ()-[r:{rel_type}]-() REQUIRE r.{prop} IS NOT NULL"
                    constraints.append(cypher)
                    
            return constraints

    def extract_indexes(self) -> List[str]:
        """Extract all indexes from the database"""
        with self.driver.session(database=self.database) as session:
            result = session.run("SHOW INDEXES")
            indexes = []
            
            for record in result:
                index_type = record.get("type", "")
                entity_type = record.get("entityType", "")
                labels = record.get("labelsOrTypes", [])
                properties = record.get("properties", [])
                name = record.get("name", "")
                
                if index_type in ["BTREE", "RANGE"] and entity_type == "NODE":
                    label = labels[0] if labels else ""
                    if len(properties) == 1:
                        prop = properties[0]
                        cypher = f"CREATE INDEX {name} FOR (n:{label}) ON (n.{prop})"
                    else:
                        props = ", ".join([f"n.{prop}" for prop in properties])
                        cypher = f"CREATE INDEX {name} FOR (n:{label}) ON ({props})"
                    indexes.append(cypher)
                elif index_type == "TEXT":
                    label = labels[0] if labels else ""
                    prop = properties[0] if properties else ""
                    cypher = f"CREATE TEXT INDEX {name} FOR (n:{label}) ON (n.{prop})"
                    indexes.append(cypher)
                    
            return indexes

    def sanitize_value(self, value: Any) -> str:
        """Sanitize values for Cypher query generation"""
        if value is None:
            return "null"
        elif isinstance(value, str):
            # Handle multiline strings and special characters
            escaped = (value
                      .replace("\\", "\\\\")  # Escape backslashes first
                      .replace("\n", "\\n")   # Escape newlines
                      .replace("\r", "\\r")   # Escape carriage returns
                      .replace("\t", "\\t")   # Escape tabs
                      .replace("'", "\\'")    # Escape single quotes
                      .replace('"', '\\"'))   # Escape double quotes
            return f"'{escaped}'"
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, list):
            sanitized_items = [self.sanitize_value(item) for item in value]
            return f"[{', '.join(sanitized_items)}]"
        elif isinstance(value, dict):
            sanitized_items = []
            for k, v in value.items():
                # Sanitize property keys as well
                clean_key = str(k).replace(" ", "_").replace("-", "_")
                # Ensure key is valid identifier
                if clean_key and (clean_key[0].isalpha() or clean_key[0] == '_'):
                    sanitized_items.append(f"{clean_key}: {self.sanitize_value(v)}")
            return f"{{{', '.join(sanitized_items)}}}"
        else:
            # Convert to string and sanitize
            str_value = str(value)
            escaped = (str_value
                      .replace("\\", "\\\\")
                      .replace("\n", "\\n")
                      .replace("\r", "\\r")
                      .replace("\t", "\\t")
                      .replace("'", "\\'")
                      .replace('"', '\\"'))
            return f"'{escaped}'"

    def format_properties(self, properties: Dict[str, Any]) -> str:
        """Format node/relationship properties for Cypher"""
        if not properties:
            return ""
        
        prop_strings = []
        for key, value in properties.items():
            # Sanitize property keys
            clean_key = str(key).replace(" ", "_").replace("-", "_")
            # Ensure key is valid identifier
            if clean_key and (clean_key[0].isalpha() or clean_key[0] == '_'):
                prop_strings.append(f"{clean_key}: {self.sanitize_value(value)}")
        
        return f"{{{', '.join(prop_strings)}}}"

    def extract_nodes(self, batch_size: int = 1000) -> List[str]:
        """Extract all nodes from the database"""
        with self.driver.session(database=self.database) as session:
            # Get all unique labels
            labels_result = session.run("CALL db.labels()")
            labels = [record["label"] for record in labels_result]
            
            node_queries = []
            
            for label in labels:
                self.logger.info(f"Extracting nodes with label: {label}")
                
                # Process nodes in batches
                skip = 0
                while True:
                    query = f"""
                    MATCH (n:{label})
                    RETURN n
                    SKIP {skip} LIMIT {batch_size}
                    """
                    
                    result = session.run(query)
                    records = list(result)
                    
                    if not records:
                        break
                    
                    batch_queries = []
                    for record in records:
                        node = record["n"]
                        node_labels = ":".join(node.labels)
                        properties = dict(node)
                        
                        if properties:
                            props_str = self.format_properties(properties)
                            if props_str:  # Only add if we have valid properties
                                cypher = f"CREATE (:{node_labels} {props_str})"
                            else:
                                cypher = f"CREATE (:{node_labels})"
                        else:
                            cypher = f"CREATE (:{node_labels})"
                        
                        batch_queries.append(cypher)
                    
                    # Combine batch queries for better performance
                    if batch_queries:
                        combined_query = ";\n".join(batch_queries) + ";"
                        node_queries.append(combined_query)
                    
                    skip += batch_size
                    
            return node_queries

    def extract_relationships(self, batch_size: int = 1000) -> List[str]:
        """Extract all relationships from the database"""
        with self.driver.session(database=self.database) as session:
            # Get all unique relationship types
            rel_types_result = session.run("CALL db.relationshipTypes()")
            rel_types = [record["relationshipType"] for record in rel_types_result]
            
            relationship_queries = []
            
            for rel_type in rel_types:
                self.logger.info(f"Extracting relationships of type: {rel_type}")
                
                skip = 0
                while True:
                    query = f"""
                    MATCH (a)-[r:{rel_type}]->(b)
                    RETURN a, r, b, labels(a) as start_labels, labels(b) as end_labels
                    SKIP {skip} LIMIT {batch_size}
                    """
                    
                    result = session.run(query)
                    records = list(result)
                    
                    if not records:
                        break
                    
                    batch_queries = []
                    for record in records:
                        start_node = record["a"]
                        relationship = record["r"]
                        end_node = record["b"]
                        start_labels = record["start_labels"]
                        end_labels = record["end_labels"]
                        
                        # Create match patterns for start and end nodes
                        start_props = self.format_properties(dict(start_node))
                        end_props = self.format_properties(dict(end_node))
                        start_label_str = ":".join(start_labels)
                        end_label_str = ":".join(end_labels)
                        
                        rel_props = dict(relationship)
                        rel_props_str = self.format_properties(rel_props) if rel_props else ""
                        
                        # Build the cypher query
                        start_match = f"(a:{start_label_str}"
                        if start_props:
                            start_match += f" {start_props}"
                        start_match += ")"
                        
                        end_match = f"(b:{end_label_str}"
                        if end_props:
                            end_match += f" {end_props}"
                        end_match += ")"
                        
                        if rel_props_str:
                            cypher = f"MATCH {start_match}, {end_match}\nCREATE (a)-[:{rel_type} {rel_props_str}]->(b)"
                        else:
                            cypher = f"MATCH {start_match}, {end_match}\nCREATE (a)-[:{rel_type}]->(b)"
                        
                        batch_queries.append(cypher)
                    
                    if batch_queries:
                        combined_query = ";\n".join(batch_queries) + ";"
                        relationship_queries.append(combined_query)
                    
                    skip += batch_size
                    
            return relationship_queries

    def extract_all(self, output_file: str = "neo4j_export.cypher", batch_size: int = 1000):
        """Extract the entire database and save to a Cypher file"""
        try:
            self.logger.info("Starting database extraction...")
            
            # Extract schema
            self.logger.info("Extracting constraints...")
            constraints = self.extract_constraints()
            
            self.logger.info("Extracting indexes...")
            indexes = self.extract_indexes()
            
            # Extract data
            self.logger.info("Extracting nodes...")
            nodes = self.extract_nodes(batch_size)
            
            self.logger.info("Extracting relationships...")
            relationships = self.extract_relationships(batch_size)
            
            # Write to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("// Neo4j Database Export\n")
                f.write("// Generated by Neo4jExtractor\n\n")
                
                if constraints:
                    f.write("// ===== CONSTRAINTS =====\n")
                    for constraint in constraints:
                        f.write(f"{constraint};\n")
                    f.write("\n")
                
                if indexes:
                    f.write("// ===== INDEXES =====\n")
                    for index in indexes:
                        f.write(f"{index};\n")
                    f.write("\n")
                
                if nodes:
                    f.write("// ===== NODES =====\n")
                    for node_batch in nodes:
                        f.write(f"{node_batch}\n")
                    f.write("\n")
                
                if relationships:
                    f.write("// ===== RELATIONSHIPS =====\n")
                    for rel_batch in relationships:
                        f.write(f"{rel_batch}\n")
            
            self.logger.info(f"Database exported successfully to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error during extraction: {str(e)}")
            raise
        finally:
            self.close()

    def get_database_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the database"""
        with self.driver.session(database=self.database) as session:
            stats = {}
            
            # Node count
            result = session.run("MATCH (n) RETURN count(n) as node_count")
            stats['node_count'] = result.single()['node_count']
            
            # Relationship count
            result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
            stats['relationship_count'] = result.single()['rel_count']
            
            # Labels
            result = session.run("CALL db.labels()")
            stats['labels'] = [record['label'] for record in result]
            
            # Relationship types
            result = session.run("CALL db.relationshipTypes()")
            stats['relationship_types'] = [record['relationshipType'] for record in result]
            
            return stats
        
        
class Neo4jImporter:
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        """
        Initialize Neo4j connection for importing

        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            username: Neo4j username
            password: Neo4j password
            database: Database name (default: "neo4j")
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger = logging.getLogger(__name__)

    def close(self):
        """Close the Neo4j driver connection"""
        if self.driver:
            self.driver.close()

    def clear_database(self, confirm: bool = False):
        """
        Clear all data from the target database
        WARNING: This will delete all nodes and relationships!

        Args:
            confirm: Set to True to confirm deletion
        """
        if not confirm:
            self.logger.warning(
                "Database clear was called but not confirmed. Set confirm=True to proceed."
            )
            return

        with self.driver.session(database=self.database) as session:
            self.logger.info("Clearing database - removing all relationships...")
            session.run("MATCH ()-[r]->() DELETE r")

            self.logger.info("Clearing database - removing all nodes...")
            session.run("MATCH (n) DELETE n")

            # Drop constraints and indexes
            self.logger.info("Dropping constraints...")
            try:
                result = session.run("SHOW CONSTRAINTS")
                for record in result:
                    constraint_name = record.get("name")
                    if constraint_name:
                        session.run(f"DROP CONSTRAINT {constraint_name}")
            except Exception as e:
                self.logger.warning(f"Error dropping constraints: {e}")

            self.logger.info("Dropping indexes...")
            try:
                result = session.run("SHOW INDEXES")
                for record in result:
                    index_name = record.get("name")
                    index_type = record.get("type", "")
                    # Don't drop system indexes
                    if index_name and not index_name.startswith("__"):
                        session.run(f"DROP INDEX {index_name}")
            except Exception as e:
                self.logger.warning(f"Error dropping indexes: {e}")

            self.logger.info("Database cleared successfully!")

    def parse_cypher_file(self, file_path: str) -> dict:
        """
        Parse the exported Cypher file and categorize queries

        Args:
            file_path: Path to the .cypher file

        Returns:
            Dictionary with categorized queries
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Cypher file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        queries = {"constraints": [], "indexes": [], "nodes": [], "relationships": []}

        # Split content into sections
        sections = {
            "constraints": re.search(
                r"// ===== CONSTRAINTS =====\n(.*?)(?=// ===== |\Z)", content, re.DOTALL
            ),
            "indexes": re.search(
                r"// ===== INDEXES =====\n(.*?)(?=// ===== |\Z)", content, re.DOTALL
            ),
            "nodes": re.search(
                r"// ===== NODES =====\n(.*?)(?=// ===== |\Z)", content, re.DOTALL
            ),
            "relationships": re.search(
                r"// ===== RELATIONSHIPS =====\n(.*?)(?=// ===== |\Z)",
                content,
                re.DOTALL,
            ),
        }

        for section_name, section_match in sections.items():
            if section_match:
                section_content = section_match.group(1).strip()
                if section_content:
                    # Parse queries more carefully to handle multi-line queries
                    queries[section_name] = self.parse_section_queries(
                        section_content, section_name
                    )

        return queries

    def parse_section_queries(
        self, section_content: str, section_name: str
    ) -> List[str]:
        """
        Parse queries from a section, handling multi-line queries properly

        Args:
            section_content: Raw section content
            section_name: Name of the section for logging

        Returns:
            List of valid Cypher queries
        """
        queries = []
        current_query = ""
        lines = section_content.split("\n")

        for line in lines:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("//"):
                continue

            current_query += line + "\n"

            # Check if this completes a query (ends with semicolon)
            if line.endswith(";"):
                query = current_query.strip()
                if query.endswith(";"):
                    query = query[:-1]  # Remove trailing semicolon

                # Validate query starts with expected keywords
                if self.is_valid_cypher_query(query, section_name):
                    queries.append(query)
                else:
                    self.logger.warning(
                        f"Skipping invalid query in {section_name}: {query[:100]}..."
                    )

                current_query = ""

        # Handle any remaining query without semicolon
        if current_query.strip():
            query = current_query.strip()
            if self.is_valid_cypher_query(query, section_name):
                queries.append(query)
            else:
                self.logger.warning(
                    f"Skipping invalid query in {section_name}: {query[:100]}..."
                )

        return queries

    def is_valid_cypher_query(self, query: str, section_name: str) -> bool:
        """
        Validate if a query is a valid Cypher query for the given section

        Args:
            query: The query to validate
            section_name: Section name (constraints, indexes, nodes, relationships)

        Returns:
            True if valid, False otherwise
        """
        if not query or len(query.strip()) < 5:
            return False

        query_upper = query.strip().upper()

        if section_name == "constraints":
            return query_upper.startswith(
                "CREATE CONSTRAINT"
            ) or query_upper.startswith("DROP CONSTRAINT")
        elif section_name == "indexes":
            return (
                query_upper.startswith("CREATE INDEX")
                or query_upper.startswith("CREATE TEXT INDEX")
                or query_upper.startswith("DROP INDEX")
            )
        elif section_name == "nodes":
            return query_upper.startswith("CREATE (")
        elif section_name == "relationships":
            return query_upper.startswith("MATCH (") and "CREATE (" in query_upper

        return True

    def execute_query_with_retry(
        self, session, query: str, max_retries: int = 3, delay: float = 1.0
    ) -> bool:
        """
        Execute a query with retry logic

        Args:
            session: Neo4j session
            query: Cypher query to execute
            max_retries: Maximum number of retry attempts
            delay: Delay between retries in seconds

        Returns:
            True if successful, False otherwise
        """
        for attempt in range(max_retries + 1):
            try:
                session.run(query)
                return True
            except Exception as e:
                if attempt < max_retries:
                    self.logger.warning(
                        f"Query failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}"
                    )
                    self.logger.warning(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    self.logger.error(
                        f"Query failed after {max_retries + 1} attempts: {str(e)}"
                    )
                    self.logger.error(f"Failed query: {query[:200]}...")
                    return False
        return False

    def execute_queries_batch(
        self, session, queries: List[str], batch_name: str, batch_size: int = 100
    ):
        """
        Execute a list of queries in batches

        Args:
            session: Neo4j session
            queries: List of Cypher queries
            batch_name: Name for logging purposes
            batch_size: Number of queries per batch
        """
        if not queries:
            self.logger.info(f"No {batch_name} queries to execute")
            return

        self.logger.info(f"Executing {len(queries)} {batch_name} queries...")

        successful = 0
        failed = 0

        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]
            self.logger.info(
                f"Processing {batch_name} batch {i // batch_size + 1}/{(len(queries) + batch_size - 1) // batch_size}"
            )

            for query in batch:
                if self.execute_query_with_retry(session, query):
                    successful += 1
                else:
                    failed += 1

        self.logger.info(
            f"{batch_name} execution completed: {successful} successful, {failed} failed"
        )

    def import_database(
        self, cypher_file: str, clear_target: bool = False, batch_size: int = 100
    ):
        """
        Import the database from a Cypher file

        Args:
            cypher_file: Path to the exported .cypher file
            clear_target: Whether to clear the target database first
            batch_size: Number of queries to execute per batch
        """
        try:
            self.logger.info(f"Starting database import from {cypher_file}")

            # Clear database if requested
            if clear_target:
                self.clear_database(confirm=True)

            # Parse the cypher file
            self.logger.info("Parsing Cypher file...")
            queries = self.parse_cypher_file(cypher_file)

            # Log statistics
            self.logger.info(
                f"Found queries: "
                f"Constraints: {len(queries['constraints'])}, "
                f"Indexes: {len(queries['indexes'])}, "
                f"Nodes: {len(queries['nodes'])}, "
                f"Relationships: {len(queries['relationships'])}"
            )

            with self.driver.session(database=self.database) as session:
                # Execute in order: constraints -> indexes -> nodes -> relationships

                # 1. Create constraints
                self.execute_queries_batch(
                    session, queries["constraints"], "constraint", 1
                )

                # 2. Create indexes
                self.execute_queries_batch(session, queries["indexes"], "index", 1)

                # Wait for indexes to come online
                if queries["indexes"]:
                    self.logger.info("Waiting for indexes to come online...")
                    time.sleep(5)

                # 3. Create nodes
                self.execute_queries_batch(
                    session, queries["nodes"], "node", batch_size
                )

                # 4. Create relationships
                self.execute_queries_batch(
                    session, queries["relationships"], "relationship", batch_size
                )

            self.logger.info("Database import completed successfully!")

        except Exception as e:
            self.logger.error(f"Error during import: {str(e)}")
            raise
        finally:
            self.close()

    def verify_import(self, original_stats: dict = None) -> dict:
        """
        Verify the import by getting database statistics

        Args:
            original_stats: Original database statistics for comparison

        Returns:
            Dictionary with current database statistics
        """
        with self.driver.session(database=self.database) as session:
            stats = {}

            # Node count
            result = session.run("MATCH (n) RETURN count(n) as node_count")
            stats["node_count"] = result.single()["node_count"]

            # Relationship count
            result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
            stats["relationship_count"] = result.single()["rel_count"]

            # Labels
            result = session.run("CALL db.labels()")
            stats["labels"] = [record["label"] for record in result]

            # Relationship types
            result = session.run("CALL db.relationshipTypes()")
            stats["relationship_types"] = [
                record["relationshipType"] for record in result
            ]

            # Constraints count
            result = session.run("SHOW CONSTRAINTS")
            stats["constraint_count"] = len(list(result))

            # Indexes count
            result = session.run("SHOW INDEXES")
            stats["index_count"] = len(
                [r for r in result if not r.get("name", "").startswith("__")]
            )

            self.logger.info("Import Verification:")
            self.logger.info(f"Nodes: {stats['node_count']}")
            self.logger.info(f"Relationships: {stats['relationship_count']}")
            self.logger.info(f"Labels: {len(stats['labels'])}")
            self.logger.info(f"Relationship Types: {len(stats['relationship_types'])}")
            self.logger.info(f"Constraints: {stats['constraint_count']}")
            self.logger.info(f"Indexes: {stats['index_count']}")

            if original_stats:
                self.logger.info("\nComparison with original:")
                for key in ["node_count", "relationship_count", "constraint_count"]:
                    if key in original_stats:
                        original = original_stats[key]
                        current = stats[key]
                        status = "✓" if original == current else "✗"
                        self.logger.info(f"{key}: {original} -> {current} {status}")

            return stats

    def debug_cypher_file(
        self, cypher_file: str, output_file: str = "debug_queries.txt"
    ):
        """
        Debug the Cypher file by analyzing problematic queries

        Args:
            cypher_file: Path to the exported .cypher file
            output_file: Path to output debug information
        """
        try:
            self.logger.info(f"Debugging Cypher file: {cypher_file}")

            with open(cypher_file, "r", encoding="utf-8") as file:
                content = file.read()

            debug_info = []
            debug_info.append("=== CYPHER FILE DEBUG ANALYSIS ===\n")

            # Check for common issues
            lines = content.split("\n")
            problematic_lines = []

            for i, line in enumerate(lines, 1):
                line_stripped = line.strip()
                if line_stripped and not line_stripped.startswith("//"):
                    # Check for unescaped characters
                    if "'" in line_stripped and not line_stripped.count("'") % 2 == 0:
                        problematic_lines.append(f"Line {i}: Unmatched single quotes")

                    # Check for newlines in properties
                    if (
                        "\\n" not in line_stripped
                        and "\n" in line_stripped
                        and "CREATE" in line_stripped
                    ):
                        problematic_lines.append(
                            f"Line {i}: Possible unescaped newline"
                        )

                    # Check for non-Cypher content
                    if not any(
                        keyword in line_stripped.upper()
                        for keyword in [
                            "CREATE",
                            "MATCH",
                            "MERGE",
                            "SET",
                            "DELETE",
                            "RETURN",
                            "WITH",
                            "//",
                        ]
                    ):
                        if len(line_stripped) > 10:  # Ignore short lines
                            problematic_lines.append(
                                f"Line {i}: Not valid Cypher: {line_stripped[:50]}..."
                            )

            if problematic_lines:
                debug_info.append("PROBLEMATIC LINES FOUND:\n")
                debug_info.extend(problematic_lines)
                debug_info.append("\n")
            else:
                debug_info.append("No obvious syntax issues found in file structure.\n")

            # Try parsing each section
            queries = self.parse_cypher_file(cypher_file)

            debug_info.append("PARSING RESULTS:\n")
            debug_info.append(f"Constraints: {len(queries['constraints'])} queries")
            debug_info.append(f"Indexes: {len(queries['indexes'])} queries")
            debug_info.append(f"Nodes: {len(queries['nodes'])} queries")
            debug_info.append(
                f"Relationships: {len(queries['relationships'])} queries\n"
            )

            # Test a few queries from each section
            with self.driver.session(database=self.database) as session:
                for section_name, section_queries in queries.items():
                    if section_queries:
                        debug_info.append(f"\nTesting first {section_name} query:")
                        test_query = section_queries[0]
                        debug_info.append(f"Query: {test_query[:200]}...")

                        try:
                            # For CREATE queries, use EXPLAIN to test syntax without executing
                            if test_query.strip().upper().startswith("CREATE"):
                                session.run(f"EXPLAIN {test_query}")
                                debug_info.append("✓ Syntax valid")
                            else:
                                debug_info.append(
                                    "? Skipped syntax test for non-CREATE query"
                                )
                        except Exception as e:
                            debug_info.append(f"✗ Syntax error: {str(e)}")

            # Write debug info to file
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(debug_info))

            self.logger.info(f"Debug information written to: {output_file}")

        except Exception as e:
            self.logger.error(f"Error during debug: {str(e)}")
            raise

    def execute_custom_query(self, query: str):
        """Execute a custom Cypher query"""
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            return list(result)