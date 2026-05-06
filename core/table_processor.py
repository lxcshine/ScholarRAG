# -*- coding: utf-8 -*-
"""
Table Processor for Agentic RAG Pipeline
Handles table extraction, schema generation, code execution for table queries.
Integrated with existing ChromaDB vector store and LLM router.
"""

import re
import logging
import pandas as pd
import io
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from langchain_core.documents import Document

from config.settings import settings
from core.llm_router import LLMRouter

logger = logging.getLogger(__name__)


class TableSchema:
    """Represents the global schema of a single table."""

    def __init__(self, table_id: str, name: str, df: pd.DataFrame,
                 source_doc_id: str = "", section: str = "", page: int = 0):
        self.table_id = table_id
        self.name = name
        self.df = df
        self.source_doc_id = source_doc_id
        self.section = section
        self.page = page

    def to_description(self) -> str:
        """Generate a global table description text for embedding."""
        lines = []
        lines.append(f"Table: {self.name}")
        lines.append(f"Source Document: {self.source_doc_id}")
        if self.section:
            lines.append(f"Section: {self.section}")
        lines.append(f"Shape: {self.df.shape[0]} rows x {self.df.shape[1]} columns")
        lines.append("")
        lines.append("Columns:")
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            non_null = self.df[col].notna().sum()
            unique = self.df[col].nunique()
            sample = self.df[col].dropna().head(3).tolist()
            sample_clean = [str(v) if pd.notna(v) else "N/A" for v in sample]
            lines.append(f"  - {col} (dtype: {dtype}, non_null: {non_null}, unique: {unique}, sample: {sample_clean})")
        lines.append("")
        lines.append("Sample Data (first 3 rows):")
        df_clean = self.df.head(3).fillna("N/A")
        lines.append(df_clean.to_string(index=False))
        return "\n".join(lines)

    def to_vector_doc(self) -> Document:
        """Convert to a Document for vector storage."""
        desc = self.to_description()
        return Document(
            page_content=desc,
            metadata={
                "doc_id": self.source_doc_id,
                "table_id": self.table_id,
                "table_name": self.name,
                "section": self.section,
                "page": self.page,
                "chunk_type": "table_schema",
                "is_parent": True,
            }
        )


class TableProcessor:
    """
    Processes tables from PDF documents for Agentic RAG.
    Phase 1: Extract tables and generate global schema representations.
    Phase 2: Retrieve table schemas based on user query intent.
    Phase 3: Generate Pandas code to answer table queries.
    Phase 4: Execute code in sandbox and return results.
    Phase 5: Generate natural language response.
    """

    def __init__(self, llm=None):
        self.llm = llm or LLMRouter.get_llm()
        self.table_schemas: Dict[str, TableSchema] = {}

    def extract_tables_from_markdown(self, markdown_text: str, doc_id: str = "",
                                      section: str = "", page: int = 0) -> List[TableSchema]:
        """
        Phase 1: Extract tables from markdown text and generate schema representations.
        Parses markdown tables and creates TableSchema objects.
        """
        schemas = []
        table_pattern = re.compile(
            r'\[Table_(\d+)\]\n(.*?)\n\[/Table\]',
            re.DOTALL
        )

        for match in table_pattern.finditer(markdown_text):
            table_idx = match.group(1)
            table_md = match.group(2)

            try:
                df = self._parse_markdown_table(table_md)
                if df is not None and not df.empty:
                    table_id = f"{doc_id}_table_{table_idx}"
                    table_name = f"Table {table_idx}"
                    schema = TableSchema(
                        table_id=table_id,
                        name=table_name,
                        df=df,
                        source_doc_id=doc_id,
                        section=section,
                        page=page
                    )
                    schemas.append(schema)
                    self.table_schemas[table_id] = schema
                    logger.info(f"Extracted table schema: {table_id} ({df.shape[0]}x{df.shape[1]})")
            except Exception as e:
                logger.warning(f"Failed to parse table {table_idx}: {e}")

        return schemas

    def _parse_markdown_table(self, md: str) -> Optional[pd.DataFrame]:
        """Parse a markdown table string into a pandas DataFrame."""
        lines = [l.strip() for l in md.strip().split('\n') if l.strip()]
        if len(lines) < 2:
            return None

        header_line = lines[0]
        sep_line = lines[1]

        if not re.match(r'^[\s\|\-\:]+$', sep_line):
            return None

        headers = [h.strip() for h in header_line.split('|') if h.strip()]
        if not headers:
            return None

        rows = []
        for line in lines[2:]:
            cells = [c.strip() for c in line.split('|') if c.strip()]
            if len(cells) == len(headers):
                rows.append(cells)

        if not rows:
            return None

        df = pd.DataFrame(rows, columns=headers)

        for col in df.columns:
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            if numeric_col.notna().sum() > len(df) * 0.5:
                df[col] = numeric_col

        return df

    def generate_schema_docs(self) -> List[Document]:
        """Generate Document objects for all extracted table schemas."""
        docs = []
        for schema in self.table_schemas.values():
            docs.append(schema.to_vector_doc())
        return docs

    def retrieve_table_schema(self, query: str, top_k: int = 3) -> List[TableSchema]:
        """
        Phase 2: Retrieve relevant table schemas based on query.
        This is called by ContextManager during retrieval.
        """
        relevant_schemas = []

        for schema in self.table_schemas.values():
            score = self._match_query_to_table(query, schema)
            if score > 0.3:
                relevant_schemas.append((score, schema))

        relevant_schemas.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in relevant_schemas[:top_k]]

    def _match_query_to_table(self, query: str, schema: TableSchema) -> float:
        """Score how well a query matches a table schema."""
        query_lower = query.lower()
        desc_lower = schema.to_description().lower()

        score = 0.0

        table_keywords = ['table', 'compare', 'accuracy', 'precision', 'recall',
                          'f1', 'score', 'metric', 'result', 'performance',
                          'dataset', 'method', 'model', 'baseline', 'ablation',
                          'parameter', 'hyperparameter', 'configuration']

        for kw in table_keywords:
            if kw in query_lower:
                score += 0.1

        col_keywords = [col.lower() for col in schema.df.columns]
        for kw in col_keywords:
            if kw in query_lower:
                score += 0.2

        if schema.name.lower() in query_lower:
            score += 0.3

        common_words = set(query_lower.split()) & set(desc_lower.split())
        score += len(common_words) * 0.05

        return min(score, 1.0)

    def is_table_query(self, query: str) -> bool:
        """Determine if a query is likely a table-related query."""
        table_indicators = [
            'table', 'compare', 'accuracy', 'precision', 'recall', 'f1',
            'score', 'metric', 'result', 'performance', 'dataset',
            'method', 'model', 'baseline', 'ablation', 'parameter',
            'hyperparameter', 'configuration', 'vs', 'versus',
            'which method', 'which model', 'best', 'worst',
            'highest', 'lowest', 'average', 'mean', 'sum', 'count',
            'rank', 'top', 'bottom', 'difference', 'gap'
        ]

        query_lower = query.lower()
        indicator_count = sum(1 for ind in table_indicators if ind in query_lower)
        return indicator_count >= 2

    def generate_pandas_code(self, query: str, schema: TableSchema) -> str:
        """
        Phase 3: Generate Pandas code to answer the query.
        Uses LLM to generate executable Python code.
        """
        table_desc = schema.to_description()

        prompt = f"""You are a data analyst expert. Given a table schema and a user query, generate Python Pandas code to answer the query.

Table Schema:
{table_desc}

User Query: {query}

Requirements:
1. The DataFrame is available as variable `df`
2. Generate ONLY executable Python code, no explanations
3. Store the final result in variable `final_result`
4. `final_result` should be a simple value (number, string) or a small summary DataFrame
5. Handle potential missing values gracefully
6. Use appropriate aggregation (sum, mean, max, min, count, etc.)
7. If comparing values, compute the difference or ratio

Example output format:
```python
result = df.groupby('method')['accuracy'].mean()
final_result = result.to_string()
```

Generate the code now:
"""

        try:
            response = self.llm.invoke(prompt)
            code = response.content if hasattr(response, 'content') else str(response)

            code_block = re.search(r'```(?:python)?\s*\n(.*?)\n```', code, re.DOTALL)
            if code_block:
                code = code_block.group(1).strip()
            else:
                code = code.strip()

            logger.info(f"Generated code for table query: {code[:100]}...")
            return code
        except Exception as e:
            logger.error(f"Failed to generate pandas code: {e}")
            return ""

    def execute_code(self, code: str, schema: TableSchema, max_retries: int = 2) -> Any:
        """
        Phase 4: Execute the generated code in a sandbox environment.
        Includes self-correction on errors.
        """
        local_vars = {'df': schema.df.copy(), 'final_result': None}
        error_msg = ""

        for attempt in range(max_retries + 1):
            try:
                exec(code, {}, local_vars)
                result = local_vars.get('final_result')
                logger.info(f"Code executed successfully, result type: {type(result)}")
                return result
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Code execution error (attempt {attempt + 1}): {error_msg}")

                if attempt < max_retries:
                    code = self._self_correct_code(code, error_msg, schema)
                    if not code:
                        break

        return None

    def _self_correct_code(self, original_code: str, error_msg: str, schema: TableSchema) -> str:
        """Ask LLM to self-correct the code based on error message."""
        table_desc = schema.to_description()

        prompt = f"""The following Pandas code failed to execute. Please fix it.

Table Schema:
{table_desc}

Original Code:
{original_code}

Error Message:
{error_msg}

Requirements:
1. Fix the error and generate corrected code
2. Generate ONLY executable Python code, no explanations
3. Store the final result in variable `final_result`
4. The DataFrame is available as variable `df`

Corrected code:
"""

        try:
            response = self.llm.invoke(prompt)
            code = response.content if hasattr(response, 'content') else str(response)

            code_block = re.search(r'```(?:python)?\s*\n(.*?)\n```', code, re.DOTALL)
            if code_block:
                code = code_block.group(1).strip()
            else:
                code = code.strip()

            return code
        except Exception as e:
            logger.error(f"Self-correction failed: {e}")
            return ""

    def generate_response(self, query: str, result: Any, schema: TableSchema) -> str:
        """
        Phase 5: Generate natural language response from execution result.
        """
        result_str = str(result) if result is not None else "No result could be computed."

        prompt = f"""You are a research assistant. Answer the user's question based on the table data analysis result.

User Query: {query}
Table: {schema.name}
Analysis Result: {result_str}

Provide a clear, concise answer that directly addresses the user's question. 
Include specific numbers and comparisons where relevant.
If the result is a table, summarize the key findings.

Answer:
"""

        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return f"Analysis result: {result_str}"

    def process_table_query(self, query: str) -> Optional[str]:
        """
        Main entry point: Process a table-related query end-to-end.
        Returns the natural language answer or None if no relevant table found.
        """
        if not self.table_schemas:
            return None

        relevant_tables = self.retrieve_table_schema(query)
        if not relevant_tables:
            return None

        best_schema = relevant_tables[0]

        code = self.generate_pandas_code(query, best_schema)
        if not code:
            return None

        result = self.execute_code(code, best_schema)
        if result is None:
            return None

        response = self.generate_response(query, result, best_schema)
        return response

    def get_table_info(self, table_id: str) -> Optional[Dict]:
        """Get information about a specific table."""
        schema = self.table_schemas.get(table_id)
        if not schema:
            return None
        return {
            "table_id": schema.table_id,
            "name": schema.name,
            "shape": schema.df.shape,
            "columns": list(schema.df.columns),
            "source_doc_id": schema.source_doc_id,
            "section": schema.section,
        }

    def list_tables(self) -> List[Dict]:
        """List all available tables."""
        return [self.get_table_info(tid) for tid in self.table_schemas]
