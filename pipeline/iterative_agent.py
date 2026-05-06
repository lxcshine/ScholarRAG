# -*- coding: utf-8 -*-
import json
import re
import logging
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from config.prompts import SUFFICIENCY_CHECK_PROMPT, FINAL_GENERATION_PROMPT
from pipeline.context_manager import ContextManager
from pipeline.context_compressor import compress_context
from pipeline.conversation_memory import ConversationMemory
from config.settings import settings
from core.llm_router import LLMRouter
from core.table_processor import TableProcessor

logger = logging.getLogger(__name__)


class IterativeRAGAgent:
    def __init__(self, context_mgr: ContextManager, llm=None, conversation_memory: ConversationMemory = None):
        self.context_mgr = context_mgr
        self.llm = llm or LLMRouter.get_llm()
        self.conversation_memory = conversation_memory
        self.table_processor = TableProcessor(llm=self.llm)

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get query embedding"""
        try:
            from core.embedder import get_embedder
            embedder = get_embedder()
            emb = embedder.embed_query(query)
            return np.array(emb)
        except Exception as e:
            logger.warning(f"Failed to get query embedding: {e}, using standard retrieval")
            return None

    def _build_contextual_query(self, query: str) -> str:
        """Build contextualized query"""
        if self.conversation_memory and not self.conversation_memory.is_empty():
            return self.conversation_memory.get_query_with_context(query)
        return query

    def _build_generation_prompt(self, query: str, context: str) -> str:
        """Build generation prompt with conversation history"""
        if self.conversation_memory and not self.conversation_memory.is_empty():
            conversation_context = self.conversation_memory.get_conversation_context()
            return f"""[Conversation History]
{conversation_context}

[Retrieved References]
{context}

[Current Question]
{query}

Please answer the current question based on the conversation history and references. Maintain coherence and consistency."""
        
        return FINAL_GENERATION_PROMPT.format(context=context, query=query)

    def run(self, query: str) -> str:
        if self.conversation_memory and settings.ENABLE_CONVERSATION_MEMORY:
            self.conversation_memory.add_user_message(query)
        
        contextual_query = self._build_contextual_query(query)
        
        if self.table_processor.table_schemas and self.table_processor.is_table_query(contextual_query):
            logger.info("Detected table query, using table processor")
            table_response = self.table_processor.process_table_query(contextual_query)
            if table_response:
                if self.conversation_memory:
                    self.conversation_memory.add_ai_message(table_response)
                return table_response
            logger.info("Table query returned no result, falling back to standard retrieval")
        
        query_emb = self._get_query_embedding(contextual_query)
        
        if settings.RETRIEVAL_MODE == "fast" and len(query) < 40:
            docs = self.context_mgr.retrieve(contextual_query, query_emb)
            if not docs:
                response = "No relevant documents found in knowledge base."
                if self.conversation_memory:
                    self.conversation_memory.add_ai_message(response)
                return response
            
            context = compress_context(docs, query)
            prompt_text = self._build_generation_prompt(query, context)
            response = self.llm.invoke(prompt_text).content
            
            if self.conversation_memory:
                self.conversation_memory.add_ai_message(response)
                self.conversation_memory.compress_history(self.llm)
            
            return response

        all_docs = []
        current_query = contextual_query
        current_emb = query_emb

        for i in range(min(settings.MAX_ITERATIONS, 2)):
            docs = self.context_mgr.retrieve(current_query, current_emb)
            if not docs:
                break
            all_docs.extend(docs)

            seen = set()
            unique = []
            for d in all_docs:
                cid = d.metadata.get("chunk_id")
                if cid and cid not in seen:
                    seen.add(cid)
                    unique.append(d)
            all_docs = unique

            context = compress_context(all_docs, query)
            if not context.strip():
                response = "No relevant documents found in knowledge base."
                if self.conversation_memory:
                    self.conversation_memory.add_ai_message(response)
                return response

            if settings.RETRIEVAL_MODE != "fast" and i < settings.MAX_ITERATIONS - 1:
                try:
                    preview = context[:600] + "..."
                    check_prompt = ChatPromptTemplate.from_template(SUFFICIENCY_CHECK_PROMPT)
                    check_resp = (check_prompt | self.llm).invoke({
                        "query": query,
                        "context_preview": preview
                    }).content
                    clean = re.sub(r"^```(?:json)?\n?|```$", "", check_resp, flags=re.MULTILINE).strip()
                    check_data = json.loads(clean)
                    if check_data.get("is_sufficient", False):
                        break
                    current_query += f" [Supplement: {check_data.get('missing_info', '')}]"
                    current_emb = self._get_query_embedding(current_query)
                except:
                    pass

        final_context = compress_context(all_docs, query)
        prompt_text = self._build_generation_prompt(query, final_context)
        response = self.llm.invoke(prompt_text).content
        
        if self.conversation_memory:
            self.conversation_memory.add_ai_message(response)
            self.conversation_memory.compress_history(self.llm)
        
        return response
