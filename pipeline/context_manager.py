# -*- coding: utf-8 -*-

import logging
import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from typing import List, Optional
from core.vector_store import Chroma
from core.reranker import Reranker
from core.hybrid_retriever import HybridRetriever
from core.metadata_filter import MetadataFilter
from core.cache_manager import CacheManager
from core.rf_mem_retriever import RFMemRetriever
from retrieval.query_rewriter import expand_query
from retrieval.hyde_generator import generate_hyde_doc
from config.settings import settings

logger = logging.getLogger(__name__)


class ContextManager:
    def __init__(self, vstore: Chroma, reranker: Reranker, llm: ChatOllama):
        self.vstore = vstore
        self.reranker = reranker
        self.llm = llm
        self.hybrid_retriever = HybridRetriever(vstore)
        self.cache = CacheManager()
        self.executor = ThreadPoolExecutor(max_workers=settings.MAX_CONCURRENT_REQUESTS)
        
        self.rf_mem_retriever: Optional[RFMemRetriever] = None
        if settings.ENABLE_RF_MEM:
            self._init_rf_mem()

    def _init_rf_mem(self):
        """Initialize RF-Mem memory retriever"""
        try:
            logger.info("Initializing RF-Mem memory retriever...")
            
            all_docs = self.vstore.get()
            if not all_docs or not all_docs.get("embeddings"):
                logger.warning("Vector store is empty, skipping RF-Mem initialization")
                return
            
            embeddings = np.array(all_docs["embeddings"])
            documents = [
                Document(page_content=content, metadata=all_docs["metadatas"][i])
                for i, content in enumerate(all_docs["documents"])
            ]
            
            self.rf_mem_retriever = RFMemRetriever(
                memory_embeddings=embeddings,
                memory_texts=documents,
                K=settings.RF_MEM_K,
                lambda_temp=settings.RF_MEM_LAMBDA_TEMP,
                theta_high=settings.RF_MEM_THETA_HIGH,
                theta_low=settings.RF_MEM_THETA_LOW,
                tau=settings.RF_MEM_TAU,
                beam_width_B=settings.RF_MEM_BEAM_WIDTH,
                fanout_F=settings.RF_MEM_FANOUT,
                max_rounds_R=settings.RF_MEM_MAX_ROUNDS,
                alpha=settings.RF_MEM_ALPHA
            )
            
            logger.info("RF-Mem memory retriever initialized successfully")
            
        except Exception as e:
            logger.error(f"RF-Mem initialization failed: {e}")
            self.rf_mem_retriever = None

    def retrieve(self, query: str, query_emb: Optional[np.ndarray] = None) -> List[Document]:
        if settings.ENABLE_RF_MEM and self.rf_mem_retriever and query_emb is not None:
            return self._retrieve_with_rf_mem(query, query_emb)
        
        return self._retrieve_standard(query)

    def _retrieve_with_rf_mem(self, query: str, query_emb: np.ndarray) -> List[Document]:
        """Retrieve using RF-Mem intelligent retrieval"""
        logger.info("Using RF-Mem dual-path memory retrieval")
        
        results = self.rf_mem_retriever.retrieve(query_emb)
        
        docs = [doc for doc, score in results]
        
        if not docs:
            logger.warning("RF-Mem returned no results, falling back to standard retrieval")
            return self._retrieve_standard(query)
        
        return docs

    def _retrieve_standard(self, query: str) -> List[Document]:
        """Standard retrieval flow (original logic)"""
        search_queries = [query]
        if settings.ENABLE_QUERY_REWRITE:
            search_queries.extend(expand_query(query, self.llm))
        if settings.ENABLE_HYDE:
            hyde_doc = generate_hyde_doc(query, self.llm)
            if hyde_doc:
                search_queries.append(hyde_doc)

        all_docs = []
        for q in search_queries:
            try:
                all_docs.extend(self.vstore.similarity_search(q, k=settings.TOP_K_RETRIEVAL))
            except Exception as e:
                logger.error(f"Retrieval failed for {q}: {e}")

        seen = set()
        unique_docs = []
        for d in all_docs:
            cid = d.metadata.get("chunk_id")
            if cid and cid not in seen:
                seen.add(cid)
                unique_docs.append(d)

        if not unique_docs:
            logger.warning("No document chunks retrieved")
            return []

        return self.reranker.rerank(query, unique_docs[:20], top_n=settings.TOP_N_RERANK)

    def _match_filters(self, meta: dict, filters: dict) -> bool:
        """Helper: Check if document metadata matches filter conditions"""
        if "venue" in filters and filters["venue"].upper() not in meta.get("venue", "").upper():
            return False
        if "year_min" in filters:
            year = meta.get("year", "")
            if isinstance(year, str) and year.isdigit():
                if int(year) < filters["year_min"]:
                    return False
        return True
