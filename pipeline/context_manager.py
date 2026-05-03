import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from typing import List
from core.vector_store import Chroma
from core.reranker import Reranker
from core.hybrid_retriever import HybridRetriever
from core.metadata_filter import MetadataFilter
from core.cache_manager import CacheManager
from retrieval.query_rewriter import expand_query
from retrieval.hyde_generator import generate_hyde_doc
from config.settings import settings

logger = logging.getLogger(__name__)


class ContextManager:
    def __init__(self, vstore: Chroma, reranker: Reranker, llm: ChatOllama):
        self.vstore = vstore
        self.reranker = reranker
        self.llm = llm
        self.hybrid_retriever = HybridRetriever(vstore)  # 初始化混合检索
        self.cache = CacheManager()
        self.executor = ThreadPoolExecutor(max_workers=settings.MAX_CONCURRENT_REQUESTS)

    def retrieve(self, query: str) -> List[Document]:
        # 1. 构建检索 Query 列表（按需启用 HyDE / 重写）
        search_queries = [query]
        if settings.ENABLE_QUERY_REWRITE:
            search_queries.extend(expand_query(query, self.llm))
        if settings.ENABLE_HYDE:
            hyde_doc = generate_hyde_doc(query, self.llm)
            if hyde_doc:
                search_queries.append(hyde_doc)

        # 2. 执行向量检索
        all_docs = []
        for q in search_queries:
            try:
                all_docs.extend(self.vstore.similarity_search(q, k=settings.TOP_K_RETRIEVAL))
            except Exception as e:
                logger.error(f"检索失败 {q}: {e}")

        # 3. 去重（基于 chunk_id）
        seen = set()
        unique_docs = []
        for d in all_docs:
            cid = d.metadata.get("chunk_id")
            if cid and cid not in seen:
                seen.add(cid)
                unique_docs.append(d)

        if not unique_docs:
            logger.warning("未检索到任何文档块")
            return []

        # 4. 重排序（仅对 Top-20 进行，提速）
        return self.reranker.rerank(query, unique_docs[:20], top_n=settings.TOP_N_RERANK)

    def _match_filters(self, meta: dict, filters: dict) -> bool:
        """辅助：检查文档元数据是否匹配过滤条件"""
        if "venue" in filters and filters["venue"].upper() not in meta.get("venue", "").upper():
            return False
        if "year_min" in filters:
            year = meta.get("year", "")
            if isinstance(year, str) and year.isdigit():
                if int(year) < filters["year_min"]:
                    return False
        return True
