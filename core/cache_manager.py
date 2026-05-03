import hashlib
import time
import logging
from typing import Optional, List, Dict
from langchain_core.documents import Document
from config.settings import settings

logger = logging.getLogger(__name__)


class CacheManager:
    """
    查询结果+嵌入结果缓存
    对应综述: III.C Query Optimization (减少重复计算)
    """

    def __init__(self, ttl_seconds: int = None):
        self.ttl = ttl_seconds or settings.CACHE_TTL_SECONDS
        self.query_cache: Dict[str, tuple] = {}  # query_hash -> (results, timestamp)
        self.embedding_cache: Dict[str, List[float]] = {}  # text_hash -> embedding

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get_query_results(self, query: str, filters: Dict) -> Optional[List[Document]]:
        """获取缓存的查询结果"""
        if not settings.ENABLE_QUERY_CACHE:
            return None

        cache_key = self._hash(f"{query}|{sorted(filters.items())}")
        if cache_key in self.query_cache:
            results, timestamp = self.query_cache[cache_key]
            if time.time() - timestamp < self.ttl:
                logger.info(f"📦 查询缓存命中: {query[:50]}...")
                return results
            else:
                del self.query_cache[cache_key]  # 过期清理
        return None

    def set_query_results(self, query: str, filters: Dict, results: List[Document]):
        """缓存查询结果"""
        if not settings.ENABLE_QUERY_CACHE:
            return
        cache_key = self._hash(f"{query}|{sorted(filters.items())}")
        self.query_cache[cache_key] = (results, time.time())
        logger.info(f"💾 查询结果已缓存: {len(results)} 文档")

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """获取缓存的嵌入向量"""
        if not settings.ENABLE_EMBEDDING_CACHE:
            return None
        key = self._hash(text)
        return self.embedding_cache.get(key)

    def set_embedding(self, text: str, embedding: List[float]):
        """缓存嵌入向量"""
        if not settings.ENABLE_EMBEDDING_CACHE:
            return
        key = self._hash(text)
        self.embedding_cache[key] = embedding
