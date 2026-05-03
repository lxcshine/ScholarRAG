import logging
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from config.settings import settings

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    混合检索器：BM25 + 向量检索融合
    对应综述: III.D.1 Mix/Hybrid Retrieval
    """

    def __init__(self, vector_store, bm25_docs: List[Document] = None):
        self.vector_store = vector_store
        self.bm25_index = None
        self.doc_map = {}  # id -> Document

        if bm25_docs:
            self._build_bm25_index(bm25_docs)

    def _build_bm25_index(self, docs: List[Document]):
        """构建BM25索引（离线预处理）"""
        corpus = []
        for doc in docs:
            doc_id = doc.metadata.get("chunk_id", f"{len(corpus)}")
            # BM25使用纯文本，去除标记
            text = doc.page_content.replace("[Section:", "").replace("[Table", "")
            corpus.append(text.split())
            self.doc_map[doc_id] = doc

        self.bm25_index = BM25Okapi(corpus)
        logger.info(f"✅ BM25索引构建完成: {len(corpus)} 个文档块")

    def _hybrid_score(self, query: str, docs: List[Document],
                      bm25_scores: Dict[str, float],
                      vector_scores: Dict[str, float]) -> List[Tuple[Document, float]]:
        """融合BM25和向量分数"""
        results = []
        for doc in docs:
            doc_id = doc.metadata.get("chunk_id")
            # 归一化分数
            bm25 = bm25_scores.get(doc_id, 0)
            vector = vector_scores.get(doc_id, 0)
            # 加权融合
            score = settings.BM25_WEIGHT * bm25 + (1 - settings.BM25_WEIGHT) * vector
            results.append((doc, score))
        return sorted(results, key=lambda x: x[1], reverse=True)

    def search(self, query: str, top_k: int = None) -> List[Document]:
        top_k = top_k or settings.TOP_K_RETRIEVAL

        # 1. BM25初筛（快速）
        bm25_results = []
        if self.bm25_index:
            query_tokens = query.split()
            bm25_scores = {doc_id: self.bm25_index.get_scores(query_tokens)[i]
                           for i, doc_id in enumerate(self.doc_map.keys())}
            top_bm25 = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)[:settings.TOP_K_HYBRID]
            bm25_results = [self.doc_map[doc_id] for doc_id, _ in top_bm25]

        # 2. 向量检索（精准）
        vector_results = self.vector_store.similarity_search(query, k=top_k)
        vector_scores = {d.metadata.get("chunk_id"): 1.0 - i / top_k for i, d in enumerate(vector_results)}

        # 3. 融合排序
        if settings.ENABLE_HYBRID_SEARCH and bm25_results:
            all_docs = {d.metadata.get("chunk_id"): d for d in vector_results + bm25_results}
            bm25_norm = {k: v / max(bm25_scores.values(), default=1) for k, v in bm25_scores.items()}
            fused = self._hybrid_score(query, list(all_docs.values()), bm25_norm, vector_scores)
            return [doc for doc, _ in fused[:top_k]]

        return vector_results
