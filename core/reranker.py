import logging
from sentence_transformers import CrossEncoder
from config.settings import settings
from langchain_core.documents import Document
from typing import List

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self):
        try:
            self.model = CrossEncoder(settings.RERANK_MODEL)
        except Exception as e:
            logger.error(f"加载重排序模型失败: {e}");
            raise

    def rerank(self, query: str, docs: List[Document], top_n: int = None) -> List[Document]:
        if not docs: return []
        top_n = top_n or settings.TOP_N_RERANK
        texts = [d.page_content for d in docs]
        pairs = [[query, t] for t in texts]
        scores = self.model.predict(pairs, show_progress_bar=False)
        scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:top_n]]
