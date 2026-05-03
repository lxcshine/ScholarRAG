import os
import logging
from langchain_chroma import Chroma
from config.settings import settings
from core.embedder import get_embedder
from langchain_core.documents import Document
from typing import List

logger = logging.getLogger(__name__)


def init_vector_store(chunks: List[Document]) -> Chroma:
    os.makedirs(settings.CHROMA_PATH, exist_ok=True)
    embedder = get_embedder()
    return Chroma.from_documents(documents=chunks, embedding=embedder, persist_directory=settings.CHROMA_PATH,
                                 collection_name="research_papers")


def load_vector_store() -> Chroma:
    if not os.path.exists(settings.CHROMA_PATH):
        logger.warning("向量库不存在，请先运行 python main.py --rebuild")
        return None
    embedder = get_embedder()
    return Chroma(persist_directory=settings.CHROMA_PATH, embedding_function=embedder,
                  collection_name="research_papers")
