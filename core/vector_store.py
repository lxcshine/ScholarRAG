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
    
    batch_size = settings.EMBEDDING_BATCH_SIZE
    total = len(chunks)
    logger.info(f"Starting embedding: {total} chunks in batches of {batch_size}")
    
    store = Chroma(
        embedding_function=embedder,
        persist_directory=settings.CHROMA_PATH,
        collection_name="research_papers"
    )
    
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        store.add_documents(batch)
        processed = min(i + batch_size, total)
        progress = processed / total * 100
        logger.info(f"Embedding progress: {processed}/{total} ({progress:.1f}%)")
    
    logger.info(f"Vector index built successfully: {total} chunks")
    return store


def load_vector_store() -> Chroma:
    if not os.path.exists(settings.CHROMA_PATH):
        logger.warning("Vector store not found, please run python main.py --rebuild first")
        return None
    embedder = get_embedder()
    return Chroma(persist_directory=settings.CHROMA_PATH, embedding_function=embedder,
                  collection_name="research_papers")
