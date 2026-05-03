import os
import logging
from pydantic_settings import BaseSettings
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


class Settings(BaseSettings):
    # ========== 模型路由配置 ==========
    LLM_MODE: str = os.getenv("LLM_MODE", "online")

    # 本地模型
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    LOCAL_LLM_MODEL: str = os.getenv("LOCAL_LLM_MODEL", "gemma4:31b")

    # 在线模型（智谱）
    GLM_API_KEY: str = os.getenv("GLM_API_KEY", "")
    GLM_BASE_URL: str = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
    ONLINE_LLM_MODEL: str = os.getenv("GLM_MODEL", "glm-4-flash")  # ✅ 读取 GLM_MODEL

    # ========== 嵌入模型（固定本地） ==========
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "bge-m3")
    EMBEDDING_MAX_LENGTH: int = int(os.getenv("EMBEDDING_MAX_LENGTH", "2048"))
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

    # ========== 重排序模型 ==========
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # ========== 检索策略 ==========
    RETRIEVAL_MODE: str = os.getenv("RETRIEVAL_MODE", "fast")
    MAX_ITERATIONS: int = 1
    TOP_K_RETRIEVAL: int = 10
    TOP_N_RERANK: int = 3
    ENABLE_HYDE: bool = False
    ENABLE_QUERY_REWRITE: bool = False
    ENABLE_HYBRID_SEARCH: bool = True
    ENABLE_METADATA_FILTER: bool = True

    # ========== 系统配置 ==========
    CHROMA_PATH: str = str(Path(__file__).resolve().parent.parent / "chroma_db")
    PDF_DIR: str = str(Path(__file__).resolve().parent.parent / "papers")
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_CONTEXT_TOKENS: int = 8000
    CACHE_TTL_SECONDS: int = 3600
    MAX_CONCURRENT_REQUESTS: int = 4

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
