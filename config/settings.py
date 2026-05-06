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
    ONLINE_LLM_MODEL: str = os.getenv("GLM_MODEL", "glm-4-flash")  # ? 读取 GLM_MODEL

    # ========== 嵌入模型（固定本地） ==========
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
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

    # ========== System Configuration ==========
    CHROMA_PATH: str = str(Path(__file__).resolve().parent.parent / "chroma_db")
    PDF_DIR: str = str(Path(__file__).resolve().parent.parent / "papers")
    HISTORY_DIR: str = str(Path(__file__).resolve().parent.parent / "history")
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_CONTEXT_TOKENS: int = 16000
    CACHE_TTL_SECONDS: int = 3600
    MAX_CONCURRENT_REQUESTS: int = 4

    # ========== RF-Mem Memory Retrieval Configuration ==========
    ENABLE_RF_MEM: bool = True
    RF_MEM_K: int = 10
    RF_MEM_LAMBDA_TEMP: float = 20.0
    RF_MEM_THETA_HIGH: float = 0.6
    RF_MEM_THETA_LOW: float = 0.3
    RF_MEM_TAU: float = 0.2
    RF_MEM_BEAM_WIDTH: int = 3
    RF_MEM_FANOUT: int = 2
    RF_MEM_MAX_ROUNDS: int = 3
    RF_MEM_ALPHA: float = 0.5

    # ========== Conversation Memory Configuration ==========
    ENABLE_CONVERSATION_MEMORY: bool = True
    MAX_CONVERSATION_TURNS: int = 20
    MAX_CONVERSATION_TOKENS: int = 16000
    ENABLE_CONVERSATION_SUMMARY: bool = True
    ENABLE_PERSISTENT_HISTORY: bool = True

    # ========== MySQL History Configuration ==========
    MYSQL_HOST: str = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT: int = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER: str = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "123456")
    MYSQL_DATABASE: str = os.getenv("MYSQL_DATABASE", "research_rag")
    ENABLE_MYSQL_HISTORY: bool = os.getenv("ENABLE_MYSQL_HISTORY", "true").lower() == "true"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
