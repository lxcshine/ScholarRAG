import logging
from typing import Optional
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from config.settings import settings

logger = logging.getLogger(__name__)


class LLMRouter:
    """
    Hybrid model router: local/online generation model one-click switching
    Embedding model fixed to local qwen3-embedding:0.6b, not affected by this router
    """
    _instance = None
    _llm = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_llm(cls, force_mode: str = None) -> any:
        """Get configured LLM instance (singleton + lazy loading)"""
        if cls._llm is not None and force_mode is None:
            return cls._llm

        mode = force_mode or settings.LLM_MODE

        if mode == "local":
            logger.info(f"Using local model: {settings.LOCAL_LLM_MODEL}")
            cls._llm = ChatOllama(
                model=settings.LOCAL_LLM_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0.1,
                num_ctx=16384,
                verbose=False
            )
        else:  # online
            if not settings.GLM_API_KEY:
                logger.warning("GLM_API_KEY not configured, falling back to local model")
                return cls.get_llm("local")
            logger.info(f"Using online model: {settings.ONLINE_LLM_MODEL}")
            cls._llm = ChatOpenAI(
                model=settings.ONLINE_LLM_MODEL,
                openai_api_key=settings.GLM_API_KEY,
                openai_api_base=settings.GLM_BASE_URL,
                temperature=0.1,
                max_tokens=2048,
                verbose=False
            )

        return cls._llm

    @classmethod
    def switch_mode(cls, mode: str) -> bool:
        """Dynamically switch model mode"""
        if mode not in ["local", "online"]:
            return False
        settings.LLM_MODE = mode
        cls._llm = None  # Clear cache, reload next time
        logger.info(f"Model switched to: {mode.upper()}")
        return True
