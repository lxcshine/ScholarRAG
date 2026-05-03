import logging
from typing import Optional
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI  # 兼容GLM的OpenAI接口
from config.settings import settings

logger = logging.getLogger(__name__)


class LLMRouter:
    """
    混合模型路由：本地/在线生成模型一键切换
    嵌入模型固定使用本地bge-m3，不受此路由影响
    """
    _instance = None
    _llm = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_llm(cls, force_mode: str = None) -> any:
        """获取配置好的LLM实例（单例+懒加载）"""
        if cls._llm is not None and force_mode is None:
            return cls._llm

        mode = force_mode or settings.LLM_MODE

        if mode == "local":
            logger.info(f"🔌 使用本地模型: {settings.LOCAL_LLM_MODEL}")
            cls._llm = ChatOllama(
                model=settings.LOCAL_LLM_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0.1,
                num_ctx=16384,  # 本地模型可适当调大
                verbose=False
            )
        else:  # online
            if not settings.GLM_API_KEY:
                logger.warning("⚠️ GLM_API_KEY 未配置，回退到本地模型")
                return cls.get_llm("local")
            logger.info(f"🌐 使用在线模型: {settings.ONLINE_LLM_MODEL}")
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
        """动态切换模型模式"""
        if mode not in ["local", "online"]:
            return False
        settings.LLM_MODE = mode
        cls._llm = None  # 清除缓存，下次重新加载
        logger.info(f"🔄 模型已切换至: {mode.upper()}")
        return True
