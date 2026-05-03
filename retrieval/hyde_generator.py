import logging
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from config.prompts import HYDE_PROMPT

logger = logging.getLogger(__name__)


def generate_hyde_doc(query: str, llm: ChatOllama) -> str:
    prompt = ChatPromptTemplate.from_template(HYDE_PROMPT)
    chain = prompt | llm
    try:
        return chain.invoke({"query": query}).content.strip()
    except Exception as e:
        logger.warning(f"HyDE生成失败: {e}")
        return ""
