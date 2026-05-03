import json
import re
import logging
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from config.prompts import QUERY_EXPANSION_PROMPT
from config.settings import settings

logger = logging.getLogger(__name__)


def expand_query(query: str, llm: ChatOllama) -> list:
    prompt = ChatPromptTemplate.from_template(QUERY_EXPANSION_PROMPT)
    chain = prompt | llm
    try:
        response = chain.invoke({"query": query}).content.strip()
        response = re.sub(r"^```(?:json)?\n?|```$", "", response, flags=re.MULTILINE).strip()
        return json.loads(response)
    except Exception as e:
        logger.warning(f"查询扩展失败，回退至原始Query: {e}")
        return [query]
