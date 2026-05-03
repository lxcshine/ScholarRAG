import json
import re
import logging
from langchain_core.prompts import ChatPromptTemplate
from config.prompts import SUFFICIENCY_CHECK_PROMPT, FINAL_GENERATION_PROMPT
from pipeline.context_manager import ContextManager
from pipeline.context_compressor import compress_context
from config.settings import settings
from core.llm_router import LLMRouter

logger = logging.getLogger(__name__)


class IterativeRAGAgent:
    def __init__(self, context_mgr: ContextManager, llm=None):
        self.context_mgr = context_mgr
        self.llm = llm or LLMRouter.get_llm()

    def run(self, query: str) -> str:
        # 【快速路径】简单查询直接检索+生成
        if settings.RETRIEVAL_MODE == "fast" and len(query) < 40:
            docs = self.context_mgr.retrieve(query)
            if not docs:
                return "📚 知识库中未找到相关文献。"
            context = compress_context(docs, query)
            prompt = ChatPromptTemplate.from_template(FINAL_GENERATION_PROMPT)
            return (prompt | self.llm).invoke({"context": context, "query": query}).content

        # 标准流程（最多2轮）
        all_docs = []
        current_query = query

        for i in range(min(settings.MAX_ITERATIONS, 2)):  # 强制最多2轮
            docs = self.context_mgr.retrieve(current_query)
            if not docs:
                break
            all_docs.extend(docs)

            # 去重
            seen = set()
            unique = []
            for d in all_docs:
                cid = d.metadata.get("chunk_id")
                if cid and cid not in seen:
                    seen.add(cid)
                    unique.append(d)
            all_docs = unique

            context = compress_context(all_docs, query)
            if not context.strip():
                return "📚 知识库中未找到相关文献。"

            # 充分性检查（仅balanced/accurate模式启用）
            if settings.RETRIEVAL_MODE != "fast" and i < settings.MAX_ITERATIONS - 1:
                try:
                    preview = context[:600] + "..."
                    check_prompt = ChatPromptTemplate.from_template(SUFFICIENCY_CHECK_PROMPT)
                    check_resp = (check_prompt | self.llm).invoke({
                        "query": query,
                        "context_preview": preview
                    }).content
                    clean = re.sub(r"^```(?:json)?\n?|```$", "", check_resp, flags=re.MULTILINE).strip()
                    check_data = json.loads(clean)
                    if check_data.get("is_sufficient", False):
                        break
                    current_query += f" [补充:{check_data.get('missing_info', '')}]"
                except:
                    pass  # 解析失败则继续

        # 最终生成
        final_prompt = ChatPromptTemplate.from_template(FINAL_GENERATION_PROMPT)
        final_context = compress_context(all_docs, query)
        return (final_prompt | self.llm).invoke({
            "context": final_context,
            "query": query
        }).content
