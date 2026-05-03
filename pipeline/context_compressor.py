import logging
import tiktoken
from langchain_core.documents import Document
from typing import List
from config.settings import settings

logger = logging.getLogger(__name__)

try:
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
except KeyError:
    tokenizer = tiktoken.get_encoding("cl100k_base")


def compress_context(docs: List[Document], query: str = "") -> str:
    """
    【修复版】上下文压缩：优先保留摘要/表格/指标数据
    对应综述: IV.A Context Curation
    """
    if not docs:
        return ""

    # 智能排序：摘要/表格/指标优先
    def priority_score(doc: Document) -> int:
        content = doc.page_content.lower()
        meta = doc.metadata
        score = 0
        if "abstract" in content or meta.get("section", "").lower() == "abstract":
            score += 100
        if "[table" in content or "table_" in content.lower():
            score += 50
        if any(kw in content for kw in ["mAP", "accuracy", "F1", "iou", "dataset"]):
            score += 30
        return score

    sorted_docs = sorted(docs, key=priority_score, reverse=True)

    context_parts = []
    current_tokens = 0

    for doc in sorted_docs:
        meta = doc.metadata
        # 注入结构化头部
        header = f"【{meta.get('title', 'Unknown')}】({meta.get('year', '?')}) [DocID:{meta.get('doc_id', '?')}]\n"

        # 特殊处理：摘要/表格保持完整
        content = doc.page_content
        if "abstract" in content.lower() or "[table" in content.lower():
            # 摘要/表格不截断，直接注入
            full_content = header + content
            content_tokens = len(tokenizer.encode(full_content))
            if current_tokens + content_tokens <= settings.MAX_CONTEXT_TOKENS:
                context_parts.append(full_content)
                current_tokens += content_tokens
        else:
            # 普通文本：安全截断
            content_tokens = len(tokenizer.encode(content))
            if content_tokens > 1000:
                tokens = tokenizer.encode(content)
                content = tokenizer.decode(tokens[:900]) + "..."
                content_tokens = 950

            full_content = header + content
            if current_tokens + content_tokens <= settings.MAX_CONTEXT_TOKENS:
                context_parts.append(full_content)
                current_tokens += content_tokens

        if current_tokens >= settings.MAX_CONTEXT_TOKENS * 0.9:
            logger.info(f"📏 上下文达上限 ({current_tokens}/{settings.MAX_CONTEXT_TOKENS})")
            break

    return "\n\n---\n\n".join(context_parts)
