import re
import logging
from langchain_core.documents import Document
from typing import List

logger = logging.getLogger(__name__)


def validate_citations(response: str, docs: List[Document]) -> str:
    valid_ids = {d.metadata.get("doc_id", "") for d in docs}
    cited_ids = re.findall(r'\[DocID:\s*([^\]]+)\]', response)
    invalid_count = 0
    for cid in cited_ids:
        if cid not in valid_ids:
            response = response.replace(f"[DocID: {cid}]", "[⚠️ 引用源缺失或无法定位]")
            invalid_count += 1
    if invalid_count > 0: logger.warning(f"检测到 {invalid_count} 个无效引用已替换")
    return response
