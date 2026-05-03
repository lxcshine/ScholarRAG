import re
import logging
from typing import List, Optional, Dict
from langchain_core.documents import Document
from config.settings import settings

logger = logging.getLogger(__name__)


class MetadataFilter:
    """
    基于元数据的预过滤，快速缩小检索范围
    对应综述: III.B.2 Metadata Attachments
    """

    @staticmethod
    def extract_query_filters(query: str) -> Dict[str, any]:
        """从查询中自动提取过滤条件"""
        filters = {}

        # 提取年份: "2023", "近两年", "recent"
        year_match = re.search(r'(20\d{2})', query)
        if year_match and settings.FILTER_BY_YEAR:
            year = int(year_match.group(1))
            filters["year_min"] = year - 2  # 默认±2年
            filters["year_max"] = year + 1

        # 提取会议: CVPR/ICCV/NeurIPS等
        venues = ["CVPR", "ICCV", "ECCV", "NeurIPS", "ICML", "ICLR", "IEEE", "ACL", "EMNLP"]
        for venue in venues:
            if venue.lower() in query.lower() and settings.FILTER_BY_VENUE:
                filters["venue"] = venue
                break

        # 提取作者/机构（简单匹配）
        # 可扩展：对接学术知识图谱

        return filters

    @staticmethod
    def apply_filters(docs: List[Document], filters: Dict[str, any]) -> List[Document]:
        """应用过滤条件"""
        if not filters:
            return docs

        filtered = []
        for doc in docs:
            meta = doc.metadata
            keep = True

            # 年份过滤
            if "year_min" in filters or "year_max" in filters:
                doc_year = meta.get("year", "")
                if isinstance(doc_year, str) and doc_year.isdigit():
                    doc_year = int(doc_year)
                else:
                    doc_year = 0  # 未知年份保留

                if "year_min" in filters and doc_year < filters["year_min"]:
                    keep = False
                if "year_max" in filters and doc_year > filters["year_max"]:
                    keep = False

            # 会议过滤
            if "venue" in filters:
                doc_venue = meta.get("venue", "").upper()
                if filters["venue"].upper() not in doc_venue:
                    keep = False

            if keep:
                filtered.append(doc)

        logger.info(f"🔍 元数据过滤: {len(docs)} → {len(filtered)} 文档")
        return filtered
