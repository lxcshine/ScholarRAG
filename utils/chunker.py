import re
import logging
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class AdvancedPaperChunker:
    """
    面向学术文献的高级分块器 (对应综述 Section III.B Indexing Optimization)
    核心策略:
      1. Small2Big (Parent-Child): 子块检索，父块注入生成上下文
      2. Metadata Attachments: 绑定 section/parent_id/chunk_type 便于过滤与溯源
      3. Table Atomicity: 表格保持完整，超长表按行切分但强制保留表头
      4. Pre-Sanitization: 源头过滤控制符/纯符号/过短块，彻底杜绝 Ollama NaN/500 报错
    """

    def __init__(self, child_chunk_size: int = 800, child_overlap: int = 100, parent_chunk_size: int = 1500):
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )
        self.parent_chunk_size = parent_chunk_size
        # 匹配 CVPR/IEEE/NeurIPS 常见标题层级
        self.header_pattern = re.compile(
            r'^(?:[IVX]+\.|[0-9]+(?:\.[0-9]+)*|Appendix\s*[A-Z]|Abstract|Keywords|'
            r'Introduction|Related\s*Work|Methodology|Experiments|Results|Conclusion|'
            r'References|Acknowledgements|Bibliography)\s*.*$',
            re.IGNORECASE | re.MULTILINE
        )

    def _sanitize_text(self, text: str) -> str:
        """源头清洗：移除控制字符、规范化空白、过滤无效块（防 Ollama 序列化崩溃）"""
        if not isinstance(text, str):
            return ""
        # 1. 移除不可见控制字符 & 换页符
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        # 2. 连续空白压缩为单空格
        text = re.sub(r'\s+', ' ', text).strip()
        # 3. 过滤纯符号/无意义字符/过短文本（防止嵌入模型返回 NaN）
        if len(text) < 10 or re.match(r'^[\W_]+$', text):
            return ""
        return text

    def _parse_blocks(self, text: str) -> List[Dict[str, Any]]:
        """按学术标题层级切分结构块"""
        blocks, current_header, current_block = [], "Main Text", []
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                if current_block:
                    blocks.append({"type": "paragraph", "content": "\n".join(current_block), "header": current_header})
                current_block = []
                continue
            if self.header_pattern.match(line):
                if current_block:
                    blocks.append({"type": "paragraph", "content": "\n".join(current_block), "header": current_header})
                current_block = []
                current_header = line
            else:
                current_block.append(line)
        if current_block:
            blocks.append({"type": "paragraph", "content": "\n".join(current_block), "header": current_header})
        return blocks

    def _detect_tables(self, blocks: List[Dict]) -> List[Dict]:
        """启发式检测表格并标记类型，防止被递归切分器破坏"""
        processed = []
        for block in blocks:
            c = block["content"]
            lines = c.split('\n')
            is_table = len(lines) >= 3 and any(
                '\t' in l or re.match(r'^[\s\-\|]+\S+[\s\-\|]+', l) or re.match(r'^\d+\s+\d+', l) for l in lines[:5]
            )
            if is_table:
                processed.append({"type": "table", "content": c, "header": block["header"]})
            else:
                processed.append(block)
        return processed

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        all_chunks = []
        for doc in docs:
            if not doc.page_content.strip():
                continue

            blocks = self._parse_blocks(doc.page_content)
            blocks = self._detect_tables(blocks)
            parent_id_counter = 0

            for block in blocks:
                # 预清洗：确保送入下游的文本绝对安全
                clean_content = self._sanitize_text(block["content"])
                if not clean_content:
                    continue

                parent_content = f"## {block['header']}\n\n{clean_content}"
                parent_meta = {
                    **doc.metadata,
                    "section": block["header"],
                    "chunk_type": block["type"],
                    "is_parent": True,
                    "parent_id": None
                }

                # 表格处理：超长表格按行拆分，但保留表头与结构标记
                if block["type"] == "table" and len(parent_content) > self.parent_chunk_size:
                    table_lines = parent_content.split('\n')
                    header_lines = [l for l in table_lines[:3] if l.strip()]
                    row_chunks, current_rows = [], []
                    for line in table_lines[len(header_lines):]:
                        if line.strip():
                            current_rows.append(line)
                            if len('\n'.join(current_rows)) > 350:
                                chunk_text = '\n'.join(header_lines + current_rows)
                                if self._sanitize_text(chunk_text):
                                    row_chunks.append(chunk_text)
                                current_rows = []
                    if current_rows:
                        chunk_text = '\n'.join(header_lines + current_rows)
                        if self._sanitize_text(chunk_text):
                            row_chunks.append(chunk_text)

                    pid = f"{doc.metadata.get('doc_id', 'doc')}_{parent_id_counter}"
                    parent_meta["chunk_id"] = pid
                    all_chunks.append(Document(page_content=parent_content, metadata=parent_meta))

                    for idx, ct in enumerate(row_chunks):
                        all_chunks.append(Document(
                            page_content=ct,
                            metadata={**doc.metadata, "section": block["header"], "chunk_type": "table_part",
                                      "is_parent": False, "parent_id": pid, "table_part_index": idx}
                        ))
                    parent_id_counter += 1

                else:
                    # 普通文本：Small2Big 策略（子块检索 + 父级上下文注入）
                    child_texts = self.child_splitter.split_text(clean_content)
                    if not child_texts:
                        continue

                    pid = f"{doc.metadata.get('doc_id', 'doc')}_{parent_id_counter}"
                    parent_meta["chunk_id"] = pid
                    all_chunks.append(Document(page_content=parent_content, metadata=parent_meta))

                    for idx, ct in enumerate(child_texts):
                        enriched = f"[Section: {block['header']}]\n{ct}"
                        # 二次清洗防脏数据
                        if not self._sanitize_text(enriched):
                            continue
                        all_chunks.append(Document(
                            page_content=enriched,
                            metadata={**doc.metadata, "section": block["header"], "chunk_type": "text",
                                      "is_parent": False, "parent_id": pid, "chunk_index": idx}
                        ))
                    parent_id_counter += 1

        logger.info(f"✅ 高级分块完成：共处理 {len(docs)} 篇文献，生成 {len(all_chunks)} 个索引块")
        return all_chunks


# 保持向后兼容的入口函数
def chunk_documents(docs: List[Document]) -> List[Document]:
    return AdvancedPaperChunker().chunk_documents(docs)
