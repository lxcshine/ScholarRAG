import os
import re
import logging
from typing import Dict, List, Optional, Tuple
import fitz
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class ConferencePDFExtractor:
    NOISE_PATTERNS = [
        r'^\s*\d+\s*$', r'^.*(?:Proceedings|Conference|Workshop).*\d{4}.*$',
        r'^©\s*\d{4}.*', r'^[A-Z]{2,}\s+\d{4}\s*$', r'^Page\s+\d+\s*of\s+\d+'
    ]
    SECTION_PATTERNS = {
        "abstract": re.compile(r'^\s*abstract\s*$', re.I),
        "introduction": re.compile(r'^\s*(1\s+)?introduction\s*$', re.I),
        "related_work": re.compile(r'^\s*(2\s+)?related\s*(work|literature|studies)\s*$', re.I),
        "method": re.compile(r'^\s*(3\s+)?(method|approach|framework|model|proposed)\s*$', re.I),
        "experiments": re.compile(r'^\s*(4\s+)?(experiments|evaluation|results|implementation)\s*$', re.I),
        "conclusion": re.compile(r'^\s*(5\s+)?(conclusion|discussion|future\s*work)\s*$', re.I),
        "references": re.compile(r'^\s*(references|bibliography)\s*$', re.I),
    }

    def __init__(self, extract_tables: bool = True, preserve_formulas: bool = True, max_pages: int = 0):
        self.extract_tables = extract_tables
        self.preserve_formulas = preserve_formulas
        self.max_pages = max_pages

    def _is_noise_line(self, line: str, page_num: int, total_pages: int) -> bool:
        clean = line.strip()
        if not clean or len(clean) < 3: return True
        for pattern in self.NOISE_PATTERNS:
            if re.match(pattern, clean, re.I): return True
        return False

    def _detect_columns(self, page: fitz.Page) -> Tuple[int, List[fitz.Rect]]:
        blocks = page.get_text("dict")["blocks"]
        text_blocks = [b for b in blocks if b.get("type") == 0]
        if len(text_blocks) < 5: return 1, []
        centers = [(b["bbox"][0] + b["bbox"][2]) / 2 for b in text_blocks]
        median = sorted(centers)[len(centers) // 2]
        left = [b for b in text_blocks if (b["bbox"][0] + b["bbox"][2]) / 2 < median - 20]
        right = [b for b in text_blocks if (b["bbox"][0] + b["bbox"][2]) / 2 > median + 20]
        if len(left) > 10 and len(right) > 10:
            return 2, [fitz.Rect(min(b["bbox"][0] for b in left), min(b["bbox"][1] for b in left),
                                 max(b["bbox"][2] for b in left), max(b["bbox"][3] for b in left)),
                       fitz.Rect(min(b["bbox"][0] for b in right), min(b["bbox"][1] for b in right),
                                 max(b["bbox"][2] for b in right), max(b["bbox"][3] for b in right))]
        return 1, []

    def _extract_text_with_layout(self, page: fitz.Page, col_count: int, col_rects: List[fitz.Rect]) -> str:
        if col_count == 1: return page.get_text("text", sort=True)
        page_text = page.get_text("dict")
        left_lines, right_lines = [], []
        for b in page_text["blocks"]:
            if b.get("type") != 0: continue
            bbox = fitz.Rect(b["bbox"])
            if col_rects[0].intersects(bbox):
                for line in b.get("lines", []):
                    txt = "".join(s["text"] for s in line.get("spans", []))
                    if txt.strip(): left_lines.append((txt, line.get("bbox", [0, 0, 0, 0])[1]))
            elif col_rects[1].intersects(bbox):
                for line in b.get("lines", []):
                    txt = "".join(s["text"] for s in line.get("spans", []))
                    if txt.strip(): right_lines.append((txt, line.get("bbox", [0, 0, 0, 0])[1]))
        left_lines.sort(key=lambda x: x[1])
        right_lines.sort(key=lambda x: x[1])
        merged, i, j = [], 0, 0
        while i < len(left_lines) or j < len(right_lines):
            if i < len(left_lines) and (j >= len(right_lines) or left_lines[i][1] <= right_lines[j][1] + 5):
                merged.append(left_lines[i][0]);
                i += 1
            if j < len(right_lines) and (i >= len(left_lines) or right_lines[j][1] <= left_lines[i][1] + 5):
                merged.append(right_lines[j][0]);
                j += 1
        return "\n".join(merged)

    def _extract_tables_as_markdown(self, page: fitz.Page) -> str:
        try:
            tabs = page.find_tables()
        except AttributeError:
            return ""
        if not tabs.tables: return ""
        md_parts = []
        for i, tab in enumerate(tabs.tables):
            try:
                data = tab.extract()
                if not data or len(data) < 2: continue
                is_threeline = len(data) >= 2 and any("---" in str(c) for c in data[1])
                header = "| " + " | ".join(str(c or "").replace("\n", " ").strip() for c in data[0]) + " |\n"
                sep = "| " + " | ".join(["---"] * len(data[0])) + " |\n"
                rows_data = data[2:] if is_threeline else data[1:]
                rows = ["| " + " | ".join(str(c or "").replace("\n", " ").replace("|", "\\|").strip() for c in r) + " |"
                        for r in rows_data]
                md_parts.append(f"\n[Table_{i + 1}]\n{header}{sep}" + "\n".join(rows) + "\n[/Table]\n")
            except Exception as e:
                logger.warning(f"表格转换失败 (Page {page.number}): {e}")
        return "".join(md_parts)

    def _protect_formulas(self, text: str) -> str:
        if not self.preserve_formulas: return text
        text = re.sub(r'\$\$([^$]+)\$\$', r'[Formula:block]\1[/Formula]', text)
        text = re.sub(r'(?<!\$)\$([^\$]+)\$(?!\$)', r'[Formula:inline]\1[/Formula]', text)
        return text

    def _parse_metadata(self, text: str, fitz_meta: Dict, first_page: str) -> Dict:
        title = fitz_meta.get("title", "").strip()
        authors = fitz_meta.get("author", "Unknown")
        year = "Unknown"
        venue = "Unknown"
        if not title:
            for line in first_page.split("\n")[:15]:
                l = line.strip()
                if 20 < len(l) < 120 and not l.lower().startswith(("abstract", "introduction", "copyright")):
                    title = l;
                    break
        year_match = re.search(r'(20\d{2})', text[:2000])
        if year_match: year = year_match.group(1)
        venue_match = re.search(r'(CVPR|ICCV|ECCV|NeurIPS|ICML|ICLR|IEEE|ACL|AAAI)', text[:1000], re.I)
        if venue_match: venue = venue_match.group(1).upper()
        return {"title": title, "authors": authors, "year": year, "venue": venue}

    def extract(self, pdf_path: str) -> Document:
        if not os.path.exists(pdf_path): raise FileNotFoundError(pdf_path)
        doc_id = f"doc_{os.path.basename(pdf_path).replace('.pdf', '')}"
        doc = fitz.open(pdf_path)
        fitz_meta = doc.metadata or {}
        first_page_text = doc[0].get_text("text", sort=True) if len(doc) > 0 else ""
        text_parts, struct_markers = [], []
        col_count = 1  # 默认值，防止未定义

        for idx in range(len(doc)):
            if self.max_pages and idx >= self.max_pages: break
            page = doc[idx]
            col_count, col_rects = self._detect_columns(page)
            page_text = self._extract_text_with_layout(page, col_count, col_rects)
            clean_lines = [l for l in page_text.split("\n") if not self._is_noise_line(l, idx, len(doc))]
            page_text = "\n".join(clean_lines)
            if self.extract_tables:
                page_text += self._extract_tables_as_markdown(page)
            if self.preserve_formulas:
                page_text = self._protect_formulas(page_text)
            for line in page_text.split("\n")[:50]:
                for st, pat in self.SECTION_PATTERNS.items():
                    if pat.match(line.strip().lower()):
                        struct_markers.append(f"[Section:{st}:page={idx}]");
                        break
            text_parts.append(page_text.strip())

        raw_text = "\n\n".join(text_parts)
        clean_text = re.sub(r'\n{3,}', '\n\n', raw_text).strip()
        meta = self._parse_metadata(clean_text, fitz_meta, first_page_text)

        # 修复：构建 metadata 字典，跳过空列表
        final_meta = {
            "source": pdf_path,
            "doc_id": doc_id,
            "page_count": len(doc),
            "is_two_column": (col_count == 2),
            "has_tables": self.extract_tables,
            "extraction_engine": "PyMuPDF+LayoutAware",
            "chunk_ready": True
        }

        # 仅当 struct_markers 非空时才添加
        if struct_markers:
            final_meta["struct_markers"] = struct_markers

        # 合并解析的元数据
        final_meta.update(meta)

        doc.close()
        return Document(page_content=clean_text, metadata=final_meta)


def extract_pdf(pdf_path: str) -> Document:
    return ConferencePDFExtractor().extract(pdf_path)
