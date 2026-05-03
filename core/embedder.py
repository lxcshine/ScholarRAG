import hashlib
import logging
import re
import time
import tiktoken
from typing import List, Optional, Dict
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from config.settings import settings

logger = logging.getLogger(__name__)


class SafeOllamaEmbeddings(Embeddings):
    """
    生产级 Ollama 嵌入包装器（修复 NaN/500 崩溃）
    特性: 严格清洗 + 异常兜底 + 缓存 + 维度对齐
    """

    def __init__(self, model: str = None, base_url: str = None, max_length: int = None):
        self.model = "bge-m3"
        self.base_url = settings.OLLAMA_BASE_URL
        self.max_length = max_length or settings.EMBEDDING_MAX_LENGTH
        self._client = OllamaEmbeddings(model=self.model, base_url=self.base_url)

        try:
            self._tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")

        self._cache: Dict[str, tuple] = {}
        self.cache_ttl = settings.CACHE_TTL_SECONDS
        logger.info(f"✅ 嵌入模型初始化: {self.model} (本地, 维度=1024)")

    def _sanitize(self, text: str) -> str:
        """严格清洗文本，防止触发 Ollama 的 NaN Bug"""
        if not isinstance(text, str):
            text = str(text)
        # 1. 移除控制字符 & 不可见符号
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        # 2. 规范化空白
        text = re.sub(r'\s+', ' ', text).strip()
        # 3. 替换已知触发词
        text = text.replace('NaN', 'NotANumber').replace('Infinity', 'Inf').replace('-Infinity', 'NegInf')
        # 4. 最小长度保护
        if len(text) < 3:
            return "[MINIMAL_VALID_TEXT]"
        return text

    def _count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text)) if text else 0

    def _chunk_text(self, text: str) -> List[str]:
        text = self._sanitize(text)
        if self._count_tokens(text) <= self.max_length:
            return [text]
        sentences = [s.strip() + '.' for s in text.replace('\n', '. ').split('.') if s.strip()]
        chunks, current, current_tokens = [], [], 0
        for sent in sentences:
            sent_tokens = self._count_tokens(sent)
            if current_tokens + sent_tokens > self.max_length and current:
                chunks.append(' '.join(current))
                current, current_tokens = [], 0
            current.append(sent)
            current_tokens += sent_tokens
        if current:
            chunks.append(' '.join(current))
        return chunks if chunks else [text[:1000]]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        results = []
        current_time = time.time()

        for i, raw_text in enumerate(texts):
            clean_text = self._sanitize(raw_text)
            cache_key = hashlib.md5(f"{self.model}:{clean_text[:500]}".encode()).hexdigest()

            # 缓存命中
            if cache_key in self._cache:
                vec, ts = self._cache[cache_key]
                if current_time - ts < self.cache_ttl:
                    results.append(vec)
                    continue

            # 执行嵌入
            chunks = self._chunk_text(clean_text)
            vec = None
            try:
                if len(chunks) == 1:
                    vec = self._client.embed_query(chunks[0])
                else:
                    vectors = [self._client.embed_query(c) for c in chunks]
                    dim = len(vectors[0])
                    vec = [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]

                self._cache[cache_key] = (vec, current_time)
                results.append(vec)

            except Exception as e:
                # 捕获 Ollama 500 / NaN / 网络超时等所有异常
                logger.warning(f"⚠️ 嵌入失败 (chunk {i}): {str(e)[:80]} | 使用零向量兜底")
                # bge-m3 固定输出 1024 维
                vec = [0.0] * 1024
                results.append(vec)

        return results

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def get_embedder() -> SafeOllamaEmbeddings:
    return SafeOllamaEmbeddings()
