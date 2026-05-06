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
    Production-grade Ollama embedding wrapper (fixes NaN/500 crashes)
    Features: Strict sanitization + exception fallback + caching + dimension alignment
    """

    def __init__(self, model: str = None, base_url: str = None, max_length: int = None):
        self.model = "qwen3-embedding:0.6b"
        self.base_url = settings.OLLAMA_BASE_URL
        self.max_length = max_length or settings.EMBEDDING_MAX_LENGTH
        self._client = OllamaEmbeddings(model=self.model, base_url=self.base_url)

        try:
            self._tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")

        self._cache: Dict[str, tuple] = {}
        self.cache_ttl = settings.CACHE_TTL_SECONDS
        logger.info(f"Embedding model initialized: {self.model} (local, dim=1024)")

    def _sanitize(self, text: str) -> str:
        """Strictly sanitize text to prevent Ollama NaN Bug"""
        if not isinstance(text, str):
            text = str(text)
        # 1. Remove control characters & invisible symbols
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        # 2. Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # 3. Replace all NaN variants (case-insensitive, word boundaries)
        text = re.sub(r'(?i)\bnan\b', 'NotANumber', text)
        text = re.sub(r'(?i)\bnull\b', 'NullValue', text)
        text = re.sub(r'(?i)\bnone\b', 'NoneValue', text)
        text = text.replace('Infinity', 'Inf').replace('-Infinity', 'NegInf')
        # 4. Minimum length protection
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

            # Cache hit
            if cache_key in self._cache:
                vec, ts = self._cache[cache_key]
                if current_time - ts < self.cache_ttl:
                    results.append(vec)
                    continue

            # Execute embedding
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
                # Catch all exceptions: Ollama 500 / NaN / network timeout
                logger.warning(f"Embedding failed (chunk {i}): {str(e)[:80]} | Using zero vector fallback")
                # qwen3-embedding fixed output 1024 dimensions
                vec = [0.0] * 1024
                results.append(vec)

        return results

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def get_embedder() -> SafeOllamaEmbeddings:
    return SafeOllamaEmbeddings()
