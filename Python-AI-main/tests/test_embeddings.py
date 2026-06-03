"""BGE 嵌入封装测试。"""

from __future__ import annotations

import numpy as np
import pytest

from campus_rag.config import BGE_CACHE_DIR
from campus_rag.embeddings import BGEEmbedder


class DummySentenceTransformer:
    def __init__(self) -> None:
        self.seen_texts: list[str] = []

    def encode(
        self,
        texts: list[str],
        batch_size: int,
        show_progress_bar: bool,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> np.ndarray:
        self.seen_texts = texts
        raw = np.arange(len(texts) * 4, dtype=np.float32).reshape(len(texts), 4) + 1.0
        if normalize_embeddings:
            raw = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        return raw


def _has_cached_model_files() -> bool:
    return any(BGE_CACHE_DIR.rglob("modules.json")) or any(BGE_CACHE_DIR.rglob("config.json"))


def test_embedder_is_lazy_loaded() -> None:
    embedder = BGEEmbedder(model_name="local-test-model")

    assert embedder.model_name == "local-test-model"
    assert embedder._model is None


def test_encode_returns_float32_normalized_vectors() -> None:
    embedder = BGEEmbedder(model_name="local-test-model")
    dummy = DummySentenceTransformer()
    embedder._model = dummy  # 避免单元测试加载真实大模型。
    embedder.dim = 4

    vectors = embedder.encode(["选课时间", "图书馆开放"], show_progress=False)

    assert vectors.shape == (2, 4)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)


def test_empty_encode_uses_configured_dimension() -> None:
    embedder = BGEEmbedder(model_name="local-test-model")
    embedder.dim = 4

    vectors = embedder.encode([])

    assert vectors.shape == (0, 4)
    assert vectors.dtype == np.float32


def test_encode_queries_adds_instruction_prefix() -> None:
    embedder = BGEEmbedder(model_name="local-test-model")
    dummy = DummySentenceTransformer()
    embedder._model = dummy
    embedder.dim = 4

    embedder.encode_queries(["奖学金怎么申请？"])

    assert dummy.seen_texts == ["为这个句子生成表示以用于检索相关文章：奖学金怎么申请？"]


def test_encode_documents_keeps_original_text() -> None:
    embedder = BGEEmbedder(model_name="local-test-model")
    dummy = DummySentenceTransformer()
    embedder._model = dummy
    embedder.dim = 4

    embedder.encode_documents(["宿舍报修流程"])

    assert dummy.seen_texts == ["宿舍报修流程"]


@pytest.mark.model
def test_real_bge_encode_dimension_and_normalization() -> None:
    if not _has_cached_model_files():
        pytest.skip("本地 BGE 模型缓存缺失，跳过真实模型集成测试。")

    embedder = BGEEmbedder()

    vectors = embedder.encode(["同济大学图书馆开放时间"], show_progress=False)

    assert vectors.shape == (1, 1024)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3)
