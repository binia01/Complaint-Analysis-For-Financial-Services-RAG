"""Tests for src.rag_engine.RAGPipeline (using mocks for external services)."""

import os
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.rag_engine import RAGPipeline


# ── Helpers ───────────────────────────────────────────────────────────


def _make_document(text: str, date: str = "2024-01-01") -> Document:
    return Document(page_content=text, metadata={"date": date})


FAKE_DOCS: List[Document] = [
    _make_document("Customer was charged twice."),
    _make_document("Savings account had fraudulent activity."),
]


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure GOOGLE_API_KEY is always set during tests."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-real")


@pytest.fixture()
def mock_rag() -> RAGPipeline:
    """Return a RAGPipeline with all heavy dependencies mocked out."""
    with (
        patch.object(
            RAGPipeline, "__init__", lambda self, *a, **kw: None
        ),
    ):
        pipeline = RAGPipeline.__new__(RAGPipeline)

    # Manually wire lightweight mocks
    pipeline.retriever = MagicMock()
    pipeline.retriever.invoke.return_value = FAKE_DOCS

    pipeline.chain = MagicMock()
    pipeline.chain.invoke.return_value = "Mocked LLM answer"

    pipeline.llm = MagicMock()
    pipeline.prompt = MagicMock()
    pipeline.embedding_fn = MagicMock()
    pipeline.vector_db = MagicMock()
    return pipeline


# ── Tests ─────────────────────────────────────────────────────────────


class TestRAGPipelineInit:
    """Tests around initialisation guards."""

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            # Patch heavy imports but let __init__ run
            with (
                patch(
                    "src.rag_engine.HuggingFaceEmbeddings"
                ),
                patch("src.rag_engine.Chroma"),
                patch("src.rag_engine.ChatGoogleGenerativeAI"),
            ):
                RAGPipeline()


class TestAnswerQuestion:
    """Tests for RAGPipeline.answer_question."""

    def test_returns_expected_keys(self, mock_rag: RAGPipeline) -> None:
        result: Dict[str, Any] = mock_rag.answer_question("test query")
        assert "query" in result
        assert "answer" in result
        assert "source_documents" in result

    def test_query_is_echoed(self, mock_rag: RAGPipeline) -> None:
        result = mock_rag.answer_question("What are top complaints?")
        assert result["query"] == "What are top complaints?"

    def test_answer_comes_from_chain(self, mock_rag: RAGPipeline) -> None:
        result = mock_rag.answer_question("anything")
        assert result["answer"] == "Mocked LLM answer"
        mock_rag.chain.invoke.assert_called_once_with("anything")

    def test_source_documents_from_retriever(
        self, mock_rag: RAGPipeline
    ) -> None:
        result = mock_rag.answer_question("anything")
        assert len(result["source_documents"]) == 2
        mock_rag.retriever.invoke.assert_called_once_with("anything")

    def test_retriever_called_with_user_query(
        self, mock_rag: RAGPipeline
    ) -> None:
        mock_rag.answer_question("credit card fraud")
        mock_rag.retriever.invoke.assert_called_once_with("credit card fraud")


class TestBuildVectorStore:
    """Tests for the static parquet builder (mocked I/O)."""

    def test_file_not_found_raises(self, tmp_path: Any) -> None:
        db = MagicMock()
        with pytest.raises(FileNotFoundError):
            RAGPipeline._build_vector_store_from_parquet(
                db, str(tmp_path / "missing.parquet")
            )
