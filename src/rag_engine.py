"""RAG pipeline wiring Chroma retriever, Gemini LLM, and prompt chain."""

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow.parquet as pq
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "gemini-2.5-flash"

_PROMPT_TEMPLATE = """\
You are a Senior Financial Analyst for CrediTrust.
Answer the user's question based ONLY on the context provided below.
If the answer is not in the context, say "I do not have enough information."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline.

    The pipeline reads pre-computed embeddings from a Parquet file (or an
    existing ChromaDB directory), sets up a retriever, and chains it with
    a Google Gemini LLM to answer questions grounded in complaint evidence.
    """

    def __init__(
        self,
        parquet_path: str = "./data/complaint_embeddings.parquet",
        vector_db_path: str = "./chroma_db_full",
        embed_model: str = DEFAULT_EMBED_MODEL,
        llm_model: str = DEFAULT_LLM_MODEL,
        retriever_k: int = 5,
    ) -> None:
        if "GOOGLE_API_KEY" not in os.environ:
            raise ValueError("GOOGLE_API_KEY environment variable is missing!")

        self.vector_db_path = vector_db_path
        self.parquet_path = parquet_path

        # 1. Embedding model
        logger.info("Loading embedding model (%s)...", embed_model)
        self.embedding_fn = HuggingFaceEmbeddings(model_name=embed_model)

        # 2. Vector store
        self.vector_db = self._load_or_build_vector_store(vector_db_path)

        # 3. Retriever
        self.retriever = self.vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": retriever_k},
        )

        # 4. LLM
        logger.info("Initializing Google Gemini (%s)...", llm_model)
        self.llm = ChatGoogleGenerativeAI(
            model=llm_model, temperature=0
        )

        # 5. Prompt & chain
        self.prompt = PromptTemplate(
            template=_PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        )
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    # ------------------------------------------------------------------
    # Vector-store helpers
    # ------------------------------------------------------------------

    def _load_or_build_vector_store(self, path: str) -> Chroma:
        """Load an existing Chroma store or build one from Parquet."""
        if os.path.exists(path) and os.listdir(path):
            logger.info("Loading existing vector store from %s...", path)
            return Chroma(
                persist_directory=path,
                embedding_function=self.embedding_fn,
            )

        logger.info("Vector store not found – building from %s...", self.parquet_path)
        db = Chroma(
            persist_directory=path,
            embedding_function=self.embedding_fn,
        )
        self._build_vector_store_from_parquet(db, self.parquet_path)
        return db

    @staticmethod
    def _build_vector_store_from_parquet(
        db: Chroma,
        parquet_path: str,
        total_limit: int = 20_000,
        batch_size: int = 1_000,
    ) -> None:
        """Stream a Parquet file into the Chroma vector store."""
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file missing: {parquet_path}")

        logger.info("Streaming Parquet file...")
        try:
            parquet_file = pq.ParquetFile(parquet_path)
        except Exception as exc:
            raise ImportError(
                f"Could not read parquet. Ensure pyarrow is installed. Error: {exc}"
            ) from exc

        total_added: int = 0

        for batch in parquet_file.iter_batches(batch_size=batch_size):
            if total_added >= total_limit:
                break

            df_batch: pd.DataFrame = batch.to_pandas()
            docs: List[Document] = []

            for _, row in df_batch.iterrows():
                if total_added >= total_limit:
                    break

                text_content: Optional[str] = row.get("document")
                if not isinstance(text_content, str) or len(text_content) < 5:
                    continue

                raw_meta = row.get("metadata", {})
                if not isinstance(raw_meta, dict):
                    clean_meta = {
                        "source": "parquet_import",
                        "original_id": str(row.get("id", "unknown")),
                    }
                else:
                    clean_meta = {
                        k: str(v) if v is not None else "Unknown"
                        for k, v in raw_meta.items()
                    }

                docs.append(
                    Document(page_content=text_content, metadata=clean_meta)
                )
                total_added += 1

            if docs:
                db.add_documents(docs)
                logger.info(
                    "Indexed %d docs (total: %d)...", len(docs), total_added
                )

        if total_added == 0:
            raise ValueError(
                "No valid documents found! "
                "The 'document' column might be empty."
            )

        logger.info("Indexing complete – %d documents total.", total_added)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer_question(self, query: str) -> Dict[str, Any]:
        """Retrieve evidence and generate an answer for *query*."""
        logger.info("Query: %s", query)
        docs: List[Document] = self.retriever.invoke(query)
        response: str = self.chain.invoke(query)
        return {"query": query, "answer": response, "source_documents": docs}
