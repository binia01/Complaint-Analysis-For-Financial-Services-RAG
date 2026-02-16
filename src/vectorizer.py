"""Embedding and ChromaDB vector-store creation pipeline."""

import logging
import os
import shutil
from typing import List

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class VectorPipeline:
    """Convert complaint text into vector embeddings stored in ChromaDB."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        logger.info("Loading embedding model (%s)...", model_name)
        self.embedding_fn = HuggingFaceEmbeddings(model_name=model_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_documents(df: pd.DataFrame) -> List[Document]:
        """Convert DataFrame rows into LangChain Document objects."""
        docs: List[Document] = []
        for _, row in df.iterrows():
            doc = Document(
                page_content=row["cleaned_narrative"],
                metadata={
                    "complaint_id": str(row["Complaint ID"]),
                    "product": row["CrediTrust_Product"],
                    "issue": row["Issue"],
                    "date": row["Date received"],
                    "state": str(row["State"]),
                },
            )
            docs.append(doc)
        return docs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_vector_store(
        self,
        df: pd.DataFrame,
        persist_dir: str = "./vector_store",
    ) -> Chroma:
        """Run the full pipeline: Data -> Docs -> Chunks -> VectorDB.

        Args:
            df: Preprocessed complaint DataFrame.
            persist_dir: Path where ChromaDB will be persisted.

        Returns:
            The populated Chroma vector store.
        """
        logger.info("Converting %d rows to Documents...", len(df))
        raw_docs = self._create_documents(df)

        logger.info("Splitting text (chunk_size=%d)...", self.chunk_size)
        chunked_docs: List[Document] = self.splitter.split_documents(raw_docs)
        logger.info(
            "Generated %d chunks from %d complaints.",
            len(chunked_docs),
            len(df),
        )

        if os.path.exists(persist_dir):
            logger.info("Clearing existing vector store at %s...", persist_dir)
            shutil.rmtree(persist_dir)

        logger.info("Embedding chunks and saving to ChromaDB...")
        vector_db: Chroma = Chroma.from_documents(
            documents=chunked_docs,
            embedding=self.embedding_fn,
            persist_directory=persist_dir,
        )
        logger.info("Vector store saved to %s", persist_dir)
        return vector_db
