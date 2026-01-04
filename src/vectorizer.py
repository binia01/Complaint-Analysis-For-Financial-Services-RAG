import os
import shutil
from typing import List, Dict
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

class VectorPipeline:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initializes the RAG pipeline components.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize Text Splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize Embedding Model
        # Using CPU-friendly model by default
        print("[VectorPipeline] Loading embedding model (all-MiniLM-L6-v2)...")
        self.embedding_fn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def _create_documents(self, df: pd.DataFrame) -> List[Document]:
        """Converts DataFrame rows into LangChain Document objects."""
        docs = []
        for _, row in df.iterrows():
            # Create a Document for each complaint
            # We store metadata to trace the chunk back to the original issue
            doc = Document(
                page_content=row['cleaned_narrative'],
                metadata={
                    "complaint_id": str(row['Complaint ID']),
                    "product": row['CrediTrust_Product'],
                    "issue": row['Issue'],
                    "date": row['Date received'],
                    "state": str(row['State'])
                }
            )
            docs.append(doc)
        return docs

    def create_vector_store(self, df: pd.DataFrame, persist_dir: str = "./vector_store"):
        """
        Main execution flow: Data -> Docs -> Chunks -> Embeddings -> VectorDB
        """
        # 1. Convert to Documents
        print(f"[VectorPipeline] Converting {len(df)} rows to Documents...")
        raw_docs = self._create_documents(df)

        # 2. Split into Chunks
        print(f"[VectorPipeline] Splitting text (Chunk Size: {self.chunk_size})...")
        chunked_docs = self.splitter.split_documents(raw_docs)
        print(f"[VectorPipeline] Generated {len(chunked_docs)} chunks from {len(df)} original complaints.")

        # 3. Create/Reset Vector DB
        if os.path.exists(persist_dir):
            print(f"[VectorPipeline] Clearing existing vector store at {persist_dir}...")
            shutil.rmtree(persist_dir)

        print("[VectorPipeline] Embedding chunks and saving to ChromaDB (this takes time)...")
        
        # Batch processing happens automatically inside Chroma.from_documents
        vector_db = Chroma.from_documents(
            documents=chunked_docs,
            embedding=self.embedding_fn,
            persist_directory=persist_dir
        )
        print(f"[VectorPipeline] Success! Vector store saved to {persist_dir}")
        return vector_db