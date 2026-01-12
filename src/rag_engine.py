import os
import shutil
import pandas as pd
import pyarrow.parquet as pq
from typing import Dict, Any, List
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

class RAGPipeline:
    def __init__(self, parquet_path: str = "./data/complaint_embeddings.parquet", vector_db_path: str = "./chroma_db_full"):
        """
        Initializes the RAG pipeline.
        """
        # 0. Check Google API Key
        if "GOOGLE_API_KEY" not in os.environ:
            raise ValueError("GOOGLE_API_KEY environment variable is missing!")

        self.vector_db_path = vector_db_path
        self.parquet_path = parquet_path
        
        # 1. Load Embedding Model
        print("[RAG] Loading embedding model (all-MiniLM-L6-v2)...")
        self.embedding_fn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # 2. Check/Initialize Vector DB
        if os.path.exists(vector_db_path) and os.listdir(vector_db_path):
            print(f"[RAG] Loading existing Vector Store from {vector_db_path}...")
            self.vector_db = Chroma(
                persist_directory=vector_db_path,
                embedding_function=self.embedding_fn
            )
        else:
            print(f"[RAG] Vector Store not found at {vector_db_path}.")
            print(f"[RAG] Building Vector Store from {self.parquet_path}...")
            
            # Initialize empty DB first
            self.vector_db = Chroma(
                persist_directory=vector_db_path,
                embedding_function=self.embedding_fn
            )
            # Populate it
            self._build_vector_store_from_parquet(self.parquet_path)

        # 3. Setup Retriever
        self.retriever = self.vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # 4. Initialize Gemini
        print("[RAG] Initializing Google Gemini...")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_output_tokens=500
        )

        # 5. Define Prompt & Chain
        self.prompt = PromptTemplate(
            template="""You are a Senior Financial Analyst for CrediTrust. 
            Answer the user's question based ONLY on the context provided below. 
            If the answer is not in the context, say "I do not have enough information."
            
            CONTEXT:
            {context}

            QUESTION: 
            {question}

            ANSWER:""",
            input_variables=["context", "question"]
        )
        
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()} 
            | self.prompt 
            | self.llm 
            | StrOutputParser()
        )

    def _build_vector_store_from_parquet(self, parquet_path: str):
        """
        Reads Parquet with schema ['id', 'document', 'embedding', 'metadata'].
        Extracts text from 'document' and details from 'metadata'.
        """
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file missing: {parquet_path}")

        print("   Streaming Parquet file...")
        try:
            parquet_file = pq.ParquetFile(parquet_path)
        except Exception as e:
            raise ImportError(f"Could not read parquet. Ensure pyarrow is installed. Error: {e}")

        # Limit total rows for safety
        TOTAL_LIMIT = 20000 
        BATCH_SIZE = 1000
        
        total_docs_added = 0
        
        for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE):
            if total_docs_added >= TOTAL_LIMIT:
                break
                
            df_batch = batch.to_pandas()
            
            docs = []
            for _, row in df_batch.iterrows():
                if total_docs_added >= TOTAL_LIMIT:
                    break

                # 1. Extract Text (Column name is 'document')
                text_content = row.get('document')
                
                if not isinstance(text_content, str) or len(text_content) < 5:
                    continue # Skip empty/short rows

                # 2. Extract Metadata (Column name is 'metadata')
                # In this parquet structure, metadata is likely a dictionary object
                raw_meta = row.get('metadata', {})
                
                # Safety: Ensure it's a dictionary
                if not isinstance(raw_meta, dict):
                    # Fallback if it's None or weird format
                    clean_meta = {
                        "source": "parquet_import", 
                        "original_id": str(row.get('id', 'unknown'))
                    }
                else:
                    # Clean the dictionary (Chroma requires str/int/float, no Nones)
                    clean_meta = {}
                    for k, v in raw_meta.items():
                        if v is None:
                            clean_meta[k] = "Unknown"
                        else:
                            clean_meta[k] = str(v)

                docs.append(Document(page_content=text_content, metadata=clean_meta))
                total_docs_added += 1

            # 3. Add batch to Chroma
            if docs:
                self.vector_db.add_documents(docs)
                print(f"   Indexed {len(docs)} docs (Total: {total_docs_added})...")

        if total_docs_added == 0:
            raise ValueError("No valid documents found! The 'document' column might be empty.")
            
        print("   Indexing Complete!")

    def answer_question(self, query: str) -> Dict[str, Any]:
        print(f"\n[Query] {query}")
        docs = self.retriever.invoke(query)
        response = self.chain.invoke(query)
        return {"query": query, "answer": response, "source_documents": docs}