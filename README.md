# Complaint Analysis for Financial Services

This project is a data analysis pipeline designed to process, clean, and visualize consumer complaints regarding financial services. It focuses on analyzing narrative data to understand customer issues across different product categories.

## Features

*   **Efficient Data Loading**: Loads large datasets in chunks to manage memory usage effectively, filtering for rows with narratives immediately.
*   **Data Preprocessing**:
    *   Filters out missing narratives.
    *   Maps raw product names to simplified categories: 'Credit Cards', 'Savings Accounts', 'Personal Loans', and 'Money Transfers'.
    *   Cleans narrative text by removing redacted placeholders (e.g., 'XXXX'), special characters, and extra whitespace.
*   **Data Sampling**:
    *   Creates stratified samples to preserve the percentage distribution of products, ensuring representative subsets for computationally intensive tasks.
*   **Vector Embedding Pipeline**:
    *   Converts complaint narratives into vector embeddings using HuggingFace models (`all-MiniLM-L6-v2`).
    *   Stores embeddings in a local ChromaDB vector store for semantic search and RAG (Retrieval-Augmented Generation) applications.
*   **RAG Evaluation Loop (Task 3)**: Scores the RAG pipeline qualitatively against a fixed question set and exports results to `data/task3_evaluation.csv`.
*   **Interactive RAG Chat (Task 4)**: Streamlit front end powered by Gemini and the Chroma vector store for guided complaint analysis.
*   **Exploratory Data Analysis (EDA)**:
    *   Visualizes the distribution of complaints by product.
    *   Analyzes and plots the distribution of word counts in complaint narratives.

## Project Structure

```
├── data/
│   ├── raw/
│   │   ├── complaints.csv
│   │   └── complaint_embeddings.parquet
│   ├── processed/
│   │   └── filtered_complaints.csv
│   └── task3_evaluation.csv
├── notebooks/
│   ├── task1_eda.ipynb           # EDA
│   ├── task2_embedding.ipynb     # Chunking + embedding + vector store build
│   └── task3_evaluation.ipynb    # RAG evaluation loop
├── src/
│   ├── loader.py                 # Chunked CSV loader
│   ├── preprocessor.py           # Cleaning + product mapping
│   ├── rag_engine.py             # RAGPipeline used by evaluation and app
│   ├── sampler.py                # Stratified sampling utility
│   ├── vectorizer.py             # Embedding + vector store creation
│   └── visualizer.py             # Plot helpers
├── chroma_db_full/               # Persisted Chroma store used by RAGPipeline
├── vector_store/                 # Vector store built in Task 2 notebook
├── requirements.txt              # Python dependencies
|-- app.py                    # Streamlit chat app (Task 4)
└── README.md                     # Project documentation
```

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd Complaint-Analysis-For-Financial-Services
    ```

2.  Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Prepare Data**: Ensure your raw data file `complaints.csv` is placed in the `data/raw/` directory.

2.  **Run the notebooks** (Tasks 1–3):

    *Task 1 — EDA*: Open `notebooks/task1_eda.ipynb` to explore complaint distributions and narrative lengths using `ComplaintLoader`, `ComplaintPreprocessor`, and `ComplaintVisualizer`.

    *Task 2 — Embeddings*: Open `notebooks/task2_embedding.ipynb` to stratify the cleaned data (`data/processed/filtered_complaints.csv`), chunk narratives, and build a Chroma store at `vector_store/` via `VectorPipeline`. The notebook also runs a quick retrieval smoke test.

    *Task 3 — RAG Evaluation*: Open `notebooks/task3_evaluation.ipynb` to load the prebuilt embeddings (`data/raw/complaint_embeddings.parquet`) and vector store (`chroma_db_full/`), run a fixed set of qualitative RAG queries through `RAGPipeline`, and export results to `data/task3_evaluation.csv`.

3.  **Run the Streamlit chat (Task 4)**:

    The chat app in `./app.py` wraps `RAGPipeline` for interactive Q&A with Gemini.

    ```bash
    export GOOGLE_API_KEY=<your_gemini_key>
    streamlit run ./app.py
    ```

    The app will reuse `data/raw/complaint_embeddings.parquet` and the `chroma_db_full/` store. Use the sidebar to enter a Google API key if it is not already exported.

## Dependencies

*   pandas
*   numpy
*   matplotlib
*   seaborn
*   langchain
*   langchain-community
*   langchain-huggingface
*   langchain-chroma
*   langchain-google-genai
*   sentence-transformers
*   chromadb
*   pyarrow
*   streamlit

Gemini-powered steps (Tasks 3 and 4) require `GOOGLE_API_KEY` to be set in your environment or supplied via the Streamlit sidebar.

## Modules

### `src.loader`
Contains the `ComplaintLoader` class, which handles reading the CSV file in chunks to avoid memory issues.

### `src.preprocessor`
Contains the `ComplaintPreprocessor` class, responsible for:
*   Filtering missing values.
*   Mapping specific financial products to broader categories.
*   Text cleaning (lowercasing, removing special characters and redactions).

### `src.sampler`
Contains the `ComplaintSampler` class, which creates stratified samples of the dataset to ensure product distribution is preserved in smaller subsets.

### `src.vectorizer`
Contains the `VectorPipeline` class, which handles:
*   Splitting text into chunks.
*   Generating embeddings using HuggingFace models.
*   Creating and persisting a ChromaDB vector store.

### `src.visualizer`
Contains the `ComplaintVisualizer` class for generating charts such as product distribution bar charts and word count histograms.

### `src.rag_engine`
Contains the `RAGPipeline` class that wires the Chroma retriever, Gemini LLM, and prompt into a runnable chain. Used by the evaluation notebook and the Streamlit chat.

### `app`
Streamlit entry point for Task 4. Provides a chat UI, handles API key input, and surfaces retrieved source snippets for each response.