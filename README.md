# CrediTrust – Complaint Analysis for Financial Services

![Tests](https://github.com/<owner>/Complaint-Analysis-For-Financial-Services/actions/workflows/test.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-chat%20UI-red)

An end-to-end data-analysis and **Retrieval-Augmented Generation (RAG)** pipeline that processes consumer complaints about financial services, embeds them in a vector store, and exposes an interactive Streamlit chat powered by Google Gemini.

---

## Features

| Area | Details |
|------|---------|
| **Data Loading** | Chunked CSV reader that filters missing narratives on the fly to keep memory usage low. |
| **Preprocessing** | Product taxonomy mapping, text cleaning (redaction markers, special characters, whitespace). |
| **Stratified Sampling** | Preserves product-category distribution in smaller subsets for faster experimentation. |
| **Vector Embeddings** | HuggingFace `all-MiniLM-L6-v2` → ChromaDB for semantic similarity search. |
| **RAG Evaluation** | Qualitative scoring loop against a fixed question set (→ `data/task3_evaluation.csv`). |
| **Interactive Chat** | Streamlit front-end with Gemini-backed Q&A, source-evidence display, and sidebar API-key input. |
| **Testing** | pytest suite (21 tests) covering loader, preprocessor, sampler, and RAG engine (mocked). |
| **CI/CD** | GitHub Actions workflow: lint → test → Docker build on every push. |

## Project Structure

```
├── .github/workflows/test.yml    # CI pipeline
├── data/
│   ├── raw/                      # complaints.csv, complaint_embeddings.parquet
│   ├── processed/                # filtered_complaints.csv
│   └── task3_evaluation.csv
├── notebooks/
│   ├── task1_eda.ipynb
│   ├── task2_embedding.ipynb
│   └── task3_evaluation.ipynb
├── src/
│   ├── loader.py                 # ComplaintLoader
│   ├── preprocessor.py           # ComplaintPreprocessor
│   ├── sampler.py                # ComplaintSampler
│   ├── vectorizer.py             # VectorPipeline
│   ├── visualizer.py             # ComplaintVisualizer
│   └── rag_engine.py             # RAGPipeline
├── test/
│   ├── conftest.py               # Shared fixtures
│   ├── test_loader.py
│   ├── test_preprocessor.py
│   ├── test_sampler.py
│   └── test_rag_engine.py
├── chroma_db_full/               # Persisted Chroma store (runtime)
├── vector_store/                 # Store built in Task 2
├── app.py                        # Streamlit chat (Task 4)
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── pyproject.toml                # Tool configs (pytest, flake8, mypy, isort)
└── README.md
```

---

## Quick Start (Local)

```bash
# 1. Clone & enter
git clone <repository-url>
cd Complaint-Analysis-For-Financial-Services

# 2. Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Gemini API key
export GOOGLE_API_KEY=<your_key>

# 5. Run the Streamlit app
streamlit run app.py
```

> **Tip:** If you skip step 4, the app will prompt you for the key in the sidebar.

---

## How to Run via Docker

### Prerequisites
* [Docker](https://docs.docker.com/get-docker/) installed and running.

### Build the image

```bash
docker build -t creditrust-analyst:latest .
```

### Run the container

```bash
docker run -d \
  --name creditrust \
  -p 8501:8501 \
  -e GOOGLE_API_KEY=<your_key> \
  creditrust-analyst:latest
```

The app is now available at **http://localhost:8501**.

### Stop / remove

```bash
docker stop creditrust && docker rm creditrust
```

---

## Running Tests

```bash
# Inside the virtual environment
pytest test/ -v

# With coverage report
pytest test/ --cov=src --cov-report=term-missing
```

---

## Linting & Type Checking

```bash
flake8 src/ app.py
mypy src/
isort --check-only src/ test/
```

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `task1_eda.ipynb` | Explore complaint distributions and narrative lengths. |
| `task2_embedding.ipynb` | Stratify data, chunk, embed → ChromaDB store. |
| `task3_evaluation.ipynb` | Run fixed RAG queries and export evaluation CSV. |

---

## Modules

| Module | Description |
|--------|-------------|
| `src.loader` | `ComplaintLoader` – chunked CSV reading with immediate narrative filtering. |
| `src.preprocessor` | `ComplaintPreprocessor` – missing-value filtering, product mapping, text cleaning. |
| `src.sampler` | `ComplaintSampler` – stratified sampling preserving product proportions. |
| `src.vectorizer` | `VectorPipeline` – text splitting, HuggingFace embedding, ChromaDB persistence. |
| `src.visualizer` | `ComplaintVisualizer` – product distribution and word-count charts. |
| `src.rag_engine` | `RAGPipeline` – Chroma retriever + Gemini LLM chain for grounded Q&A. |
| `app` | Streamlit entry-point – chat UI, API-key handling, source evidence display. |

---

## Dependencies

See [requirements.txt](requirements.txt) for the full list. Key packages:

`pandas` · `numpy` · `matplotlib` · `seaborn` · `langchain` · `langchain-chroma` · `langchain-huggingface` · `langchain-google-genai` · `sentence-transformers` · `chromadb` · `pyarrow` · `streamlit`

Gemini-powered steps (Tasks 3–4) require `GOOGLE_API_KEY` in your environment or via the Streamlit sidebar.