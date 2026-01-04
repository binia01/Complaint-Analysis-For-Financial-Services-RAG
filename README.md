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
*   **Exploratory Data Analysis (EDA)**:
    *   Visualizes the distribution of complaints by product.
    *   Analyzes and plots the distribution of word counts in complaint narratives.

## Project Structure

```
├── data/
│   ├── processed/          # Directory for storing processed data
│   └── raw/                # Directory for raw input data (complaints.csv)
├── notebooks/
│   ├── task1_eda.ipynb     # Jupyter notebook for Exploratory Data Analysis
│   └── task2_embedding.ipynb # Jupyter notebook for Vector Embedding generation
├── src/
│   ├── loader.py           # Module for loading data efficiently
│   ├── preprocessor.py     # Module for cleaning and transforming data
│   ├── sampler.py          # Module for creating stratified samples
│   ├── vectorizer.py       # Module for creating vector embeddings and storing them
│   └── visualizer.py       # Module for generating plots and visualizations
├── vector_store/           # Directory for storing ChromaDB vector database
├── requirements.txt        # List of Python dependencies
└── README.md               # Project documentation
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

2.  **Run Analysis**:
    Open the `notebooks/task1_eda.ipynb` notebook to run the analysis pipeline. The notebook demonstrates how to:
    *   Load the data using `ComplaintLoader`.
    *   Preprocess and clean the data using `ComplaintPreprocessor`.
    *   Visualize the results using `ComplaintVisualizer`.

    You can start Jupyter Notebook or Jupyter Lab:
    ```bash
    jupyter notebook
    ```
    Then navigate to `notebooks/task1_eda.ipynb` and run the cells.

3.  **Generate Embeddings**:
    Open the `notebooks/task2_embedding.ipynb` notebook. This notebook demonstrates how to:
    *   Sample the dataset using `ComplaintSampler`.
    *   Create a vector store using `VectorPipeline` (LangChain + ChromaDB).
    *   Perform a semantic search query on the indexed complaints.

## Dependencies

*   pandas
*   numpy
*   matplotlib
*   seaborn
*   langchain
*   langchain-community
*   langchain-huggingface
*   langchain-chroma
*   sentence-transformers
*   chromadb

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