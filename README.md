# FinDoc Agentic AI 🚀

An advanced financial document analysis system powered by **LangGraph** and **FastAPI**. This agentic workflow processes PDF documents (like Annual Reports), extracts key metrics, analyzes financial health, and generates risk assessment reports.

## 🌟 Features

-   **Multi-Agent Architecture**: Uses `Decomposer`, `Retriever`, `Analyst`, `Validator`, and `Summarizer` agents.
-   **RAG (Retrieval Augmented Generation)**: Efficiently searches large PDFs using FAISS and HuggingFace embeddings.
-   **Financial Logic**: Deterministic math for ratios (Debt-to-Equity, Interest Coverage) combined with AI-based text understanding.
-   **Evaluation Pipeline**: Built-in tools to measure latency, accuracy, and confidence.
-   **Deployment Ready**: Configured for Render/Heroku with `Procfile` and `requirements.txt`.

## 📂 Project Structure

-   `app/`: FastAPI application entry point and routes.
-   `agents/`: Logic for individual agents.
-   `graph/`: LangGraph workflow definition.
-   `ingestion/`: PDF loading, chunking, and vector database indexing.
-   `evaluation/`: Scripts to test and benchmark the system.
-   `data/`: Storage for PDFs and Vector Store.

## 🚀 How to Run Locally

### 1. Prerequisites
-   Python 3.10+
-   OpenAI API Key

### 2. Setup
Clone the repo and install dependencies:
```bash
pip install -r requirements.txt
```

Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_key_here
```

### 3. Start the Server
Run the FastAPI server:
```bash
python -m app.main
```
Or use the pre-configured VS Code launch profile.

The API will be available at: `http://localhost:8000/docs`

### 4. Run Evaluation
To test the agents and generate a performance report:
```bash
python -m evaluation.evaluator
```
Results will be saved to `evaluation/results.md`.

## ☁️ Deployment (Render)

1.  Push this repository to GitHub/GitLab.
2.  Create a **Web Service** on [Render](https://render.com/).
3.  Connect your repository.
4.  Render will automatically detect the `Procfile` and `requirements.txt`.
5.  **Important**: Add your `OPENAI_API_KEY` in the Render Environment Variables settings.
6.  Deploy!

## 📊 Evaluation Results
See [evaluation/results.md](evaluation/results.md) for the latest benchmark runs.
