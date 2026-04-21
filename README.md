# Research Paper Q&A

This project is a paper-grounded question answering system for research PDFs. It loads uploaded papers, builds a local vector index, routes questions through a LangGraph workflow, and returns answers with sources attached.

The codebase includes three ways to use it:

- a Streamlit app for interactive use
- a CLI for quick terminal experiments
- a notebook for module checks and lightweight validation

## What the project does

- Reads 1 to 5 PDF papers.
- Extracts text and chunks it for retrieval.
- Builds a ChromaDB knowledge base.
- Uses LangGraph to route between retrieval, tools, and memory.
- Supports local PDF questions and optional web/ArXiv context in the UI.
- Exposes sources so answers can be checked against the original text.

## Installation

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

If you are on Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

On Windows PowerShell, use:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If script execution is blocked in PowerShell, run this once in an elevated or user shell:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. Install dependencies

Use one of the following:

```bash
pip install -r requirements.txt
```

or, if you prefer `uv`:

```bash
uv sync
```

### 3. Configure environment variables

Create a `.env` file in the project root and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Optional settings:

```env
GROQ_MODEL=llama-3.1-8b-instant
```

If you do not set `GROQ_MODEL`, the project tries a small list of Groq models and picks the first one that works.

## Run the project

### Streamlit app

```bash
streamlit run capstone_streamlit.py
```

In the sidebar, upload 1 to 5 PDFs, optionally label them, and start asking questions.

### CLI version

```bash
python main.py
```

The CLI scans the `papers/` folder, builds the index, and then lets you ask questions from the terminal.

### Notebook

Open `day13_capstone.ipynb` if you want to run the notebook checks and inspect the pipeline step by step.

## Project structure

- `capstone_streamlit.py`: Streamlit UI.
- `main.py`: Command-line entry point.
- `src/extractor.py`: PDF extraction and chunking.
- `src/knowledge_base.py`: ChromaDB collection setup and retrieval gate.
- `src/graph.py`: LangGraph graph construction and question routing.
- `src/nodes.py`: Node logic for retrieval, tool use, answering, and memory.
- `src/evaluator.py`: RAG evaluation helpers.
- `src/tools.py`: Date, web, and ArXiv helper tools.
- `day13_capstone.ipynb`: Notebook used for module and integration checks.

## Requirements and data

- Python 3.12 or newer.
- A working Groq API key.
- PDF papers placed in the `papers/` directory for the CLI and notebook.

The Streamlit app lets you upload PDFs directly, so you do not need to pre-populate `papers/` for that path.

## Notes

- The system is designed to stay grounded in the uploaded papers.
- When web context is enabled in the UI, it can use ArXiv or web snippets for broader questions.
- The notebook is intended to validate the modules quickly, not to benchmark model quality.

## Author

Kanishka
