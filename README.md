# ScholarMind: Agentic RAG for Advanced Research Synthesis

ScholarMind is an advanced Agentic RAG (Retrieval-Augmented Generation) system built on the **LangGraph** framework. It is designed to assist researchers in synthesizing information across multiple PDF documents with high technical precision, iterative self-evaluation, and robust citation grounding.

---

## 🚀 Features

- **Agentic Information Routing**: Intelligently decides between local knowledge retrieval, external tool use (ArXiv), or conversation memory.
- **Self-Correcting Evaluation Loop**: Uses a built-in "Eval Node" to audit answer faithfulness and trigger retries if hallucinations are detected.
- **Multi-Document Synthesis**: Compare and contrast findings across up to 5 uploaded research papers simultaneously.
- **LaTeX Support**: Preserves and renders complex mathematical formulas with high fidelity.
- **Persistent Memory**: Uses `MemorySaver` to track conversation context across multiple turns.

---

## 🛠️ Setup Instructions

### Option 1: Standard Pip Setup
It is highly recommended to use a virtual environment.

```bash
# Create the environment
python -m venv .venv

# Activate the environment (Linux/macOS)
source .venv/bin/activate

# Install dependencies from requirements file
pip install -r requirements.txt
```

### Option 2: Using UV (Fastest)
If you have [uv](https://github.com/astral-sh/uv) installed, you can set up the project instantly:

```bash
# Synchronize environment and install dependencies
uv sync
```

### 3. Configure API Keys
ScholarMind uses **Groq** for high-speed inference. You will need a Groq API Key.

1. Create a `.env` file in the root directory.
2. Add your API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

---

## 📦 Core Dependencies

The following packages are mandatory for the system to function. They will be installed automatically via `requirements.txt` or `uv sync`:

- `streamlit`: Web interface framework
- `langgraph`: Agentic workflow orchestration
- `langchain-groq`: Groq LLM integration
- `chromadb`: Vector database for RAG
- `sentence-transformers`: Local embedding generation
- `pymupdf`: PDF text and metadata extraction
- `python-dotenv`: Environment variable management
- `ragas`: RAG evaluation metrics

---

## 💻 Usage

### Launching the Assistant
After activating your environment, start the Streamlit web interface:

```bash
streamlit run capstone_streamlit.py
```

### Interacting with Papers
1. **Upload**: Drag and drop 1 to 5 research PDFs into the sidebar.
2. **Label**: (Optional) Provide custom labels for your documents.
3. **Ask**: Enter your research queries in the chat input (e.g., "Summarize the key findings on neural architecture").
4. **Verify**: Check the **"Sources"** expander to see exactly which chunks of the papers were used.

---

## 📂 Project Structure

- `capstone_streamlit.py`: Main Streamlit UI entry point.
- `main.py`: CLI version of the assistant.
- `src/`: 
    - `graph.py`: LangGraph state machine definition.
    - `nodes.py`: Implementation of the reasoning nodes.
    - `knowledge_base.py`: ChromaDB collection management and retrieval gate.
    - `evaluator.py`: Faithfulness scoring logic.
    - `extractor.py`: PDF text extraction system.

---

## 🎓 Author
**Kanishka**  
Roll Number: 2306118  
Bachelor of Technology, KIIT University  
*Capstone Project - Agentic AI Course 2026*
