# RAG Knowledge Management System

> Query your organisation's documents using natural language. Powered by LangChain, HuggingFace, and Claude.

---

## The Problem

Organisations accumulate enormous amounts of knowledge — reports, SOPs, guidelines, manuals, policies — but that knowledge is effectively locked away. Staff waste hours hunting through folders, re-reading entire documents to find a single paragraph, or reinventing the wheel because past work is too hard to access.

**The result:** Institutional knowledge that exists but cannot be used efficiently.

---

## The Solution

This RAG (Retrieval-Augmented Generation) pipeline transforms your document library into a queryable knowledge base. Instead of searching through hundreds of pages, staff ask questions in plain English and get precise, sourced answers drawn directly from your documents.

**Example queries:**
- *"When was this KPI first introduced and what was the rationale?"*
- *"How is this metric calculated and what data sources does it use?"*
- *"In which projects or reports is metric X referenced?"*
- *"What methodology was used to measure outcome Y?"*
- *"What does our framework say about indicator Z?"*
- *"Which documents reference this evaluation approach?"*

---

## How It Works

```
Your Documents (PDF, TXT)
        ↓
  Unstructured.io API
  (partitions + chunks documents by structure)
        ↓
  HuggingFace Embeddings
  (converts chunks to vectors locally)
        ↓
  Chroma Vector Database
  (stores embeddings on your machine)
        ↓
  Your Query
        ↓
  Retrieval Pipeline
  (finds relevant chunks locally)
        ↓
  Claude (Anthropic)
  (generates a precise answer)
        ↓
  Answer
```

---

## Key Features

- **Natural language querying** — ask questions the way you think them
- **Multi-document support** — load entire folders of PDFs at once
- **Structure-aware chunking** — documents chunked by title and section, not arbitrary character counts
- **Advanced document parsing** — tables, multi-column layouts, and images handled via Unstructured.io API
- **Privacy-conscious** — embeddings generated locally via HuggingFace; only relevant chunks sent to Claude
- **Fully local option** — swap Claude for Ollama to keep LLM responses on your machine
- **Configurable retrieval** — tune similarity thresholds for precision vs. recall
- **Source attribution** — every answer shows which document and page it came from

---

## Tech Stack

| Component | Technology |
|---|---|
| Framework | LangChain |
| Document Parsing | Unstructured.io API |
| Chunking Strategy | By title (via Unstructured.io) |
| Embeddings | HuggingFace (`multi-qa-mpnet-base-dot-v1`) |
| Vector Database | Chroma |
| LLM | Claude (Anthropic) or Ollama (local) |
| Language | Python 3.11 |

---

## Why Unstructured.io API?

This project uses the **Unstructured.io cloud API** rather than local document parsing libraries. This approach:

- Works on any OS and hardware — no local compilation required
- Handles complex PDFs including tables, images, and multi-column layouts
- Chunks documents intelligently by title and section structure
- Eliminates dependency issues with `poppler`, `pikepdf`, `llvmlite`, and `tesseract`

All heavy document processing happens on Unstructured.io's servers. Your machine handles embeddings and retrieval locally.

---

## Dependencies

```bash
pip install langchain langchain-community langchain-chroma langchain-anthropic
pip install langchain-unstructured
pip install langchain-huggingface sentence-transformers
pip install python-dotenv
```

---

## Project Structure

```
rag-knowledge-management/
├── src/
│   ├── ingestion_pipeline.py        # Loads, chunks, and embeds documents
│   ├── retrieval_pipeline.py        # Basic retrieval and answer generation
│   └── history_aware_generation.py  # Conversational retrieval with chat history
├── docs/                            # Add your documents here (gitignored)
├── db/                              # Chroma vector database (auto-generated, gitignored)
├── requirements.txt
├── .env.example                     # Template for environment variables
├── .gitignore
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/tiwipaddler/RAG-Knowledge-Management.git
cd RAG-Knowledge-Management
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install langchain langchain-community langchain-chroma langchain-anthropic
pip install langchain-unstructured
pip install langchain-huggingface sentence-transformers
pip install python-dotenv
```

### 4. Set up your environment variables

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Add your API keys to `.env`:

```
ANTHROPIC_API_KEY=your-anthropic-api-key
UNSTRUCTURED_API_KEY=your-unstructured-api-key
```

- Get your Anthropic API key at [console.anthropic.com](https://console.anthropic.com)
- Get your Unstructured API key at [unstructured.io](https://unstructured.io) (free tier available)

### 5. Add your documents

Place your PDF files in the `docs/` folder:

```bash
cp your-documents.pdf docs/
```

### 6. Run the ingestion pipeline

This processes your documents and builds the vector database:

```bash
python3 src/ingestion_pipeline.py
```

### 7. Query your knowledge base

```bash
python3 src/retrieval_pipeline.py
```

### 8. Run conversational mode

```bash
python3 src/history_aware_generation.py
```

---

## Privacy Options

| Setup | Document Parsing | Embeddings | LLM | Data Exposure |
|---|---|---|---|---|
| Default | Unstructured.io API | Local (HuggingFace) | Claude (Anthropic) | Documents sent to Unstructured.io, chunks sent to Anthropic |
| Fully Local LLM | Unstructured.io API | Local (HuggingFace) | Ollama (on-device) | Documents sent to Unstructured.io only |

For fully local LLM setup, install [Ollama](https://ollama.com) and replace Claude in `retrieval_pipeline.py`:

```python
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.2")
```

---

## Use Cases

This pipeline works for any organisation that stores knowledge in documents:

- **Turn SOPs into a searchable assistant** — staff get instant answers without reading entire manuals
- **Make past work reusable** — query reports, evaluations, and deliverables to surface relevant findings
- **Accelerate onboarding** — new staff query company knowledge independently from day one
- **Centralise institutional memory** — prevent knowledge loss when staff leave

---

## Status

🚧 **This project is under active development.**

- [x] Document ingestion pipeline (Unstructured.io API)
- [x] Structure-aware chunking by title
- [x] Vector database storage (Chroma)
- [x] Basic retrieval pipeline
- [x] Claude LLM integration
- [x] Conversational retrieval with chat history
- [ ] Web interface
- [ ] Multi-user support
- [ ] Metadata filtering by document type, date, or theme
- [ ] Automatic document ingestion on file drop

---

## Contributing

Contributions are welcome. Please open an issue first to discuss what you would like to change.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

For questions or collaboration enquiries, open an issue on this repository.