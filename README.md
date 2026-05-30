# RAG Knowledge Management System

> Query your organisation's documents using natural language. Powered by LangChain, HuggingFace, and LLM of choice.

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
  Ingestion Pipeline
  (chunks + embeds documents locally)
        ↓
  Chroma Vector Database
  (stores embeddings on your machine)
        ↓
  Your Query
        ↓
  Retrieval Pipeline
  (finds relevant chunks locally)
        ↓
  Claude / Local LLM
  (generates a precise answer)
        ↓
  Answer
```

---

## Key Features

- **Natural language querying** — ask questions the way you think them
- **Multi-document support** — load entire folders of PDFs at once
- **Privacy-conscious** — embeddings generated locally via HuggingFace; only relevant chunks sent to the LLM
- **Fully local option** — swap Claude for Ollama to keep everything on your machine
- **Configurable retrieval** — tune similarity thresholds for precision vs. recall
- **Source attribution** — every answer shows which document and page it came from

---

## Tech Stack

| Component | Technology |
|---|---|
| Framework | LangChain |
| Embeddings | HuggingFace (`multi-qa-mpnet-base-dot-v1`) |
| Vector Database | Chroma |
| LLM | Claude (Anthropic) or Ollama (local) |
| Document Loaders | LangChain Community |
| Language | Python 3.11 |

---

## Project Structure

```
rag-knowledge-management/
├── src/
│   ├── ingestion_pipeline.py    # Loads, chunks, and embeds documents
│   └── retrieval_pipeline.py   # Queries the knowledge base
├── docs/                        # Add your documents here (gitignored)
├── db/                          # Chroma vector database (auto-generated, gitignored)
├── requirements.txt
├── .env.example                 # Template for environment variables
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
pip install -r requirements.txt
```

### 4. Set up your environment variables

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Add your Anthropic API key to `.env`:

```
ANTHROPIC_API_KEY=your-api-key-here
```

Get your API key at [console.anthropic.com](https://console.anthropic.com)

### 5. Add your documents

Place your PDF or TXT files in the `docs/` folder:

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

---

## Privacy Options

| Setup | Embeddings | LLM | Data Exposure |
|---|---|---|---|
| Default | Local (HuggingFace) | Claude (Anthropic) | Only relevant chunks sent to Anthropic |
| Fully Local | Local (HuggingFace) | Ollama (on-device) | Nothing leaves your machine |

For fully local setup, install [Ollama](https://ollama.com) and replace the Claude model in `retrieval_pipeline.py`:

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

- [x] Document ingestion pipeline
- [x] Vector database storage (Chroma)
- [x] Basic retrieval pipeline
- [x] Claude LLM integration
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