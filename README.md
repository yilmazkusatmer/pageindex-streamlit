# PageIndex — Vectorless RAG vs Traditional RAG

Interactive demo comparing **PageIndex (Vectorless, Reasoning-based RAG)** with **Traditional Vector-RAG** side by side.

Upload any PDF, enter your OpenAI API key, and see the difference.

## What is PageIndex?

Traditional RAG splits documents into chunks and finds similar ones via embeddings. **PageIndex** builds a hierarchical tree structure and lets an LLM navigate it — like a human using a table of contents.

| | Traditional Vector-RAG | PageIndex (Vectorless) |
|---|---|---|
| **Indexing** | Chunking + Embeddings | Hierarchical tree structure |
| **Retrieval** | Cosine similarity (blind) | LLM reasoning (targeted) |
| **Context** | Isolated chunks | Structured document context |
| **Explainability** | Similarity scores | Traceable reasoning path |
| **Infrastructure** | Vector DB required | No DB needed |

## Live Demo

Try it: [pageindex-streamlit on Streamlit Cloud](https://pageindex-streamlit.streamlit.app)

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How It Works

1. **Upload a PDF** and enter your OpenAI API key
2. **Indexing** runs both pipelines: PageIndex builds a tree, Vector-RAG creates chunk embeddings
3. **Ask a question** and compare answers side by side
4. **Explore details**: reasoning path (PageIndex) vs. retrieved chunks (Vector-RAG)

## Tech Stack

- [PageIndex](https://github.com/nicekate/pageindex) — Vectorless RAG library
- [Streamlit](https://streamlit.io) — Web UI
- [OpenAI API](https://platform.openai.com) — LLM + Embeddings
- [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) — Agent orchestration

## License

MIT
