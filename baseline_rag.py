"""
Baseline Vector-RAG — Traditional chunking + embedding approach.
Session-based: no file cache, everything in memory.
"""
import json

import numpy as np
import PyPDF2
from openai import OpenAI

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBED_MODEL = "text-embedding-3-small"
TOP_K = 5


def extract_pdf_text(pdf_file) -> list[dict]:
    """Extract text per page from a PDF file object."""
    pages = []
    reader = PyPDF2.PdfReader(pdf_file)
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": i, "text": text})
    return pages


def chunk_pages(pages: list[dict], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[dict]:
    """Split page texts into overlapping chunks."""
    chunks = []
    for page_info in pages:
        text = page_info["text"]
        page_num = page_info["page"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "page": page_num,
                    "char_start": start,
                })
            start += chunk_size - overlap
    return chunks


def get_embeddings(texts: list[str], client: OpenAI) -> np.ndarray:
    """Get embeddings from OpenAI API in batches."""
    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_embeddings.extend([e.embedding for e in response.data])
    return np.array(all_embeddings)


def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and document vectors."""
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)
    return doc_norms @ query_norm


class BaselineRAG:
    """Simple vector-based RAG pipeline. Session-based, no file persistence."""

    def __init__(self, api_key: str, model: str = "gpt-5.4-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.chunks = []
        self.embeddings = None
        self.page_count = 0

    def index(self, pdf_file) -> dict:
        """Index a PDF file object: extract, chunk, embed."""
        pages = extract_pdf_text(pdf_file)
        self.page_count = len(pages)
        self.chunks = chunk_pages(pages)
        texts = [c["text"] for c in self.chunks]
        self.embeddings = get_embeddings(texts, self.client)
        return {
            "chunks": len(self.chunks),
            "pages": self.page_count,
        }

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Retrieve top-k most similar chunks for a query."""
        if self.embeddings is None:
            raise RuntimeError("Call index() first.")
        query_emb = get_embeddings([query], self.client)[0]
        scores = cosine_similarity(query_emb, self.embeddings)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(scores[idx])
            results.append(chunk)
        return results

    def query(self, question: str, top_k: int = TOP_K) -> dict:
        """Full RAG pipeline: retrieve chunks, generate answer."""
        retrieved = self.retrieve(question, top_k)
        context = "\n\n---\n\n".join(
            f"[Page {c['page']}, Score: {c['score']:.3f}]\n{c['text']}"
            for c in retrieved
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a document QA assistant. "
                        "Answer the question ONLY based on the given context. "
                        "Be concise and precise."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
        )
        answer = response.choices[0].message.content
        pages_used = sorted(set(c["page"] for c in retrieved))
        return {
            "answer": answer,
            "chunks": retrieved,
            "pages_used": pages_used,
            "total_chunks_searched": len(self.chunks),
        }
