"""
PageIndex Demo: Vectorless RAG vs Traditional Vector-RAG
Interactive comparison — upload your own PDF, bring your own API key.
"""
import sys
import os
import json
import time
import asyncio
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path("pageindex-repo").resolve()))

from pageindex import PageIndexClient
from agents import Agent, Runner, function_tool, set_tracing_disabled
from baseline_rag import BaselineRAG

set_tracing_disabled(True)

# --- Page Setup ---
st.set_page_config(page_title="PageIndex — RAG Comparison", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
    .tree-node {
        padding: 4px 8px;
        margin: 2px 0;
        border-radius: 4px;
        font-size: 0.85em;
        border-left: 3px solid #ddd;
    }
    .tree-node-active {
        background-color: #d4edda;
        border-left: 3px solid #28a745;
        font-weight: bold;
    }
    .tree-indent-0 { margin-left: 0px; }
    .tree-indent-1 { margin-left: 20px; }
    .tree-indent-2 { margin-left: 40px; }
    .tree-indent-3 { margin-left: 60px; }
    .reasoning-step {
        background-color: #f8f9fa;
        border-left: 3px solid #6c757d;
        padding: 8px 12px;
        margin: 8px 0;
        border-radius: 4px;
        font-size: 0.9em;
    }
    .tool-step {
        background-color: #e7f3ff;
        border-left: 3px solid #0d6efd;
        padding: 8px 12px;
        margin: 8px 0;
        border-radius: 4px;
        font-size: 0.9em;
    }
    .answer-box {
        background-color: #f0fff0;
        border-left: 3px solid #28a745;
        padding: 12px 16px;
        margin: 12px 0;
        border-radius: 4px;
    }
    .chunk-box {
        background-color: #fff8f0;
        border-left: 3px solid #fd7e14;
        padding: 8px 12px;
        margin: 6px 0;
        border-radius: 4px;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.header("Settings")

    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")

    model = st.selectbox("Model", [
        "gpt-5.4-mini",
        "gpt-5-mini",
        "gpt-4.1-mini",
        "gpt-4o-mini",
        "gpt-4o",
    ], index=0)

    st.divider()

    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF document", type=["pdf"])

    if uploaded_file and api_key:
        if st.button("Index Document", type="primary"):
            with st.spinner("Indexing in progress... (may take 2-5 min)"):
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                try:
                    os.environ["OPENAI_API_KEY"] = api_key

                    # PageIndex indexing
                    workspace = Path(tempfile.mkdtemp(prefix="pageindex_"))
                    client = PageIndexClient(
                        workspace=workspace,
                        model=model,
                        retrieve_model=model,
                    )
                    doc_id = client.index(tmp_path)
                    client._ensure_doc_loaded(doc_id)

                    st.session_state["pi_client"] = client
                    st.session_state["pi_doc_id"] = doc_id
                    st.session_state["doc_meta"] = json.loads(client.get_document(doc_id))
                    st.session_state["structure"] = json.loads(client.get_document_structure(doc_id))

                    # Baseline RAG indexing
                    rag = BaselineRAG(api_key=api_key, model=model)
                    uploaded_file.seek(0)
                    rag_info = rag.index(uploaded_file)

                    st.session_state["rag"] = rag
                    st.session_state["rag_info"] = rag_info
                    st.session_state["indexed"] = True
                    st.session_state["model"] = model

                    st.success(
                        f"Indexed! PageIndex: {st.session_state['doc_meta'].get('page_count')} pages, "
                        f"Baseline: {rag_info['chunks']} chunks"
                    )
                except Exception as e:
                    st.error(f"Indexing error: {e}")
                finally:
                    os.unlink(tmp_path)

    elif not api_key:
        st.info("Please enter your OpenAI API key.")
    elif not uploaded_file:
        st.info("Please upload a PDF document.")

    st.divider()
    st.caption("Session-based — no data is stored.")


# --- Helper Functions ---

def flatten_tree(nodes, depth=0, result=None):
    if result is None:
        result = []
    for node in nodes:
        title = node.get("title", "?")
        if len(title) > 90:
            title = title[:87] + "..."
        start = node.get("start_index", "?")
        end = node.get("end_index", "?")
        result.append({
            "title": title,
            "start": start,
            "end": end,
            "depth": depth,
        })
        if node.get("nodes") and depth < 3:
            flatten_tree(node["nodes"], depth + 1, result)
    return result


def render_tree(flat_nodes, visited_pages=None):
    if visited_pages is None:
        visited_pages = set()
    html_parts = []
    for node in flat_nodes:
        depth = min(node["depth"], 3)
        start = node["start"]
        end = node["end"]
        node_pages = set()
        if isinstance(start, int) and isinstance(end, int):
            node_pages = set(range(start, end + 1))
        is_active = bool(node_pages & visited_pages)
        css_class = "tree-node tree-node-active" if is_active else "tree-node"
        indent_class = f"tree-indent-{depth}"
        marker = "&#x1F7E2; " if is_active else ""
        html_parts.append(
            f'<div class="{css_class} {indent_class}">'
            f'{marker}<b>{node["title"]}</b> '
            f'<span style="color: #888;">p. {start}-{end}</span>'
            f'</div>'
        )
    return "\n".join(html_parts)


def run_agent_query(client, doc_id, question, model):
    """Run PageIndex agent query with reasoning steps."""

    @function_tool
    def get_document() -> str:
        """Get document metadata: status, page count, name, description."""
        return client.get_document(doc_id)

    @function_tool
    def get_document_structure() -> str:
        """Get document tree structure (without text content)."""
        return client.get_document_structure(doc_id)

    @function_tool
    def get_page_content(pages: str) -> str:
        """Get text content of specific pages. Format: '5-7', '3,8', or '12'."""
        return client.get_page_content(doc_id, pages)

    agent = Agent(
        name="PageIndex-Explorer",
        instructions=(
            "You are a document QA assistant.\n"
            "TOOL USAGE:\n"
            "- Call get_document() to check status and page count.\n"
            "- Call get_document_structure() to identify relevant page ranges.\n"
            "- Call get_page_content(pages='5-7') with tight ranges.\n"
            "- Before each tool call, explain in 1-2 sentences why you are "
            "reading that section.\n"
            "Answer based only on tool output. Be concise."
        ),
        tools=[get_document, get_document_structure, get_page_content],
        model=model,
    )

    result = asyncio.run(Runner.run(agent, question))

    steps = []
    visited_pages = set()
    tool_labels = {
        "get_document": "Load document info",
        "get_document_structure": "Read tree structure",
        "get_page_content": "Fetch pages",
    }

    for item in result.new_items:
        if item.type == "message_output_item":
            text = ""
            for part in item.raw_item.content:
                if hasattr(part, "text"):
                    text += part.text
            if text.strip():
                steps.append({"type": "reasoning", "text": text.strip()})
        elif item.type == "tool_call_item":
            raw = item.raw_item
            label = tool_labels.get(raw.name, raw.name)
            args_str = getattr(raw, "arguments", "{}")
            if "pages" in str(args_str):
                try:
                    pages_str = json.loads(args_str).get("pages", "")
                    label += f" (p. {pages_str})"
                    for part in pages_str.split(","):
                        part = part.strip()
                        if "-" in part:
                            s, e = part.split("-")
                            visited_pages.update(range(int(s), int(e) + 1))
                        else:
                            visited_pages.add(int(part))
                except (json.JSONDecodeError, ValueError):
                    pass
            steps.append({"type": "tool", "text": label})

    return result.final_output, steps, visited_pages


# --- Main Content ---

st.title("Vectorless RAG vs. Traditional RAG")
st.caption("Upload a PDF, compare PageIndex with Vector-RAG — side by side.")

# Intro
with st.expander("What is PageIndex?", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Traditional Vector-RAG")
        st.markdown("""
        1. Split document into chunks
        2. Store chunks as vectors
        3. Query vector -> most similar chunks

        **Problem:** Similarity ≠ Relevance
        """)
    with col_b:
        st.markdown("### PageIndex (Vectorless)")
        st.markdown("""
        1. Build hierarchical tree structure
        2. LLM navigates via reasoning
        3. Fetch only relevant pages

        **Advantage:** Traceable reasoning path
        """)

st.divider()

# --- Query Interface ---
if not st.session_state.get("indexed"):
    st.info("Please upload a PDF and index it using the sidebar to get started.")
    st.stop()

doc_meta = st.session_state["doc_meta"]
structure = st.session_state["structure"]
flat_nodes = flatten_tree(structure if isinstance(structure, list) else [structure])

st.success(
    f"**{doc_meta.get('doc_name', 'Document')}** — "
    f"{doc_meta.get('page_count', '?')} pages indexed | "
    f"{st.session_state['rag_info']['chunks']} chunks created"
)

question = st.text_input("Ask a question about the document:", placeholder="e.g. What are the main findings?")

if st.button("Run Comparison", type="primary", disabled=not question):
    st.divider()

    tab_comparison, tab_pageindex, tab_baseline = st.tabs([
        "Comparison", "PageIndex (Detail)", "Baseline RAG (Detail)"
    ])

    # Run both pipelines
    with st.spinner("Running Baseline Vector-RAG..."):
        rag = st.session_state["rag"]
        t0 = time.time()
        baseline_result = rag.query(question)
        baseline_time = time.time() - t0

    with st.spinner("PageIndex agent navigating the tree..."):
        client = st.session_state["pi_client"]
        doc_id = st.session_state["pi_doc_id"]
        t0 = time.time()
        pi_answer, pi_steps, pi_visited_pages = run_agent_query(
            client, doc_id, question, st.session_state["model"]
        )
        pageindex_time = time.time() - t0

    # --- Tab: Comparison ---
    with tab_comparison:
        st.subheader("Side-by-Side Comparison")

        m1, m2, m3, m4 = st.columns(4)
        page_count = doc_meta.get("page_count", "?")
        with m1:
            st.metric("Baseline: Pages read", f"{len(baseline_result['pages_used'])} / {page_count}")
        with m2:
            st.metric("PageIndex: Pages read", f"{len(pi_visited_pages)} / {page_count}")
        with m3:
            st.metric("Baseline: Duration", f"{baseline_time:.1f}s")
        with m4:
            st.metric("PageIndex: Duration", f"{pageindex_time:.1f}s")

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### Baseline Vector-RAG")
            st.markdown(
                f'<div class="answer-box">{baseline_result["answer"]}</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                f"Chunks searched: {baseline_result['total_chunks_searched']} | "
                f"Top-{len(baseline_result['chunks'])} used | "
                f"Pages: {', '.join(str(p) for p in baseline_result['pages_used'])}"
            )

        with col_right:
            st.markdown("#### PageIndex (Vectorless RAG)")
            st.markdown(
                f'<div class="answer-box">{pi_answer}</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                f"Reasoning steps: {len(pi_steps)} | "
                f"Pages: {', '.join(str(p) for p in sorted(pi_visited_pages)) if pi_visited_pages else 'n/a'}"
            )

        st.info(
            f"**Baseline RAG** blindly searched {baseline_result['total_chunks_searched']} chunks "
            f"via cosine similarity and picked the top {len(baseline_result['chunks'])}. "
            f"**PageIndex** navigated the document tree and selectively retrieved "
            f"{len(pi_visited_pages)} pages."
        )

    # --- Tab: PageIndex Detail ---
    with tab_pageindex:
        left_col, right_col = st.columns([2, 3])

        with left_col:
            st.subheader(f"Document Structure ({page_count} pages)")
            st.caption("Visited nodes are highlighted in green")
            st.markdown(render_tree(flat_nodes, pi_visited_pages), unsafe_allow_html=True)

        with right_col:
            st.subheader("Reasoning Path")
            for i, step in enumerate(pi_steps, 1):
                if step["type"] == "reasoning":
                    st.markdown(
                        f'<div class="reasoning-step">&#x1F4AD; <b>Step {i}:</b> {step["text"]}</div>',
                        unsafe_allow_html=True,
                    )
                elif step["type"] == "tool":
                    st.markdown(
                        f'<div class="tool-step">&#x1F527; <b>Step {i}:</b> {step["text"]}</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("#### Answer")
            st.markdown(f'<div class="answer-box">{pi_answer}</div>', unsafe_allow_html=True)

    # --- Tab: Baseline Detail ---
    with tab_baseline:
        st.subheader("Retrieval Result")
        st.caption(
            f"{baseline_result['total_chunks_searched']} chunks | "
            f"Chunk size: 500 chars, overlap: 100 | "
            f"Embedding: text-embedding-3-small"
        )

        for i, chunk in enumerate(baseline_result["chunks"], 1):
            score_pct = chunk["score"] * 100
            st.markdown(
                f'<div class="chunk-box">'
                f'<b>Chunk #{i}</b> — Page {chunk["page"]} — '
                f'Similarity: {score_pct:.1f}%<br>'
                f'<code>{chunk["text"][:300]}{"..." if len(chunk["text"]) > 300 else ""}</code>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("#### Answer")
        st.markdown(
            f'<div class="answer-box">{baseline_result["answer"]}</div>',
            unsafe_allow_html=True,
        )
