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
    st.header("Einstellungen")

    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")

    model = st.selectbox("Model", [
        "gpt-5.4-mini",
        "gpt-5-mini",
        "gpt-4.1-mini",
        "gpt-4o-mini",
        "gpt-4o",
    ], index=0)

    st.divider()

    st.header("PDF hochladen")
    uploaded_file = st.file_uploader("PDF-Dokument waehlen", type=["pdf"])

    if uploaded_file and api_key:
        if st.button("Dokument indexieren", type="primary"):
            with st.spinner("Indexierung laeuft... (kann 2-5 Min dauern)"):
                # Save uploaded file to temp
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
                        f"Indexiert! PageIndex: {st.session_state['doc_meta'].get('page_count')} Seiten, "
                        f"Baseline: {rag_info['chunks']} Chunks"
                    )
                except Exception as e:
                    st.error(f"Fehler bei der Indexierung: {e}")
                finally:
                    os.unlink(tmp_path)

    elif not api_key:
        st.info("Bitte OpenAI API Key eingeben.")
    elif not uploaded_file:
        st.info("Bitte ein PDF hochladen.")

    st.divider()
    st.caption("Session-basiert — keine Daten werden gespeichert.")


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
            f'<span style="color: #888;">S. {start}-{end}</span>'
            f'</div>'
        )
    return "\n".join(html_parts)


def run_agent_query(client, doc_id, question, model):
    """PageIndex Agent-Query mit Reasoning-Schritten."""

    @function_tool
    def get_document() -> str:
        """Dokument-Metadaten: Status, Seitenzahl, Name, Beschreibung."""
        return client.get_document(doc_id)

    @function_tool
    def get_document_structure() -> str:
        """Baumstruktur des Dokuments (ohne Text)."""
        return client.get_document_structure(doc_id)

    @function_tool
    def get_page_content(pages: str) -> str:
        """Textinhalt bestimmter Seiten. Format: '5-7', '3,8', oder '12'."""
        return client.get_page_content(doc_id, pages)

    agent = Agent(
        name="PageIndex-Explorer",
        instructions=(
            "Du bist ein Dokumenten-QA-Assistent.\n"
            "TOOL-NUTZUNG:\n"
            "- Rufe get_document() auf, um Status und Seitenzahl zu pruefen.\n"
            "- Rufe get_document_structure() auf, um relevante Seitenbereiche zu finden.\n"
            "- Rufe get_page_content(pages='5-7') mit engen Bereichen auf.\n"
            "- Erklaere vor JEDEM Tool-Aufruf in 1-2 Saetzen, "
            "warum du diesen Abschnitt liest.\n"
            "Antworte nur basierend auf Tool-Output. Sei praezise. Antworte auf Deutsch."
        ),
        tools=[get_document, get_document_structure, get_page_content],
        model=model,
    )

    result = asyncio.run(Runner.run(agent, question))

    steps = []
    visited_pages = set()
    tool_labels = {
        "get_document": "Dokument-Info laden",
        "get_document_structure": "Baumstruktur lesen",
        "get_page_content": "Seiten abrufen",
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
                    label += f" (S. {pages_str})"
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
with st.expander("Was ist PageIndex?", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Traditional Vector-RAG")
        st.markdown("""
        1. Dokument in Chunks aufteilen
        2. Chunks als Vektoren speichern
        3. Query-Vektor -> aehnlichste Chunks

        **Problem:** Similarity ≠ Relevance
        """)
    with col_b:
        st.markdown("### PageIndex (Vectorless)")
        st.markdown("""
        1. Hierarchische Baumstruktur erstellen
        2. LLM navigiert per Reasoning
        3. Gezielt relevante Seiten abrufen

        **Vorteil:** Nachvollziehbarer Pfad
        """)

st.divider()

# --- Query Interface ---
if not st.session_state.get("indexed"):
    st.info("Bitte links ein PDF hochladen und indexieren, um zu starten.")
    st.stop()

doc_meta = st.session_state["doc_meta"]
structure = st.session_state["structure"]
flat_nodes = flatten_tree(structure if isinstance(structure, list) else [structure])

st.success(
    f"**{doc_meta.get('doc_name', 'Dokument')}** — "
    f"{doc_meta.get('page_count', '?')} Seiten indexiert | "
    f"{st.session_state['rag_info']['chunks']} Chunks erstellt"
)

question = st.text_input("Frage an das Dokument:", placeholder="z.B. Welche KI-Systeme sind verboten?")

if st.button("Vergleich starten", type="primary", disabled=not question):
    st.divider()

    tab_comparison, tab_pageindex, tab_baseline = st.tabs([
        "Vergleich", "PageIndex (Detail)", "Baseline RAG (Detail)"
    ])

    # Run both pipelines
    with st.spinner("Baseline Vector-RAG laeuft..."):
        rag = st.session_state["rag"]
        t0 = time.time()
        baseline_result = rag.query(question)
        baseline_time = time.time() - t0

    with st.spinner("PageIndex Agent navigiert..."):
        client = st.session_state["pi_client"]
        doc_id = st.session_state["pi_doc_id"]
        t0 = time.time()
        pi_answer, pi_steps, pi_visited_pages = run_agent_query(
            client, doc_id, question, st.session_state["model"]
        )
        pageindex_time = time.time() - t0

    # --- Tab: Comparison ---
    with tab_comparison:
        st.subheader("Side-by-Side Vergleich")

        m1, m2, m3, m4 = st.columns(4)
        page_count = doc_meta.get("page_count", "?")
        with m1:
            st.metric("Baseline: Seiten", f"{len(baseline_result['pages_used'])} / {page_count}")
        with m2:
            st.metric("PageIndex: Seiten", f"{len(pi_visited_pages)} / {page_count}")
        with m3:
            st.metric("Baseline: Dauer", f"{baseline_time:.1f}s")
        with m4:
            st.metric("PageIndex: Dauer", f"{pageindex_time:.1f}s")

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### Baseline Vector-RAG")
            st.markdown(
                f'<div class="answer-box">{baseline_result["answer"]}</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                f"Chunks: {baseline_result['total_chunks_searched']} | "
                f"Top-{len(baseline_result['chunks'])} | "
                f"Seiten: {', '.join(str(p) for p in baseline_result['pages_used'])}"
            )

        with col_right:
            st.markdown("#### PageIndex (Vectorless RAG)")
            st.markdown(
                f'<div class="answer-box">{pi_answer}</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                f"Reasoning-Schritte: {len(pi_steps)} | "
                f"Seiten: {', '.join(str(p) for p in sorted(pi_visited_pages)) if pi_visited_pages else 'n/a'}"
            )

        st.info(
            f"**Baseline RAG** hat blind {baseline_result['total_chunks_searched']} Chunks "
            f"per Cosine-Similarity durchsucht und die Top-{len(baseline_result['chunks'])} genommen. "
            f"**PageIndex** hat den Dokumentbaum navigiert und gezielt "
            f"{len(pi_visited_pages)} Seiten abgerufen."
        )

    # --- Tab: PageIndex Detail ---
    with tab_pageindex:
        left_col, right_col = st.columns([2, 3])

        with left_col:
            st.subheader(f"Dokumentstruktur ({page_count} Seiten)")
            st.caption("Besuchte Knoten werden gruen markiert")
            st.markdown(render_tree(flat_nodes, pi_visited_pages), unsafe_allow_html=True)

        with right_col:
            st.subheader("Reasoning-Pfad")
            for i, step in enumerate(pi_steps, 1):
                if step["type"] == "reasoning":
                    st.markdown(
                        f'<div class="reasoning-step">&#x1F4AD; <b>Schritt {i}:</b> {step["text"]}</div>',
                        unsafe_allow_html=True,
                    )
                elif step["type"] == "tool":
                    st.markdown(
                        f'<div class="tool-step">&#x1F527; <b>Schritt {i}:</b> {step["text"]}</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("#### Antwort")
            st.markdown(f'<div class="answer-box">{pi_answer}</div>', unsafe_allow_html=True)

    # --- Tab: Baseline Detail ---
    with tab_baseline:
        st.subheader("Retrieval-Ergebnis")
        st.caption(
            f"{baseline_result['total_chunks_searched']} Chunks | "
            f"Chunk-Groesse: 500 Zeichen, Overlap: 100 | "
            f"Embedding: text-embedding-3-small"
        )

        for i, chunk in enumerate(baseline_result["chunks"], 1):
            score_pct = chunk["score"] * 100
            st.markdown(
                f'<div class="chunk-box">'
                f'<b>Chunk #{i}</b> — Seite {chunk["page"]} — '
                f'Similarity: {score_pct:.1f}%<br>'
                f'<code>{chunk["text"][:300]}{"..." if len(chunk["text"]) > 300 else ""}</code>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("#### Antwort")
        st.markdown(
            f'<div class="answer-box">{baseline_result["answer"]}</div>',
            unsafe_allow_html=True,
        )
