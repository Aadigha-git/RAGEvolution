#!/usr/bin/env python3
"""Main Streamlit dashboard entrypoint."""

from __future__ import annotations

import os
import time
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from supabase import Client, create_client

from agent_architecture import AgentState, agent_graph
from rag_engine import advanced_rag_query, naive_rag_query

V1_ARCHITECTURE = "v1.0: Naive RAG (Baseline)"
V2_ARCHITECTURE = "v2.0: Optimized Advanced RAG"
V3_ARCHITECTURE = "v3.0: Agentic Framework (FSM)"
ARCHITECTURE_OPTIONS = [V1_ARCHITECTURE, V2_ARCHITECTURE, V3_ARCHITECTURE]

FSM_STEP_LABELS = {
    "classify_intent_node": "Step 1: Classifying user intent…",
    "lookup_specialist_node": "Step 2: Running Direct Database Lookup Specialist",
    "clinical_analyst_node": "Step 2: Invoking Advanced RAG Pipeline",
    "guardrail_node": "Step 3: Running Guardrail Evaluation",
}


def load_env() -> tuple[str, str, str]:
    load_dotenv()
    return (
        os.getenv("SUPABASE_URL", "").strip(),
        os.getenv("SUPABASE_KEY", "").strip(),
        os.getenv("GEMINI_API_KEY", "").strip(),
    )


def get_supabase_client(url: str, key: str) -> Client | None:
    if not url or not key:
        return None
    return create_client(url, key)


def fetch_fda_chunk_count(supabase: Client | None) -> tuple[int | None, str | None]:
    if supabase is None:
        return None, "Missing SUPABASE_URL or SUPABASE_KEY in environment."

    try:
        response = (
            supabase.table("fda_document_chunks")
            .select("*", count="exact", head=True)
            .execute()
        )
        return response.count or 0, None
    except Exception as exc:  # noqa: BLE001
        return None, f"Count query failed: {exc}"


def fetch_fda_chunk_count_by_version(
    supabase: Client | None,
    pipeline_version: str,
) -> tuple[int | None, str | None]:
    if supabase is None:
        return None, "Missing SUPABASE_URL or SUPABASE_KEY in environment."

    try:
        response = (
            supabase.table("fda_document_chunks")
            .select("*", count="exact", head=True)
            .filter("metadata->>pipeline_version", "eq", pipeline_version)
            .execute()
        )
        return response.count or 0, None
    except Exception as exc:  # noqa: BLE001
        return None, f"Version count query failed: {exc}"


def credentials_ready(supabase_url: str, supabase_key: str, gemini_api_key: str) -> bool:
    return bool(supabase_url and supabase_key and gemini_api_key)


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_latency_seconds" not in st.session_state:
        st.session_state.last_latency_seconds = None


def _format_fsm_step(node_name: str, state: dict[str, Any]) -> str:
    if node_name == "classify_intent_node":
        return f"Step 1: Intent Classified as `{state.get('intent', 'UNKNOWN')}`"
    if node_name == "lookup_specialist_node":
        return "Step 2: Direct Database Lookup Specialist completed"
    if node_name == "clinical_analyst_node":
        return "Step 2: Advanced RAG Pipeline completed"
    if node_name == "guardrail_node":
        passed = state.get("safety_check_passed", False)
        verdict = "Pass" if passed else "Review Required"
        return f"Step 3: Guardrail Evaluation {verdict}"
    return FSM_STEP_LABELS.get(node_name, f"{node_name} completed")


def run_agent_with_status(user_query: str) -> dict[str, Any]:
    """Stream the compiled LangGraph agent and capture FSM step telemetry."""
    started = time.perf_counter()
    initial_state: AgentState = {
        "user_query": user_query,
        "intent": "",
        "context": [],
        "response": "",
        "history": [],
        "safety_check_passed": False,
    }

    accumulated: dict[str, Any] = dict(initial_state)
    fsm_steps: list[str] = []

    with st.status("Agent FSM executing…", expanded=True) as status:
        for event in agent_graph.stream(initial_state):
            for node_name, node_update in event.items():
                accumulated.update(node_update)
                step_label = _format_fsm_step(node_name, accumulated)
                fsm_steps.append(step_label)
                st.write(step_label)
                status.update(label=f"Running: {node_name}", state="running")

        status.update(label="Agent FSM complete", state="complete")

    latency_seconds = time.perf_counter() - started
    return {
        "answer": accumulated.get("response", ""),
        "response": accumulated.get("response", ""),
        "intent": accumulated.get("intent", ""),
        "context": accumulated.get("context", []),
        "retrieved_sources": accumulated.get("context", []),
        "safety_check_passed": accumulated.get("safety_check_passed", False),
        "fsm_steps": fsm_steps,
        "history": accumulated.get("history", []),
        "metrics": {
            "latency_seconds": latency_seconds,
            "pipeline": "v3.0_agentic_fsm",
        },
    }


def render_retrieval_expander(sources: list[dict[str, Any]], title: str) -> None:
    with st.expander(title, expanded=False):
        if not sources:
            st.info("No context blocks were retrieved for this response.")
            return

        for idx, source in enumerate(sources, start=1):
            section_name = source.get("section_name") or "unknown_section"
            brand_name = source.get("brand_name") or "N/A"
            generic_name = source.get("generic_name") or "N/A"
            similarity = source.get("similarity")
            cross_score = source.get("cross_score")
            similarity_label = (
                f"{similarity:.4f}" if isinstance(similarity, (int, float)) else "N/A"
            )
            cross_label = (
                f"{cross_score:.4f}" if isinstance(cross_score, (int, float)) else "N/A"
            )

            st.markdown(
                f"**Match {idx}** · `{section_name}` · "
                f"**{brand_name}** ({generic_name}) · "
                f"Similarity: `{similarity_label}` · Cross-score: `{cross_label}`"
            )
            st.text_area(
                label=f"context_block_{idx}",
                value=source.get("chunk_content", ""),
                height=180,
                disabled=True,
                label_visibility="collapsed",
            )
            if idx < len(sources):
                st.divider()


def render_fsm_trace(fsm_steps: list[str]) -> None:
    with st.expander("🧭 FSM Execution Trace", expanded=False):
        if not fsm_steps:
            st.info("No FSM steps recorded for this response.")
            return
        for step in fsm_steps:
            st.markdown(f"- {step}")


def render_assistant_message(result: dict[str, Any]) -> None:
    answer = result.get("answer") or result.get("response", "")
    metrics = result.get("metrics", {})
    sources = result.get("retrieved_sources") or result.get("context", [])
    fsm_steps = result.get("fsm_steps", [])

    st.markdown(answer)

    if fsm_steps:
        render_fsm_trace(fsm_steps)

    render_retrieval_expander(
        sources,
        title="🔍 Retrieved Context & Similarity Scores",
    )

    pipeline = metrics.get("pipeline", "unknown")
    if result.get("intent"):
        safety = result.get("safety_check_passed")
        safety_label = "Pass" if safety else "Review Required"
        st.caption(
            f"Pipeline: {pipeline} · Intent: {result.get('intent')} · "
            f"Guardrail: {safety_label}"
        )
    else:
        st.caption(f"Pipeline: {pipeline}")


def render_rag_chat(architecture_version: str) -> None:
    init_session_state()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_assistant_message(message)

    if prompt := st.chat_input("Ask about GLP-1 FDA label data…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                if architecture_version == V1_ARCHITECTURE:
                    with st.spinner("Running naive RAG pipeline…"):
                        result = naive_rag_query(prompt)
                elif architecture_version == V2_ARCHITECTURE:
                    with st.spinner("Running advanced RAG pipeline…"):
                        result = advanced_rag_query(prompt)
                elif architecture_version == V3_ARCHITECTURE:
                    result = run_agent_with_status(prompt)
                else:
                    raise ValueError("Selected architecture is not supported for chat.")

                render_assistant_message(result)

                latency = result.get("metrics", {}).get("latency_seconds")
                if isinstance(latency, (int, float)):
                    st.session_state.last_latency_seconds = latency

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result.get("answer") or result.get("response", ""),
                        "answer": result.get("answer") or result.get("response", ""),
                        "retrieved_sources": result.get("retrieved_sources", []),
                        "context": result.get("context", []),
                        "intent": result.get("intent"),
                        "safety_check_passed": result.get("safety_check_passed"),
                        "fsm_steps": result.get("fsm_steps", []),
                        "metrics": result.get("metrics", {}),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                error_text = f"Query failed: {exc}"
                st.error(error_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_text, "answer": error_text}
                )


def render_engineering_design_brief() -> None:
    """Landing section: engineering identity, platform rationale, architecture evolution."""
    st.markdown(
        """
        <style>
        .edb-hero-title {
            font-size: 3.15rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            line-height: 1.1;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 45%, #0ea5e9 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .edb-hero-subtitle {
            font-size: 1.2rem;
            font-weight: 500;
            color: #334155;
            line-height: 1.55;
            margin-bottom: 1.25rem;
            max-width: 920px;
        }
        .edb-badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 1.75rem;
        }
        .edb-badge {
            display: inline-block;
            padding: 0.28rem 0.75rem;
            border: 1px solid #cbd5e1;
            border-radius: 999px;
            background: #f8fafc;
            color: #334155;
            font-size: 0.78rem;
            font-weight: 500;
            letter-spacing: 0.01em;
        }
        .edb-card {
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.35rem 1.5rem;
            background: #ffffff;
            margin-bottom: 1.5rem;
        }
        .edb-card-title {
            font-size: 1.15rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #0d9488;
            margin-bottom: 0.75rem;
        }
        .edb-card-body {
            font-size: 0.98rem;
            line-height: 1.7;
            color: #334155;
            margin: 0;
        }
        .edb-section-heading {
            font-size: 1.85rem;
            font-weight: 750;
            letter-spacing: -0.02em;
            color: #1d4ed8;
            margin: 0.5rem 0 1.1rem 0;
        }
        .edb-phase-card {
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 1.15rem 1.2rem;
            background: #fafbfc;
            min-height: 220px;
        }
        .edb-phase-label {
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }
        .edb-phase-label.p1 { color: #dc2626; }
        .edb-phase-label.p2 { color: #d97706; }
        .edb-phase-label.p3 { color: #059669; }
        .edb-phase-title {
            font-size: 1.35rem;
            font-weight: 750;
            margin-bottom: 0.65rem;
        }
        .edb-phase-title.p1 { color: #991b1b; }
        .edb-phase-title.p2 { color: #b45309; }
        .edb-phase-title.p3 { color: #047857; }
        .edb-phase-flow {
            font-size: 0.88rem;
            line-height: 1.65;
            color: #475569;
            margin-bottom: 0.6rem;
        }
        .edb-phase-note {
            font-size: 0.78rem;
            font-style: italic;
            color: #94a3b8;
        }
        .edb-cta-heading {
            font-size: 1.75rem;
            font-weight: 800;
            color: #7c3aed;
            margin-top: 0.25rem;
            margin-bottom: 0.45rem;
        }
        .edb-cta-body {
            font-size: 0.95rem;
            color: #64748b;
            line-height: 1.6;
            margin-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="edb-hero-title">Building & Optimizing Enterprise AI Infrastructure</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="edb-hero-subtitle">A Step-by-Step Architectural Evolution from '
        "Naive RAG to an Intent-Driven Multi-Agent FSM Framework.</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="edb-badge-row">
            <span class="edb-badge">Python</span>
            <span class="edb-badge">Gemini Suite</span>
            <span class="edb-badge">Supabase halfvec(3072)</span>
            <span class="edb-badge">LangGraph</span>
            <span class="edb-badge">Streamlit</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="edb-card">
            <div class="edb-card-title">Platform Rationale · Why Streamlit?</div>
            <p class="edb-card-body">
                In an enterprise environment, rapid data prototyping and infrastructure
                observability are key. Streamlit allows me to construct a decoupled
                presentation layer directly over my live database execution threads.
                This platform serves as an interactive telemetry window. Allowing users to 
                visually trace how changing core chunking strategies and adding orchestration 
                layers alters retrieval latency and response accuracy in real time.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="edb-section-heading">Architecture Evolution</p>',
        unsafe_allow_html=True,
    )
    phase1, arrow1, phase2, arrow2, phase3 = st.columns([5, 0.4, 5, 0.4, 5])
    with phase1:
        st.markdown(
            """
            <div class="edb-phase-card">
                <div class="edb-phase-label p1">Phase 1</div>
                <div class="edb-phase-title p1">Naive RAG</div>
                <div class="edb-phase-flow">
                    Fixed Character Splitting (1000 chars)<br>
                    → Raw Vector Search<br>
                    → Monolithic Prompt
                </div>
                <div class="edb-phase-note">The Flawed Baseline</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with arrow1:
        st.markdown("<div style='text-align:center;padding-top:5.5rem;color:#94a3b8;'>→</div>", unsafe_allow_html=True)
    with phase2:
        st.markdown(
            """
            <div class="edb-phase-card">
                <div class="edb-phase-label p2">Phase 2</div>
                <div class="edb-phase-title p2">Advanced RAG</div>
                <div class="edb-phase-flow">
                    Semantic Phrase Boundary Grouping<br>
                    → 16-bit halfvec(3072) Precision Storage<br>
                    → Local Cross-Encoder Reranking (<code>mxbai-rerank</code>)
                </div>
                <div class="edb-phase-note">Optimized Retrieval Layer</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with arrow2:
        st.markdown("<div style='text-align:center;padding-top:5.5rem;color:#94a3b8;'>→</div>", unsafe_allow_html=True)
    with phase3:
        st.markdown(
            """
            <div class="edb-phase-card">
                <div class="edb-phase-label p3">Phase 3</div>
                <div class="edb-phase-title p3">Agentic FSM</div>
                <div class="edb-phase-flow">
                    LangGraph Intent Router<br>
                    → Direct Lookup Metadata Paths vs. Heavy Analytical Processing<br>
                    → LLM-as-a-Judge Guardrails
                </div>
                <div class="edb-phase-note">Intent-Driven Orchestration</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        '<p class="edb-cta-heading">🔬 Enter the Interactive Simulation Lab</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="edb-cta-body">Use the sidebar toggle to switch between architectural '
        "versions. Ask complex cross-relational questions to watch the pipeline logs adapt, "
        "display latency metrics, and reveal hidden retrieval flaws.</p>",
        unsafe_allow_html=True,
    )


def render_sidebar_metrics(architecture_version: str) -> None:
    st.sidebar.markdown("### Execution Metrics")
    latency = st.session_state.get("last_latency_seconds")
    if latency is None:
        st.sidebar.metric(
            "Last Query Latency",
            "—",
            help="Run a v1.0, v2.0, or v3.0 chat query to populate.",
        )
    else:
        st.sidebar.metric("Last Query Latency", f"{latency:.2f}s")

    st.sidebar.caption(f"Active architecture: {architecture_version}")


def render_evals_analytics_tab() -> None:
    st.markdown("### Evals & Analytics")
    st.caption(
        "Mock telemetry report illustrating architectural evolution across pipeline iterations."
    )

    performance_df = pd.DataFrame(
        {
            "Architecture": [
                "v1.0 Naive RAG",
                "v2.0 Advanced RAG",
                "v3.0 Agentic FSM",
            ],
            "Hallucination Rate (%)": [18.4, 9.2, 3.1],
            "Invalid Execution Paths (%)": [12.0, 6.5, 1.8],
            "Guardrail Coverage (%)": [0.0, 0.0, 100.0],
        }
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("v1 Hallucination Rate", "18.4%")
    with col2:
        st.metric("v2 Hallucination Rate", "9.2%", delta="-9.2% vs v1")
    with col3:
        st.metric("v3 Hallucination Rate", "3.1%", delta="-6.1% vs v2")

    st.markdown("#### Historical Pipeline Iterations")
    st.bar_chart(
        performance_df.set_index("Architecture")[
            ["Hallucination Rate (%)", "Invalid Execution Paths (%)"]
        ]
    )

    st.markdown("#### Agentic Safety Coverage")
    st.line_chart(
        performance_df.set_index("Architecture")[["Guardrail Coverage (%)"]]
    )

    st.info(
        "Transitioning to the v3.0 Agentic FSM introduces intent routing, specialist "
        "execution paths, and guardrail evaluation — reducing unsupported claims and "
        "invalid execution branches in mock evaluation runs."
    )


def main() -> None:
    st.set_page_config(
        page_title="GLP-1 RAG Dashboard",
        page_icon="💊",
        layout="wide",
    )

    init_session_state()
    supabase_url, supabase_key, gemini_api_key = load_env()
    supabase = get_supabase_client(supabase_url, supabase_key)

    architecture_version = st.sidebar.selectbox(
        "System Architecture Version",
        options=ARCHITECTURE_OPTIONS,
    )

    if not gemini_api_key:
        st.sidebar.warning("`GEMINI_API_KEY` is not set.")
    if not supabase_url or not supabase_key:
        st.sidebar.warning("`SUPABASE_URL` or `SUPABASE_KEY` is not set.")

    render_sidebar_metrics(architecture_version)

    render_engineering_design_brief()

    tab1, tab2, tab3 = st.tabs(
        ["🚀 System Evolution", "📊 Data Lake Telemetry", "🧪 Evals & Analytics"]
    )

    with tab1:
        st.subheader("System Evolution")

        if architecture_version in ARCHITECTURE_OPTIONS:
            if credentials_ready(supabase_url, supabase_key, gemini_api_key):
                render_rag_chat(architecture_version)
            else:
                st.info(
                    "Set `SUPABASE_URL`, `SUPABASE_KEY`, and `GEMINI_API_KEY` in `.env` "
                    "to enable chat across all architecture versions."
                )
        else:
            st.info("Select a supported architecture version from the sidebar.")

    with tab2:
        st.subheader("Data Lake Telemetry")
        total_rows, error_message = fetch_fda_chunk_count(supabase)
        v1_rows, v1_error = fetch_fda_chunk_count_by_version(supabase, "v1.0_naive")
        v2_rows, v2_error = fetch_fda_chunk_count_by_version(supabase, "v2.0_optimized")
        if error_message:
            st.metric(label="Total Items in `fda_document_chunks`", value="N/A")
            st.error(error_message)
        else:
            st.metric(label="Total Items in `fda_document_chunks`", value=total_rows)

        st.markdown("### Pipeline Version Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="v1.0_naive Rows",
                value="N/A" if v1_error else v1_rows,
            )
            if v1_error:
                st.caption(v1_error)
        with col2:
            st.metric(
                label="v2.0_optimized Rows",
                value="N/A" if v2_error else v2_rows,
            )
            if v2_error:
                st.caption(v2_error)

    with tab3:
        render_evals_analytics_tab()


if __name__ == "__main__":
    main()
