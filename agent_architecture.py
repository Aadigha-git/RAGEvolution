#!/usr/bin/env python3
"""Intent-driven FSM agent network for GLP-1 FDA Q&A using LangGraph."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Literal

from dotenv import load_dotenv
from google import genai
from langgraph.graph import END, StateGraph
from supabase import Client, create_client
from typing_extensions import TypedDict

from rag_engine import advanced_rag_query

INTENT_MODEL = "gemini-2.5-flash"
GUARDRAIL_MODEL = "gemini-2.5-flash"
LOOKUP_ANSWER_MODEL = "gemini-2.5-flash-lite"
LOOKUP_ROW_LIMIT = 10


class AgentState(TypedDict):
    user_query: str
    intent: str
    context: list
    response: str
    history: list
    safety_check_passed: bool


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _init_clients() -> tuple[Client, genai.Client]:
    load_dotenv()
    supabase_url = _require_env("SUPABASE_URL")
    supabase_key = _require_env("SUPABASE_KEY")
    gemini_api_key = _require_env("GEMINI_API_KEY")
    os.environ["GOOGLE_API_KEY"] = gemini_api_key
    return create_client(supabase_url, supabase_key), genai.Client()


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Could not parse JSON object from model output: {text[:200]}")
    return json.loads(cleaned[start : end + 1])


def _append_history(state: AgentState, event: str) -> list:
    history = list(state.get("history", []))
    history.append(event)
    return history


def _extract_lookup_terms(user_query: str) -> list[str]:
    """Pull likely brand/manufacturer tokens from the user query."""
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]{2,}", user_query)
    stopwords = {
        "who", "what", "where", "when", "which", "does", "make", "makes",
        "manufacturer", "brand", "drug", "forms", "come", "the", "for", "and",
    }
    terms = [t for t in tokens if t.lower() not in stopwords]
    if not terms:
        terms = [user_query.strip()[:80]]
    return terms[:3]


def classify_intent_node(state: AgentState) -> dict[str, Any]:
    _, gemini = _init_clients()
    prompt = (
        "Classify the user query into exactly one intent.\n"
        "Return strict JSON only, with one of:\n"
        '{"intent": "SIMPLE_LOOKUP"}\n'
        '{"intent": "CLINICAL_ANALYSIS"}\n\n'
        "SIMPLE_LOOKUP examples:\n"
        '- "Who makes Ozempic?"\n'
        '- "What forms does Wegovy come in?"\n'
        '- "Which manufacturer produces Trulicity?"\n\n'
        "CLINICAL_ANALYSIS examples:\n"
        '- "What are pancreatitis warnings for semaglutide?"\n'
        '- "Compare hypoglycemia risk between liraglutide and insulin."\n'
        '- "What drug interactions should I know for GLP-1 drugs?"\n\n'
        f"User query: {state['user_query']}"
    )
    response = gemini.models.generate_content(model=INTENT_MODEL, contents=prompt)
    parsed = _parse_json_object(response.text or "")
    intent = parsed.get("intent", "CLINICAL_ANALYSIS")
    if intent not in {"SIMPLE_LOOKUP", "CLINICAL_ANALYSIS"}:
        intent = "CLINICAL_ANALYSIS"

    return {
        "intent": intent,
        "history": _append_history(state, f"classify_intent_node -> {intent}"),
    }


def lookup_specialist_node(state: AgentState) -> dict[str, Any]:
    supabase, gemini = _init_clients()
    terms = _extract_lookup_terms(state["user_query"])

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for term in terms:
        pattern = f"%{term}%"
        response = (
            supabase.table("fda_document_chunks")
            .select(
                "id, brand_name, generic_name, manufacturer_name, "
                "section_name, chunk_content, metadata"
            )
            .or_(f"brand_name.ilike.{pattern},manufacturer_name.ilike.{pattern}")
            .limit(LOOKUP_ROW_LIMIT)
            .execute()
        )
        for row in response.data or []:
            row_id = str(row.get("id", ""))
            if row_id and row_id not in seen_ids:
                seen_ids.add(row_id)
                rows.append(row)

    context_blocks = [
        {
            "brand_name": row.get("brand_name"),
            "generic_name": row.get("generic_name"),
            "manufacturer_name": row.get("manufacturer_name"),
            "section_name": row.get("section_name"),
            "chunk_content": row.get("chunk_content"),
        }
        for row in rows
    ]

    if not context_blocks:
        answer = (
            "I could not find a direct brand or manufacturer match in the database "
            "for that lookup question."
        )
    else:
        context_text = json.dumps(context_blocks, indent=2)
        prompt = (
            "Answer the user question using only the database lookup rows below.\n"
            "If the answer is not present, say you could not find it.\n\n"
            f"Database rows:\n{context_text}\n\n"
            f"User question:\n{state['user_query']}\n\n"
            "Answer:"
        )
        response = gemini.models.generate_content(model=LOOKUP_ANSWER_MODEL, contents=prompt)
        answer = (response.text or "").strip()

    return {
        "context": context_blocks,
        "response": answer,
        "history": _append_history(state, "lookup_specialist_node executed"),
    }


def clinical_analyst_node(state: AgentState) -> dict[str, Any]:
    result = advanced_rag_query(state["user_query"])
    context_blocks = result.get("retrieved_sources", [])
    return {
        "context": context_blocks,
        "response": result.get("answer", ""),
        "history": _append_history(
            state,
            f"clinical_analyst_node executed ({result.get('metrics', {}).get('pipeline', 'advanced')})",
        ),
    }


def guardrail_node(state: AgentState) -> dict[str, Any]:
    _, gemini = _init_clients()
    context_text = json.dumps(state.get("context", []), indent=2)
    prompt = (
        "You are a safety evaluator. Compare the generated response against the "
        "provided database context blocks.\n"
        "Return strict JSON only:\n"
        '{"hallucination_detected": true}\n'
        'or {"hallucination_detected": false}\n\n'
        "Mark hallucination_detected=true if the response introduces facts not "
        "supported by the context blocks.\n\n"
        f"Context blocks:\n{context_text}\n\n"
        f"Generated response:\n{state.get('response', '')}\n"
    )
    response = gemini.models.generate_content(model=GUARDRAIL_MODEL, contents=prompt)
    parsed = _parse_json_object(response.text or "")
    hallucination_detected = bool(parsed.get("hallucination_detected", False))
    safety_check_passed = not hallucination_detected

    final_response = state.get("response", "")
    if hallucination_detected:
        final_response = (
            f"{final_response}\n\n"
            "**Safety notice:** This answer may contain unsupported claims. "
            "Please verify against official FDA labeling."
        )

    return {
        "response": final_response,
        "safety_check_passed": safety_check_passed,
        "history": _append_history(
            state,
            f"guardrail_node -> safety_check_passed={safety_check_passed}",
        ),
    }


def route_by_intent(state: AgentState) -> Literal["lookup_specialist_node", "clinical_analyst_node"]:
    if state.get("intent") == "SIMPLE_LOOKUP":
        return "lookup_specialist_node"
    return "clinical_analyst_node"


def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("classify_intent_node", classify_intent_node)
    graph.add_node("lookup_specialist_node", lookup_specialist_node)
    graph.add_node("clinical_analyst_node", clinical_analyst_node)
    graph.add_node("guardrail_node", guardrail_node)

    graph.set_entry_point("classify_intent_node")
    graph.add_conditional_edges(
        "classify_intent_node",
        route_by_intent,
        {
            "lookup_specialist_node": "lookup_specialist_node",
            "clinical_analyst_node": "clinical_analyst_node",
        },
    )
    graph.add_edge("lookup_specialist_node", "guardrail_node")
    graph.add_edge("clinical_analyst_node", "guardrail_node")
    graph.add_edge("guardrail_node", END)
    return graph.compile()


agent_graph = build_agent_graph()


def run_agent_query(user_query: str) -> AgentState:
    """Execute the full FSM agent pipeline for a user query."""
    initial_state: AgentState = {
        "user_query": user_query,
        "intent": "",
        "context": [],
        "response": "",
        "history": [],
        "safety_check_passed": False,
    }
    return agent_graph.invoke(initial_state)


if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "Who makes Ozempic?"
    final_state = run_agent_query(query)
    print(json.dumps(final_state, indent=2, default=str))
