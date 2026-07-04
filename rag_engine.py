#!/usr/bin/env python3
"""Phase 1 Naive RAG engine (retrieve + generate)."""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from sentence_transformers import CrossEncoder
from supabase import Client, create_client

# v1 naive RAG configuration (matches populate_vector_db.py embeddings).
EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-2.5-flash-lite"
TOP_K = 3
PIPELINE_VERSION = "v1.0_naive"
SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the question based on the provided context."
)

# Advanced RAG configuration.
ADVANCED_PIPELINE_VERSION = "v2.0_optimized"
ADVANCED_TOP_K_CANDIDATES = 15
ADVANCED_TOP_K_FINAL = 3
ADVANCED_GENERATION_MODEL = "gemini-2.5-flash-lite"
ADVANCED_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the question based on the provided context. "
    "Prioritize factual accuracy and safety for GLP-1-related medications."
)

_CROSS_ENCODER_MODEL_NAME = "mixedbread-ai/mxbai-rerank-xsmall-v1"
_cross_encoder: CrossEncoder | None = None


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

    supabase = create_client(supabase_url, supabase_key)
    gemini_client = genai.Client()
    return supabase, gemini_client


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(_CROSS_ENCODER_MODEL_NAME)
    return _cross_encoder


def _embed_query(gemini_client: genai.Client, user_query: str) -> list[float]:
    response = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=user_query,
    )
    return list(response.embeddings[0].values)


def _parse_embedding(raw: Any) -> list[float]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [float(v) for v in raw]
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("[") and text.endswith("]"):
            return [float(v) for v in json.loads(text)]
        return [float(v) for v in text.strip("[]").split(",") if v.strip()]
    return [float(v) for v in raw]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return -1.0
    return dot / (norm_a * norm_b)


def _retrieve_via_rpc(
    supabase: Client,
    query_embedding: list[float],
    top_k: int,
) -> list[dict[str, Any]]:
    response = supabase.rpc(
        "match_fda_document_chunks_v1_naive",
        {
            "match_count": top_k,
            "query_embedding": query_embedding,
        },
    ).execute()
    data = response.data or []
    return data if isinstance(data, list) else []


def _retrieve_via_rpc_v2(
    supabase: Client,
    query_embedding: list[float],
    match_count: int,
    filter_version: str,
) -> list[dict[str, Any]]:
    """
    Retrieve candidate chunks for advanced RAG via Supabase RPC.

    Expected SQL RPC in Supabase:
        match_document_chunks(
          match_count int,
          query_embedding halfvec(3072),
          filter_version text
        )
    """
    response = supabase.rpc(
        "match_document_chunks",
        {
            "match_count": match_count,
            "query_embedding": query_embedding,
            "filter_version": filter_version,
        },
    ).execute()
    data = response.data or []
    return data if isinstance(data, list) else []


def _retrieve_via_table_scan(
    supabase: Client,
    query_embedding: list[float],
    top_k: int,
) -> list[dict[str, Any]]:
    """Fallback when RPC is not deployed: cosine rank in Python."""
    response = (
        supabase.table("fda_document_chunks")
        .select(
            "id, brand_name, generic_name, manufacturer_name, "
            "section_name, chunk_content, metadata, embedding"
        )
        .filter("metadata->>pipeline_version", "eq", PIPELINE_VERSION)
        .execute()
    )
    rows = response.data or []
    if not isinstance(rows, list):
        return []

    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        embedding = _parse_embedding(row.get("embedding"))
        similarity = _cosine_similarity(query_embedding, embedding)
        scored.append((similarity, row))

    scored.sort(key=lambda item: item[0], reverse=True)
    top_rows: list[dict[str, Any]] = []
    for similarity, row in scored[:top_k]:
        result = dict(row)
        result["similarity"] = similarity
        result.pop("embedding", None)
        top_rows.append(result)
    return top_rows


def _retrieve_top_chunks(
    supabase: Client,
    query_embedding: list[float],
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    try:
        rows = _retrieve_via_rpc(supabase, query_embedding, top_k)
        if rows:
            return rows
    except Exception:
        pass

    return _retrieve_via_table_scan(supabase, query_embedding, top_k)


def _build_context(retrieved_rows: list[dict[str, Any]]) -> tuple[list[str], str, list[dict[str, Any]]]:
    chunks = [str(row.get("chunk_content", "")).strip() for row in retrieved_rows]
    chunks = [chunk for chunk in chunks if chunk]
    context = "\n\n---\n\n".join(chunks)
    sources = [
        {
            "chunk_content": str(row.get("chunk_content", "")).strip(),
            "section_name": row.get("section_name"),
            "brand_name": row.get("brand_name"),
            "generic_name": row.get("generic_name"),
            "manufacturer_name": row.get("manufacturer_name"),
            "similarity": row.get("similarity"),
        }
        for row in retrieved_rows
        if str(row.get("chunk_content", "")).strip()
    ]
    return chunks, context, sources


def _generate_answer(
    gemini_client: genai.Client,
    user_query: str,
    context: str,
) -> str:
    if not context.strip():
        return (
            "I could not find relevant FDA label context for that question. "
            "Make sure `fda_document_chunks` is populated and matches pipeline "
            f"`{PIPELINE_VERSION}`."
        )

    prompt = (
        f"Context:\n{context}\n\n"
        f"User question:\n{user_query}\n\n"
        "Answer:"
    )
    response = gemini_client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config={
            "system_instruction": SYSTEM_PROMPT,
        },
    )
    return (response.text or "").strip()


def naive_rag_query(user_query: str) -> dict[str, Any]:
    """Run Phase 1 naive RAG: embed query, retrieve top chunks, generate answer."""
    started = time.perf_counter()

    supabase, gemini_client = _init_clients()
    query_embedding = _embed_query(gemini_client, user_query)
    retrieved_rows = _retrieve_top_chunks(supabase, query_embedding, top_k=TOP_K)
    retrieved_chunks, context, retrieved_sources = _build_context(retrieved_rows)
    answer = _generate_answer(gemini_client, user_query, context)

    latency_seconds = time.perf_counter() - started
    return {
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "retrieved_sources": retrieved_sources,
        "metrics": {
            "latency_seconds": latency_seconds,
            "pipeline": PIPELINE_VERSION,
        },
    }


def advanced_rag_query(user_query: str) -> dict[str, Any]:
    """
    Phase 2 Advanced RAG:
      1) Embed query with text-embedding-004 (fall back to gemini-embedding-001).
      2) Retrieve 15 v2.0_optimized chunks via Supabase RPC.
      3) Re-rank with local cross-encoder and keep top 3.
      4) Generate answer with gemini-2.5-flash-lite.
    """
    started = time.perf_counter()

    supabase, gemini_client = _init_clients()
    try:
        embed_response = gemini_client.models.embed_content(
            model="text-embedding-004",
            contents=user_query,
        )
        query_embedding = list(embed_response.embeddings[0].values)
    except Exception:
        # Fallback to the same embedding model used for storage.
        query_embedding = _embed_query(gemini_client, user_query)

    try:
        initial_rows = _retrieve_via_rpc_v2(
            supabase=supabase,
            query_embedding=query_embedding,
            match_count=ADVANCED_TOP_K_CANDIDATES,
            filter_version=ADVANCED_PIPELINE_VERSION,
        )
    except Exception:
        initial_rows = []

    if not isinstance(initial_rows, list):
        initial_rows = []

    if not initial_rows:
        # Fallback: reuse naive path when no v2 candidates are available.
        result = naive_rag_query(user_query)
        elapsed = time.perf_counter() - started
        result["metrics"]["latency_seconds"] = elapsed
        result["metrics"]["pipeline"] = f"{ADVANCED_PIPELINE_VERSION}_fallback_v1"
        return result

    ranker = _get_cross_encoder()
    pairs = [[user_query, str(row.get("chunk_content", ""))] for row in initial_rows]
    scores = ranker.predict(pairs)

    ranked = [
        {**row, "cross_score": float(score)}
        for row, score in zip(initial_rows, scores)
    ]
    ranked.sort(key=lambda r: r.get("cross_score", 0.0), reverse=True)
    top_ranked = ranked[:ADVANCED_TOP_K_FINAL]

    context_chunks = [str(r.get("chunk_content", "")).strip() for r in top_ranked]
    context_chunks = [c for c in context_chunks if c]
    context = "\n\n---\n\n".join(context_chunks)

    enriched_sources: list[dict[str, Any]] = []
    for r in top_ranked:
        enriched_sources.append(
            {
                "chunk_content": str(r.get("chunk_content", "")).strip(),
                "section_name": r.get("section_name"),
                "brand_name": r.get("brand_name"),
                "generic_name": r.get("generic_name"),
                "manufacturer_name": r.get("manufacturer_name"),
                "similarity": r.get("similarity"),
                "cross_score": r.get("cross_score"),
                "pipeline_version": ADVANCED_PIPELINE_VERSION,
            }
        )

    if context.strip():
        prompt = (
            f"Context:\n{context}\n\n"
            f"User question:\n{user_query}\n\n"
            "Answer:"
        )
        response = gemini_client.models.generate_content(
            model=ADVANCED_GENERATION_MODEL,
            contents=prompt,
            config={"system_instruction": ADVANCED_SYSTEM_PROMPT},
        )
        answer = (response.text or "").strip()
    else:
        answer = (
            "I could not find relevant optimized-context chunks for that question. "
            "Try broadening the query or ensure the v2.0_optimized pipeline is populated."
        )

    latency_seconds = time.perf_counter() - started
    return {
        "answer": answer,
        "retrieved_sources": enriched_sources,
        "metrics": {
            "latency_seconds": latency_seconds,
            "pipeline": f"{ADVANCED_PIPELINE_VERSION}_advanced",
        },
    }


if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "What are major GLP-1 risks?"
    result = naive_rag_query(query)
    print(result)
