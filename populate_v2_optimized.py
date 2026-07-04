#!/usr/bin/env python3
"""
Populate Supabase with v2.0 optimized semantic chunks.

Pipeline:
  1) Split label text into sentences via terminal punctuation (. ! ?)
  2) Build rolling 3-sentence groups for localized context
  3) Embed with Gemini (3072-dim vectors for halfvec(3072) table)
  4) Insert into fda_document_chunks with pipeline_version v2.0_optimized
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv
from google import genai
from supabase import Client, create_client

# text-embedding-004 is not available on the current Gemini Developer API and
# outputs 768 dims when available elsewhere. gemini-embedding-001 produces the
# required 3072-dimensional vectors for the halfvec(3072) column.
EMBEDDING_MODEL = "gemini-embedding-001"
EXPECTED_EMBEDDING_DIMENSIONS = 3072
PIPELINE_METADATA = {"pipeline_version": "v2.0_optimized"}
SENTENCE_GROUP_SIZE = 3

SECTION_FIELDS = (
    "indications_and_usage",
    "dosage_and_administration",
    "drug_interactions",
    "warnings_and_cautions",
)

DEFAULT_INPUT = Path("data/raw/raw_glp1_data.json")
DEFAULT_EMBED_BATCH_SIZE = 5
DEFAULT_INSERT_BATCH_SIZE = 10
DEFAULT_EMBED_DELAY_SEC = 0.5

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class SemanticChunk:
    brand_name: str | None
    generic_name: str | None
    manufacturer_name: str | None
    section_name: str
    chunk_content: str
    chunk_index: int
    record_index: int


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("populate_v2_optimized")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def init_clients() -> tuple[Client, genai.Client]:
    supabase_url = require_env("SUPABASE_URL")
    supabase_key = require_env("SUPABASE_KEY")
    gemini_api_key = require_env("GEMINI_API_KEY")

    os.environ["GOOGLE_API_KEY"] = gemini_api_key

    supabase = create_client(supabase_url, supabase_key)
    gemini = genai.Client()
    return supabase, gemini


def first_value(values: list[str] | None) -> str | None:
    if not values:
        return None
    return str(values[0])


def join_field_text(values: list[str] | None) -> str:
    if not values:
        return ""
    return "\n\n".join(str(v) for v in values if v)


def split_sentences(text: str) -> list[str]:
    """Split text into clean sentences using terminal punctuation boundaries."""
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []

    raw_parts = SENTENCE_SPLIT_PATTERN.split(normalized)
    sentences: list[str] = []
    for part in raw_parts:
        sentence = part.strip()
        if not sentence:
            continue
        if sentence[-1] not in ".!?":
            sentence = f"{sentence}."
        sentences.append(sentence)
    return sentences


def rolling_sentence_groups(sentences: list[str], group_size: int = SENTENCE_GROUP_SIZE) -> list[str]:
    """Group sentences into rolling windows of N sentences."""
    if not sentences:
        return []
    if len(sentences) <= group_size:
        return [" ".join(sentences)]

    groups: list[str] = []
    for start in range(0, len(sentences) - group_size + 1):
        window = sentences[start : start + group_size]
        groups.append(" ".join(window))
    return groups


def semantic_chunk_text(text: str) -> list[str]:
    sentences = split_sentences(text)
    return rolling_sentence_groups(sentences, group_size=SENTENCE_GROUP_SIZE)


def iter_semantic_chunks(records: list[dict[str, Any]]) -> Iterator[SemanticChunk]:
    for record_index, record in enumerate(records):
        brand_name = first_value(record.get("brand_name"))
        generic_name = first_value(record.get("generic_name"))
        manufacturer_name = first_value(record.get("manufacturer_name"))

        for section_name in SECTION_FIELDS:
            section_text = join_field_text(record.get(section_name))
            for chunk_index, chunk_content in enumerate(semantic_chunk_text(section_text)):
                if not chunk_content.strip():
                    continue
                yield SemanticChunk(
                    brand_name=brand_name,
                    generic_name=generic_name,
                    manufacturer_name=manufacturer_name,
                    section_name=section_name,
                    chunk_content=chunk_content,
                    chunk_index=chunk_index,
                    record_index=record_index,
                )


def embed_chunk(gemini: genai.Client, chunk_string: str) -> list[float]:
    response = gemini.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=chunk_string,
    )
    return list(response.embeddings[0].values)


def build_row(chunk: SemanticChunk, embedding: list[float]) -> dict[str, Any]:
    return {
        "brand_name": chunk.brand_name,
        "generic_name": chunk.generic_name,
        "manufacturer_name": chunk.manufacturer_name,
        "section_name": chunk.section_name,
        "chunk_content": chunk.chunk_content,
        "embedding": embedding,
        "metadata": PIPELINE_METADATA,
    }


def insert_batch(supabase: Client, rows: list[dict[str, Any]]) -> None:
    if rows:
        supabase.table("fda_document_chunks").insert(rows).execute()


def populate(
    input_path: Path,
    embed_batch_size: int,
    insert_batch_size: int,
    embed_delay_sec: float,
    dry_run: bool,
    logger: logging.Logger,
) -> int:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open(encoding="utf-8") as f:
        dataset = json.load(f)

    records = dataset.get("records", [])
    all_chunks = list(iter_semantic_chunks(records))
    total_chunks = len(all_chunks)

    logger.info("Loaded %d drug records from %s", len(records), input_path)
    logger.info(
        "Prepared %d semantic chunks (rolling %d-sentence groups)",
        total_chunks,
        SENTENCE_GROUP_SIZE,
    )
    logger.info(
        "Embedding model: %s | target dimensions: %d | metadata: %s",
        EMBEDDING_MODEL,
        EXPECTED_EMBEDDING_DIMENSIONS,
        PIPELINE_METADATA,
    )

    if total_chunks == 0:
        logger.warning("No semantic chunks generated; nothing to insert.")
        return 0

    if dry_run:
        logger.info("Dry run enabled — skipping Gemini and Supabase calls.")
        for idx, chunk in enumerate(all_chunks[:5], start=1):
            logger.info(
                "[%d/%d] sample | brand=%s | section=%s | chars=%d | preview=%s",
                idx,
                total_chunks,
                chunk.brand_name,
                chunk.section_name,
                len(chunk.chunk_content),
                chunk.chunk_content[:120] + ("…" if len(chunk.chunk_content) > 120 else ""),
            )
        return 0

    supabase, gemini = init_clients()
    inserted = 0
    pending_rows: list[dict[str, Any]] = []

    for batch_start in range(0, total_chunks, embed_batch_size):
        batch = all_chunks[batch_start : batch_start + embed_batch_size]
        batch_num = (batch_start // embed_batch_size) + 1
        batch_total = (total_chunks + embed_batch_size - 1) // embed_batch_size

        logger.info(
            "Processing embed batch %d/%d (%d chunks)",
            batch_num,
            batch_total,
            len(batch),
        )

        for offset, chunk in enumerate(batch):
            global_index = batch_start + offset + 1
            embedding = embed_chunk(gemini, chunk.chunk_content)
            row = build_row(chunk, embedding)
            pending_rows.append(row)

            logger.info(
                "[%d/%d] embedded %d-dim | brand=%s | section=%s | chunk=%d",
                global_index,
                total_chunks,
                len(embedding),
                chunk.brand_name,
                chunk.section_name,
                chunk.chunk_index,
            )

            if len(pending_rows) >= insert_batch_size:
                insert_batch(supabase, pending_rows)
                inserted += len(pending_rows)
                logger.info(
                    "Inserted batch of %d rows into fda_document_chunks (total: %d/%d)",
                    len(pending_rows),
                    inserted,
                    total_chunks,
                )
                pending_rows.clear()

        if embed_delay_sec > 0 and batch_start + embed_batch_size < total_chunks:
            time.sleep(embed_delay_sec)

    if pending_rows:
        insert_batch(supabase, pending_rows)
        inserted += len(pending_rows)
        logger.info(
            "Inserted final batch of %d rows into fda_document_chunks (total: %d/%d)",
            len(pending_rows),
            inserted,
            total_chunks,
        )

    logger.info(
        "v2.0 optimized population complete: %d rows with metadata %s",
        inserted,
        PIPELINE_METADATA,
    )
    return inserted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate v2.0 optimized semantic chunks into Supabase."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to raw_glp1_data.json",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=DEFAULT_EMBED_BATCH_SIZE,
        help="Chunks to embed per processing batch",
    )
    parser.add_argument(
        "--insert-batch-size",
        type=int,
        default=DEFAULT_INSERT_BATCH_SIZE,
        help="Rows per Supabase insert call",
    )
    parser.add_argument(
        "--embed-delay-sec",
        type=float,
        default=DEFAULT_EMBED_DELAY_SEC,
        help="Pause between embed batches to reduce rate-limit risk",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build semantic chunks and log samples without API calls",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    logger = setup_logging()
    args = parse_args()

    try:
        populate(
            input_path=args.input,
            embed_batch_size=max(1, args.embed_batch_size),
            insert_batch_size=max(1, args.insert_batch_size),
            embed_delay_sec=max(0.0, args.embed_delay_sec),
            dry_run=args.dry_run,
            logger=logger,
        )
        return 0
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        return 1
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:
        logger.exception("v2.0 population failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
