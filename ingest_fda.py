#!/usr/bin/env python3
"""
Fetch GLP-1 receptor agonist and related metabolic drug labels from openFDA.

Pulls up to 50 records matching pharmacologic classes and active ingredients
associated with GLP-1 therapies, extracts key label text fields, and writes
`raw_glp1_data.json` for local inspection before Supabase ingestion.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from config.settings import Settings

OPENFDA_BASE_URL = "https://api.fda.gov/drug/label.json"
DEFAULT_LIMIT = 50
OUTPUT_FILENAME = "raw_glp1_data.json"

# openFDA stores the EPC as "GLP-1 Receptor Agonist [EPC]" (not the longer
# "Glucagon-Like Peptide-1 …" wording). The MoA and ingredient clauses below
# catch related metabolic agents (e.g. tirzepatide dual GIP/GLP-1 agonist).
GLP1_SEARCH = (
    "("
    'openfda.pharm_class_epc:"GLP-1 Receptor Agonist [EPC]" OR '
    'openfda.pharm_class_epc:"Glucose-dependent Insulinotropic Polypeptide Receptor Agonist [EPC]" OR '
    'openfda.pharm_class_moa:"Glucagon-like Peptide-1 (GLP-1) Agonists [MoA]" OR '
    "openfda.generic_name:(SEMAGLUTIDE OR \"ORAL SEMAGLUTIDE\" OR LIRAGLUTIDE OR "
    "TIRZEPATIDE OR DULAGLUTIDE OR EXENATIDE OR LIXISENATIDE OR ALBIGLUTIDE)"
    ")"
)

EXTRACTED_FIELDS = (
    "brand_name",
    "generic_name",
    "manufacturer_name",
    "indications_and_usage",
    "dosage_and_administration",
    "drug_interactions",
    "warnings_and_cautions",
)

OPENFDA_FIELDS = {"brand_name", "generic_name", "manufacturer_name"}


def setup_logging(log_dir: Path, level: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ingest_fda")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_dir / "ingest_fda.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def fetch_glp1_labels(
    limit: int,
    api_key: str | None,
    logger: logging.Logger,
) -> dict[str, Any]:
    params: dict[str, str | int] = {
        "search": GLP1_SEARCH,
        "limit": limit,
    }
    if api_key:
        params["api_key"] = api_key
        logger.info("Using OPENFDA_API_KEY for authenticated requests.")

    logger.info(
        "Requesting up to %d GLP-1 / metabolic drug labels from openFDA …",
        limit,
    )
    logger.info("Search query: %s", GLP1_SEARCH)

    response = requests.get(OPENFDA_BASE_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()

    meta = payload.get("meta", {})
    results = payload.get("results", [])
    total = (meta.get("results") or {}).get("total", len(results))
    logger.info("Received %d records (total matching: %s)", len(results), total)
    return payload


def _field_value(record: dict[str, Any], field: str) -> list[str] | None:
    if field in OPENFDA_FIELDS:
        values = (record.get("openfda") or {}).get(field)
    else:
        values = record.get(field)

    if values is None:
        return None
    if isinstance(values, list):
        return [str(v) for v in values if v is not None]
    return [str(values)]


def extract_record(record: dict[str, Any]) -> dict[str, Any]:
    return {field: _field_value(record, field) for field in EXTRACTED_FIELDS}


def build_cleaned_dataset(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results", [])
    records = [extract_record(r) for r in results]
    return {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": OPENFDA_BASE_URL,
        "search_query": GLP1_SEARCH,
        "record_count": len(records),
        "fields_extracted": list(EXTRACTED_FIELDS),
        "records": records,
    }


def write_glp1_output(
    cleaned: dict[str, Any],
    data_dir: Path,
    logger: logging.Logger,
) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / OUTPUT_FILENAME

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    logger.info("Wrote cleaned GLP-1 dataset → %s", output_path)
    return output_path


def log_extraction_summary(cleaned: dict[str, Any], logger: logging.Logger) -> None:
    records = cleaned.get("records", [])
    if not records:
        logger.warning("No GLP-1 records returned; verify search query or API status.")
        return

    logger.info("--- GLP-1 extraction summary ---")
    logger.info("Records written: %d", len(records))

    for field in EXTRACTED_FIELDS:
        present = sum(1 for r in records if r.get(field))
        logger.info("  %s: present in %d / %d records", field, present, len(records))

    sample = records[0]
    preview = {
        "brand_name": (sample.get("brand_name") or [None])[0],
        "generic_name": (sample.get("generic_name") or [None])[0],
        "manufacturer_name": (sample.get("manufacturer_name") or [None])[0],
        "indications_preview": (
            (sample.get("indications_and_usage") or [""])[0][:200] + "…"
            if sample.get("indications_and_usage")
            else None
        ),
    }
    logger.info("First record preview: %s", json.dumps(preview, indent=2))


def main() -> int:
    load_dotenv()
    settings = Settings.from_env()

    logger = setup_logging(settings.log_dir, settings.log_level)
    logger.info("Starting GLP-1 openFDA ingest pipeline")

    try:
        payload = fetch_glp1_labels(
            limit=DEFAULT_LIMIT,
            api_key=settings.openfda_api_key,
            logger=logger,
        )
        cleaned = build_cleaned_dataset(payload)
        output_path = write_glp1_output(cleaned, settings.data_dir, logger)
        log_extraction_summary(cleaned, logger)
        logger.info("Ingest complete. Inspect %s", output_path)
        return 0
    except requests.HTTPError as exc:
        logger.error("HTTP error from openFDA: %s", exc.response.text[:500])
        return 1
    except requests.RequestException as exc:
        logger.error("Network error: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
