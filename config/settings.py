"""Centralized environment configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: str
    openai_api_key: str
    openfda_api_key: str | None
    log_level: str
    data_dir: Path
    log_dir: Path

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            supabase_url=os.getenv("SUPABASE_URL", ""),
            supabase_anon_key=os.getenv("SUPABASE_ANON_KEY", ""),
            supabase_service_role_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openfda_api_key=os.getenv("OPENFDA_API_KEY") or None,
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            data_dir=Path(os.getenv("DATA_DIR", "./data/raw")),
            log_dir=Path(os.getenv("LOG_DIR", "./logs")),
        )
