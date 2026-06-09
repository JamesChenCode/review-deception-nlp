"""Shared utilities: config loading, logging, seeding, IDs, small table helpers.

Introduced in Step 2 (flagged in the Step 1 summary). Kept deliberately tiny
and dependency-light so the whole collection pipeline can run on a bare Python
install (no third-party packages) for offline dry runs and CI smoke tests.

Optional third-party libraries (``pyyaml``, ``numpy``, ``pandas``, ``tqdm``) are
imported defensively: if absent, functionality degrades gracefully instead of
crashing. This mirrors the project's "graceful degradation" theme.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# --- optional deps (degrade gracefully) ------------------------------------
try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - exercised only on bare installs
    yaml = None

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    tqdm = None


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
# Mirrors the subset of ``config.yaml`` that the collection pipeline needs.
# ``load_config`` uses this as a BASE and overlays the YAML file when PyYAML is
# available, so behaviour is identical with or without a YAML parser installed
# (the bare-install dry run simply runs on these defaults).
DEFAULT_CONFIG: dict[str, Any] = {
    "project": {"name": "review-deception-nlp", "random_seed": 42},
    "data_source": {"source": "fallback", "auto_fallback": True, "max_failures": 5},
    "paths": {
        "data_raw": "data/raw",
        "data_interim": "data/interim",
        "data_processed": "data/processed",
        "data_groundtruth": "data/groundtruth",
        "cache": "data/raw/cache",
        "checkpoints": "data/raw/checkpoints",
        "logs": "logs",
    },
    "scraper": {
        "base_url": "https://www.amazon.com",
        "target_categories": ["electronics", "books", "home_kitchen"],
        "max_products_per_category": 50,
        "max_reviews_per_product": 200,
        "rate_limit": {"min_delay_seconds": 3.0, "max_delay_seconds": 7.0, "jitter": True},
        "retry": {
            "max_attempts": 4,
            "backoff_factor": 2.0,
            "backoff_max_seconds": 60,
            "retry_on_status": [429, 500, 502, 503, 504],
        },
        "caching": {"enabled": True, "ttl_days": 30},
        "checkpointing": {"enabled": True, "flush_every": 10},
        "user_agents": [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/123.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
        ],
        "respect_robots_txt": True,
    },
    "fallback": {
        "dataset": "amazon_reviews_2023",
        "categories": ["Electronics", "Books", "Home_and_Kitchen"],
        "reviews_filename_template": "{category}.jsonl",
        "meta_filename_template": "meta_{category}.jsonl",
        "sample_rows": None,
    },
    "groundtruth": {
        "source_repo": "https://github.com/bretthollenbeck/fake-reviews-data",
        "local_dir": "data/groundtruth",
        "filename": "public_reviews_dataset_cleaned.csv",
        "label_strategy": "primary",
        "sample_rows": 40000,
        "enabled": True,
    },
    "steam": {
        "appids": [730, 570, 1086940, 1245620, 413150],
        "max_reviews_per_app": 700,
        "language": "english",
        "filter": "recent",
        "purchase_type": "all",
        "min_delay_seconds": 1.0,
        "max_delay_seconds": 2.5,
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    },
}


def project_root() -> Path:
    """Return the repository root (the parent of this ``src`` package)."""
    return Path(__file__).resolve().parent.parent


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto a copy of ``base`` (override wins)."""
    out = copy.deepcopy(dict(base))
    for key, val in override.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(val, Mapping):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load configuration: ``DEFAULT_CONFIG`` overlaid with ``config.yaml``.

    If PyYAML is unavailable (bare install) or the file is missing, the defaults
    are returned unchanged so offline dry runs still work. Personal overrides in
    ``config.local.yaml`` (gitignored) are applied last when present.
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if yaml is None:
        logging.getLogger(__name__).warning(
            "PyYAML not installed; using built-in DEFAULT_CONFIG (offline mode)."
        )
        return cfg
    root = project_root()
    for candidate in (Path(path), root / Path(path)):
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as fh:
                file_cfg = yaml.safe_load(fh) or {}
            cfg = deep_merge(cfg, file_cfg)
            break
    local = root / "config.local.yaml"
    if local.is_file():
        with local.open("r", encoding="utf-8") as fh:
            cfg = deep_merge(cfg, yaml.safe_load(fh) or {})
    return cfg


_CONFIGURED_LOGGERS: set[str] = set()


def get_logger(name: str = "review_deception", config: Mapping[str, Any] | None = None) -> logging.Logger:
    """Return a configured, idempotent stdout logger (never reconfigures twice)."""
    logger = logging.getLogger(name)
    if name in _CONFIGURED_LOGGERS:
        return logger
    log_cfg = (config or {}).get("logging", {}) if config else DEFAULT_CONFIG["logging"]
    level = getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(log_cfg.get("format", DEFAULT_CONFIG["logging"]["format"])))
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False
    _CONFIGURED_LOGGERS.add(name)
    return logger


def set_seeds(seed: int = 42) -> int:
    """Seed all randomness we control (stdlib ``random`` and numpy if present)."""
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    return seed


def ensure_dirs(config: Mapping[str, Any]) -> None:
    """Create every directory referenced under ``config['paths']`` if missing."""
    root = project_root()
    for value in config.get("paths", {}).values():
        (root / value).mkdir(parents=True, exist_ok=True)


def stable_id(*parts: Any, length: int = 16) -> str:
    """Deterministic short id from arbitrary parts (used when a source lacks one)."""
    joined = "||".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:length]


def progress(iterable: Iterable[Any], **kwargs: Any) -> Iterable[Any]:
    """``tqdm`` wrapper that becomes a no-op pass-through when tqdm is absent."""
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def records_to_frame(records: Sequence[Mapping[str, Any]]):
    """Return a pandas ``DataFrame`` if pandas is installed, else the list as-is.

    Lets the same record-producing code feed both the full analysis pipeline
    (pandas) and the bare-install dry run (plain list of dicts).
    """
    if pd is not None:
        return pd.DataFrame(list(records))
    return list(records)


def preview_table(
    records: Sequence[Mapping[str, Any]],
    columns: Sequence[str] | None = None,
    n: int = 5,
    max_width: int = 38,
) -> str:
    """Render a compact aligned preview of records WITHOUT requiring pandas."""
    rows = list(records)[:n]
    if not rows:
        return "(no records)"
    cols = list(columns) if columns else list(rows[0].keys())

    def cell(value: Any) -> str:
        text = "" if value is None else str(value).replace("\n", " ")
        return text[: max_width - 1] + "…" if len(text) > max_width else text

    widths = {c: min(max_width, max(len(c), *(len(cell(r.get(c))) for r in rows))) for c in cols}
    sep = " | "
    header = sep.join(c.ljust(widths[c]) for c in cols)
    rule = "-+-".join("-" * widths[c] for c in cols)
    body = "\n".join(sep.join(cell(r.get(c)).ljust(widths[c]) for c in cols) for r in rows)
    return f"{header}\n{rule}\n{body}"
