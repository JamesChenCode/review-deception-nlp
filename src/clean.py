"""Cleaning & normalization — PURE functions over the unified review schema.

Design rule: functions here are **pure** (no I/O, no globals, deterministic, and
they never mutate their input — every function returns a new DataFrame). That
makes them trivially unit-testable on small fixtures (see ``tests/``).

Assumptions
-----------
* Input is already in the unified review schema (see :mod:`src.fallback_loader`);
  this module neither fetches nor joins data.
* ``verified_purchase`` is a WEAK PROXY label (purchase verification, not
  ground-truth deception) and is treated as data, never as truth. It may be
  ``<NA>`` for editions that lack the flag (e.g., the 2014 McAuley edition).
* Lowercasing is intentionally NOT done here — it is deferred to the TF-IDF /
  bag-of-words vectorizer in :mod:`src.features` so cleaned text stays readable
  for EDA. ``clean_text`` still exposes a ``lowercase`` switch for callers.
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Sequence

import pandas as pd

from . import utils
from .fallback_loader import REVIEW_COLUMNS

LOGGER = utils.get_logger("review_deception.clean")

_WHITESPACE = re.compile(r"\s+")

_STRING_COLUMNS = (
    "review_id", "product_id", "product_category", "reviewer_id",
    "review_title", "review_text", "product_title", "brand", "main_category", "source",
)


# ---------------------------------------------------------------------------
# Scalar text helpers (pure)
# ---------------------------------------------------------------------------
def clean_text(text: object, lowercase: bool = False) -> str:
    """Normalize one string: HTML-unescape, NFKC-normalize, collapse whitespace.

    >>> clean_text("Great   product &amp; value\\n")
    'Great product & value'
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = unicodedata.normalize("NFKC", html.unescape(str(text)))
    s = _WHITESPACE.sub(" ", s).strip()
    return s.lower() if lowercase else s


def _to_boolean(series: pd.Series) -> pd.Series:
    """Coerce assorted truthy encodings to pandas nullable ``boolean`` (keeps NA)."""
    def conv(v: object) -> object:
        if v is None or v is pd.NA or (isinstance(v, float) and pd.isna(v)):
            return pd.NA
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in {"true", "1", "yes", "y"}

    return series.map(conv).astype("boolean")


# ---------------------------------------------------------------------------
# DataFrame transforms (pure: copy in, copy out)
# ---------------------------------------------------------------------------
def validate_schema(df: pd.DataFrame, required: Sequence[str] = REVIEW_COLUMNS) -> pd.DataFrame:
    """Return ``df`` unchanged if it carries every required column, else raise."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Unified-schema validation failed; missing columns: {missing}")
    return df


def coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns to their canonical dtypes (numeric, datetime, boolean, str)."""
    df = df.copy()
    if "rating" in df:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").astype("float64")
    if "price" in df:
        df["price"] = pd.to_numeric(df["price"], errors="coerce").astype("float64")
    if "helpful_votes" in df:
        df["helpful_votes"] = pd.to_numeric(df["helpful_votes"], errors="coerce").fillna(0).astype("int64")
    if "total_votes" in df:
        df["total_votes"] = pd.to_numeric(df["total_votes"], errors="coerce").astype("Int64")
    if "verified_purchase" in df:
        df["verified_purchase"] = _to_boolean(df["verified_purchase"])
    if "review_date" in df:
        df["review_date"] = pd.to_datetime(df["review_date"], utc=True, errors="coerce")
    for col in _STRING_COLUMNS:
        if col in df:
            df[col] = df[col].map(lambda v: "" if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v))
    return df


def normalize_text_columns(
    df: pd.DataFrame,
    columns: Sequence[str] = ("review_text", "review_title"),
    lowercase: bool = False,
) -> pd.DataFrame:
    """Apply :func:`clean_text` to the given text columns."""
    df = df.copy()
    for col in columns:
        if col in df:
            df[col] = df[col].map(lambda s: clean_text(s, lowercase=lowercase))
    return df


def flag_exact_duplicates(
    df: pd.DataFrame, subset: Sequence[str] = ("product_id", "review_text")
) -> pd.DataFrame:
    """Add ``is_exact_duplicate`` (True for EVERY member of a duplicate group).

    Default subset catches copy-pasted text on the same product regardless of
    reviewer — a classic review-farm signal. (Near-duplicate / fuzzy matching
    lives in :mod:`src.features`.)
    """
    df = df.copy()
    cols = [c for c in subset if c in df]
    df["is_exact_duplicate"] = df.duplicated(subset=cols, keep=False) if cols else False
    return df


def drop_exact_duplicates(
    df: pd.DataFrame, subset: Sequence[str] = ("product_id", "reviewer_id", "review_text")
) -> pd.DataFrame:
    """Drop exact duplicate rows (keep first); returns a re-indexed copy."""
    cols = [c for c in subset if c in df]
    return df.drop_duplicates(subset=cols, keep="first").reset_index(drop=True)


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the documented missing-value policy (votes→0, text→"")."""
    df = df.copy()
    if "helpful_votes" in df:
        df["helpful_votes"] = df["helpful_votes"].fillna(0).astype("int64")
    for col in ("review_text", "review_title"):
        if col in df:
            df[col] = df[col].fillna("")
    return df


def clean_reviews(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Full cleaning pipeline → analysis-ready table (input unchanged).

    validate → coerce dtypes → normalize text → handle missing → flag duplicates.
    Note the verified-purchase label may remain ``<NA>``; rows are NOT dropped on
    a missing label here (that is a modeling decision left to :mod:`src.models`).
    """
    _ = config or {}
    out = validate_schema(df)
    out = coerce_dtypes(out)
    out = normalize_text_columns(out, lowercase=False)
    out = handle_missing(out)
    out = flag_exact_duplicates(out)
    LOGGER.info("Cleaned %d reviews (%d flagged exact-duplicate).",
                len(out), int(out["is_exact_duplicate"].sum()))
    return out.reset_index(drop=True)
