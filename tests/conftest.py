"""Shared pytest fixtures: a tiny, deterministic unified-schema review table.

The fixture deliberately embeds patterns the cleaning/feature code must detect:
* exact-duplicate "generic praise" text repeated on product P1 (r1, r2, r7),
* a review burst on P1 (four reviews within hours),
* a reviewer with cross-category history (U1 reviews P1/Electronics and P3/Books),
* a None verified_purchase (simulating the 2014 edition's missing proxy),
* an HTML entity + messy whitespace (r6) to exercise text normalization,
* clearly positive (r4) and negative (r5) reviews for sentiment sign checks.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")  # headless backend for plotting tests

from src.clean import clean_reviews
from src.fallback_loader import REVIEW_COLUMNS

BASE = datetime(2023, 5, 1, 12, 0, tzinfo=timezone.utc)


def _r(rid, pid, cat, uid, rating, title, text, verified, dt, helpful, total, ptitle, price, brand):
    return {
        "review_id": rid, "product_id": pid, "product_category": cat, "reviewer_id": uid,
        "rating": rating, "review_title": title, "review_text": text,
        "verified_purchase": verified, "review_date": dt, "helpful_votes": helpful,
        "total_votes": total, "product_title": ptitle, "price": price, "brand": brand,
        "main_category": cat, "source": "scrape",
    }


@pytest.fixture
def raw_reviews_df() -> pd.DataFrame:
    rows = [
        _r("r1", "P1", "Electronics", "U1", 5, "Five Stars", "Great product works as expected",
           False, BASE, 0, None, "Item1", 20.0, "Acme"),
        _r("r2", "P1", "Electronics", "U2", 5, "Five Stars", "Great product works as expected",
           False, BASE + timedelta(hours=1), 1, None, "Item1", 20.0, "Acme"),
        _r("r3", "P1", "Electronics", "U3", 4, "Good", "Battery lasts all day, detailed notes for travel use.",
           True, BASE + timedelta(hours=2), 5, 8, "Item1", 20.0, "Acme"),
        _r("r7", "P1", "Electronics", "U6", 5, "Five Stars", "Great product works as expected",
           False, BASE + timedelta(hours=3), 0, None, "Item1", 20.0, "Acme"),
        _r("r4", "P2", "Books", "U4", 5, "Love", "Exactly what I needed, wonderful and helpful read.",
           True, BASE + timedelta(days=1), 3, 4, "Item2", 12.5, "Globex"),
        _r("r5", "P2", "Books", "U5", 1, "Bad", "Stopped working, terrible, waste of money.",
           False, BASE + timedelta(days=2), 0, 2, "Item2", 12.5, "Globex"),
        _r("r6", "P3", "Books", "U1", 3, "Ok", "Average book &amp; nothing  special\n",
           None, BASE + timedelta(days=3), 0, None, "Item3", 9.99, "Initech"),
        _r("r8", "P2", "Books", "U4", 4, "Nice", "Wonderful and helpful, exactly what I needed.",
           True, BASE + timedelta(days=1, hours=1), 2, 5, "Item2", 12.5, "Globex"),
    ]
    return pd.DataFrame(rows)[list(REVIEW_COLUMNS)]


@pytest.fixture
def clean_reviews_df(raw_reviews_df: pd.DataFrame) -> pd.DataFrame:
    return clean_reviews(raw_reviews_df)


@pytest.fixture
def medium_reviews_df(tmp_path) -> pd.DataFrame:
    """~160-row synthetic dataset (2 categories) for modeling/network smoke tests.

    Product 0 of each category is a "suspicious" all-unverified burst, so both
    proxy classes are well represented for stratified CV.
    """
    from src import fallback_loader, utils

    cfg = utils.load_config()
    fallback_loader.make_synthetic_sample(tmp_path, categories=("Electronics", "Books"),
                                          n_products=8, reviews_per_product=10, seed=1)
    rows: list[dict] = []
    for cat in ("Electronics", "Books"):
        rows += fallback_loader.load_category(cfg, cat, raw_dir=tmp_path)
    return clean_reviews(pd.DataFrame(rows))
