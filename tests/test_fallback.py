"""Tests for src.fallback_loader against the REAL 2023 format (gz + field names)."""

from __future__ import annotations

import gzip
import json

from src import fallback_loader as fl, utils


def test_normalize_helpful_field_alias():
    base = {"parent_asin": "A", "user_id": "U", "rating": 5, "text": "t",
            "verified_purchase": True, "timestamp": 1672531200000}
    assert fl.normalize_review_2023({**base, "helpful_votes": 7}, "Cat")["helpful_votes"] == 7
    assert fl.normalize_review_2023({**base, "helpful_vote": 3}, "Cat")["helpful_votes"] == 3


def test_parse_epoch_auto_detects_ms_vs_s():
    assert fl.parse_epoch(1672531200000, "auto").year == 2023   # milliseconds
    assert fl.parse_epoch(1672531200, "auto").year == 2023      # seconds


def test_iter_jsonl_reads_gzip(tmp_path):
    path = tmp_path / "x.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"a": 1}) + "\n")
    assert list(fl.iter_jsonl(path)) == [{"a": 1}]


def test_load_category_finds_gzipped_files(tmp_path):
    """Real 2023 files are gzipped; load_category must resolve {category}.jsonl.gz."""
    reviews = [{"parent_asin": "A1", "user_id": "U1", "rating": 5, "title": "t",
                "text": "good", "verified_purchase": True,
                "timestamp": 1672531200000, "helpful_vote": 2}]
    meta = [{"parent_asin": "A1", "title": "Prod", "price": 9.99,
             "store": "Acme", "main_category": "Cat"}]
    with gzip.open(tmp_path / "Cat.jsonl.gz", "wt", encoding="utf-8") as fh:
        for o in reviews:
            fh.write(json.dumps(o) + "\n")
    with gzip.open(tmp_path / "meta_Cat.jsonl.gz", "wt", encoding="utf-8") as fh:
        for o in meta:
            fh.write(json.dumps(o) + "\n")

    rows = fl.load_category(utils.load_config(), "Cat", raw_dir=tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["product_id"] == "A1"
    assert row["helpful_votes"] == 2
    assert row["price"] == 9.99
    assert row["verified_purchase"] is True
    assert row["review_date"].year == 2023
