"""Tests for src.groundtruth_loader on a tiny synthetic CSV in the REAL format."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import clean, groundtruth_loader as gtl, utils
from src.fallback_loader import REVIEW_COLUMNS


def _gt_row(rid, asin, reviewer, rating, fake_prod, fake_rev, removed, labeled_fake):
    return {
        "review_id": rid, "asin": asin, "reviewer_id": reviewer,
        "review_title": "t", "review_text": "some review text here", "review_rating": rating,
        "review_date": "2021-03-01", "product_title": "Prod", "number_of_helpful": 1.0,
        "fake_review_product": fake_prod, "reviewer_classified_fake": fake_rev,
        "reviewer_classified_honest": not fake_rev,
        "reviewer_labeled_fake": labeled_fake, "reviewer_labeled_honest": np.nan,
        "review_is_removed_by_amazon": removed,
        "fake_review_campaign_start_date": "2020-02-08",
    }


@pytest.fixture
def gt_frame() -> pd.DataFrame:
    rows = [
        _gt_row("g1", "A1", "U1", 5, True, True, 0.0, 1.0),    # primary fake
        _gt_row("g2", "A1", "U2", 3, True, True, 0.0, 0.0),    # not 5-star -> real
        _gt_row("g3", "A2", "U3", 5, False, False, 1.0, np.nan),  # non-fake product; deleted
        _gt_row("g4", "A2", "U4", 5, False, True, 0.0, np.nan),   # non-fake product -> real
        _gt_row("g5", "A3", "U5", 5, True, False, 1.0, 0.0),   # honest reviewer; deleted
        _gt_row("g6", "A3", "U1", 5, True, True, np.nan, np.nan),  # primary fake; deletion unknown
    ]
    return pd.DataFrame(rows)


def test_label_primary(gt_frame):
    y = gtl.make_groundtruth_label(gt_frame, "primary")
    assert y.tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def test_label_primary_or_deleted(gt_frame):
    y = gtl.make_groundtruth_label(gt_frame, "primary_or_deleted")
    assert y.tolist() == [1.0, 0.0, 1.0, 0.0, 1.0, 1.0]  # adds the deleted g3, g5


def test_label_deleted_excludes_unknown(gt_frame):
    y = gtl.make_groundtruth_label(gt_frame, "deleted")
    assert pd.isna(y.iloc[5])                 # g6 has unknown deletion status
    assert int(y[y.notna()].sum()) == 2       # g3, g5
    assert int(y.notna().sum()) == 5


def test_label_labeled_subset(gt_frame):
    y = gtl.make_groundtruth_label(gt_frame, "labeled")
    assert int(y.notna().sum()) == 3          # only rows with a human reviewer label
    assert int(y[y.notna()].sum()) == 1       # g1


def test_load_groundtruth_unified_schema(gt_frame, tmp_path):
    csv = tmp_path / "gt.csv"
    gt_frame.to_csv(csv, index=False)
    out = gtl.load_groundtruth(utils.load_config(), strategy="primary", path=str(csv))
    assert set(REVIEW_COLUMNS).issubset(out.columns)
    assert "is_fake" in out.columns
    assert len(out) == 6
    assert int(out["is_fake"].sum()) == 2
    assert (out["product_id"] == gt_frame["asin"]).all()      # asin -> product_id
    assert out["verified_purchase"].isna().all()              # no proxy in GT data
    assert (out["source"] == "groundtruth").all()


def test_groundtruth_flows_through_clean_and_features(gt_frame, tmp_path):
    csv = tmp_path / "gt.csv"
    gt_frame.to_csv(csv, index=False)
    out = gtl.load_groundtruth(utils.load_config(), strategy="primary", path=str(csv))
    cleaned = clean.clean_reviews(out)            # is_fake passes through cleaning
    assert "is_fake" in cleaned.columns
    assert str(cleaned["review_date"].dtype).startswith("datetime64")
