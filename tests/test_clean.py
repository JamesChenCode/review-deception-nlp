"""Tests for src.clean — cleaning/normalization pure functions."""

from __future__ import annotations

import pandas as pd
import pytest

from src import clean


def test_clean_text_entities_and_whitespace():
    assert clean.clean_text("Great   product &amp; value\n") == "Great product & value"
    assert clean.clean_text("  ＡＢＣ  ") == "ABC"  # NFKC folds full-width to ASCII + trims
    assert clean.clean_text(None) == ""
    assert clean.clean_text(float("nan")) == ""
    assert clean.clean_text("LOUD", lowercase=True) == "loud"


def test_validate_schema_raises_on_missing(raw_reviews_df):
    with pytest.raises(ValueError):
        clean.validate_schema(raw_reviews_df.drop(columns=["review_text"]))


def test_coerce_dtypes(clean_reviews_df):
    df = clean_reviews_df
    assert df["rating"].dtype == "float64"
    assert df["helpful_votes"].dtype == "int64"
    assert df["verified_purchase"].dtype == "boolean"
    # tz-aware UTC datetime (pandas 3.0 defaults to microsecond resolution)
    assert isinstance(df["review_date"].dtype, pd.DatetimeTZDtype)
    assert str(df["review_date"].dtype.tz) == "UTC"


def test_boolean_na_preserved(clean_reviews_df):
    row = clean_reviews_df.set_index("review_id").loc["r6"]
    assert pd.isna(row["verified_purchase"])  # the None (2014-style) row stays <NA>


def test_text_is_normalized(clean_reviews_df):
    row = clean_reviews_df.set_index("review_id").loc["r6"]
    assert row["review_text"] == "Average book & nothing special"


def test_exact_duplicates_flagged(clean_reviews_df):
    df = clean_reviews_df.set_index("review_id")
    for rid in ("r1", "r2", "r7"):  # identical generic text on P1
        assert bool(df.loc[rid, "is_exact_duplicate"]) is True
    assert bool(df.loc["r3", "is_exact_duplicate"]) is False
    assert int(clean_reviews_df["is_exact_duplicate"].sum()) == 3


def test_drop_exact_duplicates_keeps_distinct(clean_reviews_df):
    # identical text but DIFFERENT reviewers -> not dropped by the (product, reviewer, text) key
    deduped = clean.drop_exact_duplicates(clean_reviews_df)
    assert len(deduped) == len(clean_reviews_df)


def test_clean_reviews_is_pure(raw_reviews_df):
    before = raw_reviews_df.copy()
    clean.clean_reviews(raw_reviews_df)
    pd.testing.assert_frame_equal(raw_reviews_df, before)
