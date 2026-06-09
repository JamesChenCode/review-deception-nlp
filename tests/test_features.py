"""Tests for src.features — text + behavioral feature builders."""

from __future__ import annotations

import pandas as pd

from src import features


def test_text_stats(clean_reviews_df):
    df = features.add_text_stats(clean_reviews_df).set_index("review_id")
    assert df.loc["r1", "word_count"] == 5  # "Great product works as expected"
    assert df.loc["r1", "char_count"] == len("Great product works as expected")
    assert df.loc["r1", "exclaim_count"] == 0


def test_sentiment_sign(clean_reviews_df):
    df = features.add_sentiment(clean_reviews_df).set_index("review_id")
    assert df.loc["r4", "sentiment_compound"] > 0   # "wonderful and helpful"
    assert df.loc["r5", "sentiment_compound"] < 0   # "terrible, waste of money"


def test_reviewer_features(clean_reviews_df):
    df = features.add_reviewer_features(clean_reviews_df).set_index("review_id")
    # U1 reviewed P1 (Electronics) and P3 (Books)
    assert df.loc["r1", "reviewer_review_count"] == 2
    assert df.loc["r1", "reviewer_product_count"] == 2
    assert df.loc["r1", "reviewer_category_breadth"] == 2
    # U4 reviewed P2 twice (same category)
    assert df.loc["r4", "reviewer_review_count"] == 2
    assert df.loc["r4", "reviewer_category_breadth"] == 1


def test_product_features(clean_reviews_df):
    df = features.add_product_features(clean_reviews_df).set_index("review_id")
    assert df.loc["r1", "product_review_count"] == 4   # P1: r1,r2,r3,r7
    assert df.loc["r4", "product_review_count"] == 3   # P2: r4,r5,r8
    # P1 verified flags = [F, F, T, F] -> ratio 0.25
    assert df.loc["r1", "product_verified_ratio"] == 0.25
    assert "product_rating_skew" in df.columns


def test_burst_features(clean_reviews_df):
    df = features.add_burst_features(clean_reviews_df, window_days=7).set_index("review_id")
    assert df.loc["r1", "product_burst_max"] == 4   # four P1 reviews within hours
    assert df.loc["r1", "reviewer_burst_max"] == 2   # U1: two reviews within 3 days


def test_near_duplicate(clean_reviews_df):
    df = features.add_near_duplicate_features(clean_reviews_df, threshold=0.9).set_index("review_id")
    assert bool(df.loc["r1", "is_near_duplicate"]) is True
    assert df.loc["r1", "max_intra_product_similarity"] >= 0.99
    assert bool(df.loc["r3", "is_near_duplicate"]) is False  # distinct text on P1


def test_helpful_ratio(clean_reviews_df):
    df = features.add_helpful_ratio(clean_reviews_df, smoothing=1).set_index("review_id")
    assert 0 < df.loc["r4", "helpful_ratio"] <= 1   # total_votes present
    assert pd.isna(df.loc["r1", "helpful_ratio"])   # no total_votes -> undefined


def test_build_tfidf_shape(clean_reviews_df):
    matrix, vec = features.build_tfidf(clean_reviews_df["review_text"], min_df=1, max_df=1.0)
    assert matrix.shape[0] == len(clean_reviews_df)
    assert matrix.shape[1] > 0
    assert len(vec.get_feature_names_out()) == matrix.shape[1]


def test_assemble_behavioral_matrix(clean_reviews_df):
    enriched = features.build_feature_table(clean_reviews_df)
    X = features.assemble_behavioral_features(enriched)
    assert X.shape[0] == len(clean_reviews_df)
    assert int(X.isna().sum().sum()) == 0   # design matrix has no NaNs
    assert X.shape[1] >= 15


def test_feature_builders_are_pure(clean_reviews_df):
    before = clean_reviews_df.copy()
    features.add_text_stats(clean_reviews_df)
    features.add_reviewer_features(clean_reviews_df)
    features.add_near_duplicate_features(clean_reviews_df)
    pd.testing.assert_frame_equal(clean_reviews_df, before)
