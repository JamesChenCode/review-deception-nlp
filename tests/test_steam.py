"""Tests for src.steam_collector pure normalizers (no network)."""

from __future__ import annotations

from src import steam_collector as st
from src.fallback_loader import REVIEW_COLUMNS


def test_normalize_review_to_unified_plus_extras():
    obj = {
        "recommendationid": "r1",
        "author": {"steamid": "s1", "num_reviews": 12, "num_games_owned": 40, "playtime_forever": 300},
        "review": "fun game <br> great", "timestamp_created": 1700000000,
        "voted_up": True, "votes_up": 5, "votes_funny": 1, "weighted_vote_score": "0.6",
        "comment_count": 2, "steam_purchase": True, "received_for_free": False,
        "written_during_early_access": False,
    }
    meta = {"product_id": "730", "product_title": "CS2", "price": 0.0, "brand": "Valve", "main_category": "Action"}
    r = st.normalize_review(obj, 730, meta)
    for col in REVIEW_COLUMNS:
        assert col in r
    assert r["product_id"] == "730" and r["reviewer_id"] == "s1" and r["source"] == "steam"
    assert r["verified_purchase"] is True and r["rating"] == 5.0   # voted_up -> recommend
    assert r["helpful_votes"] == 5
    assert r["review_date"].year == 2023
    # Steam extras
    assert r["author_playtime_forever"] == 300 and r["author_num_reviews"] == 12
    assert r["weighted_vote_score"] == 0.6 and r["received_for_free"] is False


def test_voted_down_maps_to_low_rating():
    r = st.normalize_review({"voted_up": False, "author": {}}, 1, {})
    assert r["rating"] == 1.0 and r["verified_purchase"] is False


def test_normalize_meta_parses_price_and_genre():
    data = {"name": "CS2", "genres": [{"description": "Action"}],
            "price_overview": {"final": 5999}, "developers": ["Valve"]}
    m = st.normalize_meta(data, 730)
    assert m["product_title"] == "CS2" and m["price"] == 59.99
    assert m["brand"] == "Valve" and m["main_category"] == "Action"
