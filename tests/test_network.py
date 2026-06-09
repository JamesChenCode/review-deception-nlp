"""Smoke + correctness tests for src.network."""

from __future__ import annotations

from src import network


def test_bipartite_graph_kinds(medium_reviews_df):
    g = network.build_bipartite_graph(medium_reviews_df)
    assert g.number_of_nodes() > 0
    kinds = {data["kind"] for _, data in g.nodes(data=True)}
    assert kinds == {"reviewer", "product"}


def test_product_projection(medium_reviews_df):
    pg = network.project_products(medium_reviews_df, min_shared=1)
    assert pg.number_of_nodes() == medium_reviews_df["product_id"].nunique()
    # synthetic reviewers are shared across products within a category -> edges exist
    assert pg.number_of_edges() > 0


def test_suspicious_product_table(medium_reviews_df):
    table = network.suspicious_product_table(medium_reviews_df)
    assert len(table) == medium_reviews_df["product_id"].nunique()
    assert {"suspicion_score", "community", "verified_ratio", "burst_max"}.issubset(table.columns)
    assert table["suspicion_score"].is_monotonic_decreasing      # sorted most-suspicious first
    assert table["suspicion_score"].notna().all()
