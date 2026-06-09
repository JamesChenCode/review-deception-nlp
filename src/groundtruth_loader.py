"""Ground-truth loader: labeled fake-review data → unified schema (Step 5).

Loads the Hollenbeck et al. labeled fake-review dataset and maps it onto the
SAME unified review schema (see :mod:`src.fallback_loader`) plus one extra
column ``is_fake`` (the REAL label), so the trust classifier can be re-run on
ground truth through the identical pipeline.

Source & license (CONFIRMED 2026-06-08, not hardcoded from memory)
------------------------------------------------------------------
Repo : https://github.com/bretthollenbeck/fake-reviews-data  (LICENSE: MIT)
Data : public_reviews_dataset_cleaned.csv  — 381,734 rows, 23 columns, served as
       a ~64 MB zip from the author's site. Download instructions in the README;
       this loader only READS a local file (never downloads).
Paper: He, Hollenbeck, Overgoor, Proserpio, Tosyali — "Detecting fake-review
       buyers using network structure" (PNAS).

Confirmed columns used (verbatim from the real header)
------------------------------------------------------
Base review fields → unified schema (see ``COLUMN_MAP``):
    asin, review_id, reviewer_id, review_title, review_text, review_rating,
    review_date, product_title, number_of_helpful.
Label fields:
    fake_review_product (bool)            — product buys fake reviews,
    reviewer_classified_fake (bool)       — reviewer classified fake,
    reviewer_classified_honest (bool),
    reviewer_labeled_fake (0/1/NaN)       — human/rule reviewer label (subset),
    reviewer_labeled_honest (0/1/NaN),
    review_is_removed_by_amazon (0/1/NaN) — Amazon deletion 2020-2023 (Hou 2024),
    fake_review_campaign_start_date (date).

There is NO ``verified_purchase`` in this dataset (it is the ground-truth set),
so the weak proxy column is emitted as ``<NA>`` here.

Label construction
------------------
Per the dataset README's primary rule: a review is fake if it is a 5-star
review of a fake-review-purchasing product written by a fake reviewer. Stricter
variants use Amazon deletion or the human reviewer labels — see
:func:`make_groundtruth_label`.
"""

from __future__ import annotations

import argparse
from typing import Mapping

import numpy as np
import pandas as pd

from . import utils
from .fallback_loader import REVIEW_COLUMNS

LOGGER = utils.get_logger("review_deception.groundtruth")

SOURCE = "groundtruth"

# Real-header → unified-schema mapping (confirmed from the file, not memory).
COLUMN_MAP: dict[str, str] = {
    "review_id": "review_id",
    "asin": "product_id",
    "reviewer_id": "reviewer_id",
    "review_rating": "rating",
    "review_title": "review_title",
    "review_text": "review_text",
    "review_date": "review_date",
    "number_of_helpful": "helpful_votes",
    "product_title": "product_title",
}

LABEL_INPUT_COLUMNS = [
    "review_rating", "fake_review_product", "reviewer_classified_fake",
    "reviewer_classified_honest", "reviewer_labeled_fake", "reviewer_labeled_honest",
    "review_is_removed_by_amazon", "fake_review_campaign_start_date",
]

LABEL_STRATEGIES = ("primary", "primary_or_deleted", "deleted", "labeled")


def make_groundtruth_label(df: pd.DataFrame, strategy: str = "primary") -> pd.Series:
    """Construct the real ``is_fake`` label (float 0.0/1.0, ``NaN`` = excluded).

    Strategies
        primary             5-star ∧ fake product ∧ fake reviewer (README default).
        primary_or_deleted  primary, OR the review was deleted by Amazon.
        deleted             review_is_removed_by_amazon == 1 (rows with unknown
                            deletion status are excluded via NaN).
        labeled             reviewer_labeled_fake == 1, restricted to the human-
                            labeled subset (others excluded via NaN).
    """
    if strategy not in LABEL_STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; expected one of {LABEL_STRATEGIES}")
    rating5 = df["review_rating"] == 5
    fake_product = df["fake_review_product"].fillna(False).astype(bool)
    fake_reviewer = df["reviewer_classified_fake"].fillna(False).astype(bool)
    primary = (rating5 & fake_product & fake_reviewer)

    if strategy == "primary":
        return primary.astype("float64")
    if strategy == "primary_or_deleted":
        deleted = df["review_is_removed_by_amazon"].fillna(0) == 1
        return (primary | deleted).astype("float64")
    if strategy == "deleted":
        removed = df["review_is_removed_by_amazon"]
        out = (removed == 1).astype("float64")
        return out.where(removed.notna(), other=np.nan)
    # strategy == "labeled"
    labeled = df["reviewer_labeled_fake"]
    out = (labeled == 1).astype("float64")
    return out.where(labeled.notna(), other=np.nan)


def _to_unified(df: pd.DataFrame, is_fake: pd.Series) -> pd.DataFrame:
    """Map confirmed columns onto the unified schema and attach ``is_fake``."""
    out = pd.DataFrame(index=df.index)
    for src, dst in COLUMN_MAP.items():
        out[dst] = df[src]
    # Fields not present in the ground-truth dataset:
    out["product_category"] = "unknown"   # no category column in this dataset
    out["verified_purchase"] = pd.NA      # the weak proxy does not exist here
    out["total_votes"] = pd.NA
    out["price"] = pd.NA
    out["brand"] = pd.NA
    out["main_category"] = pd.NA
    out["source"] = SOURCE
    out["is_fake"] = is_fake.values
    return out[list(REVIEW_COLUMNS) + ["is_fake"]]


def load_groundtruth(config: Mapping, strategy: str | None = None,
                     sample_rows: int | None = None, path: str | None = None) -> pd.DataFrame:
    """Load the labeled dataset → unified schema + ``is_fake`` (rows w/o a label dropped).

    Reads only the columns it needs. ``sample_rows`` (or ``groundtruth.sample_rows``)
    subsamples deterministically for a tractable feature pipeline on the full set.
    """
    gt = config["groundtruth"]
    strategy = strategy or gt.get("label_strategy", "primary")
    root = utils.project_root()
    csv_path = path or (root / config["paths"]["data_groundtruth"] / gt["filename"])
    csv_path = str(csv_path)

    usecols = sorted(set(COLUMN_MAP) | set(LABEL_INPUT_COLUMNS))
    df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    is_fake = make_groundtruth_label(df, strategy=strategy)

    out = _to_unified(df, is_fake)
    out = out[out["is_fake"].notna()].copy()
    out["is_fake"] = out["is_fake"].astype("int64")

    n_sample = sample_rows if sample_rows is not None else gt.get("sample_rows")
    if n_sample and len(out) > n_sample:
        out = out.sample(n=n_sample, random_state=config["project"]["random_seed"])
    out = out.reset_index(drop=True)
    LOGGER.info("Loaded %d ground-truth reviews (strategy=%r, fake rate=%.3f)",
                len(out), strategy, out["is_fake"].mean())
    return out


def _summary() -> None:
    """`python -m src.groundtruth_loader --summary`: load + print label balance."""
    parser = argparse.ArgumentParser(description="Ground-truth loader summary.")
    parser.add_argument("--strategy", default=None, choices=LABEL_STRATEGIES)
    parser.add_argument("--sample-rows", type=int, default=5000)
    args = parser.parse_args()
    config = utils.load_config()
    df = load_groundtruth(config, strategy=args.strategy, sample_rows=args.sample_rows)
    print(f"rows={len(df)}  fake={int(df['is_fake'].sum())}  fake_rate={df['is_fake'].mean():.3f}")
    print(df[["review_id", "product_id", "reviewer_id", "rating", "is_fake"]].head().to_string(index=False))


if __name__ == "__main__":
    _summary()
