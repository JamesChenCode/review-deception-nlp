"""Feature engineering — text + behavioral signals (pure, testable functions).

All builders take a DataFrame in the (cleaned) unified schema and return a new
DataFrame with added columns, or return a matrix + fitted vectorizer. Nothing
here mutates its input.

Two families of features
------------------------
TEXT (in-scope NLP): bag-of-words, TF-IDF / vector-space model, n-grams,
sentiment (nltk VADER), plus simple stylometric stats.

BEHAVIORAL (these STRENGTHEN the weak ``verified_purchase`` proxy — they do not
assume it is correct):
    * reviewer review-burst timing (max reviews in a rolling window),
    * per-product rating-distribution skew,
    * duplicate / near-duplicate text (cosine over per-product TF-IDF),
    * helpful-vote ratios (Laplace-smoothed) + within-product normalization,
    * reviewer history breadth (distinct products / categories).

Step 5 (notebook 07) checks which of these survive against a real fake-review
label versus only correlating with purchase verification.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import utils

LOGGER = utils.get_logger("review_deception.features")

# Numeric columns assembled into the modeling design matrix (Step 4).
NUMERIC_FEATURES: tuple[str, ...] = (
    "rating", "helpful_votes", "price",
    "char_count", "word_count", "avg_word_len",
    "exclaim_count", "question_count", "upper_ratio", "digit_ratio",
    "sentiment_compound", "sentiment_pos", "sentiment_neg", "sentiment_neu",
    "reviewer_review_count", "reviewer_product_count", "reviewer_category_breadth",
    "reviewer_mean_rating", "reviewer_rating_std",
    "product_review_count", "product_mean_rating", "product_rating_std",
    "product_rating_skew", "product_verified_ratio",
    "reviewer_burst_max", "product_burst_max",
    "helpful_ratio", "helpful_votes_pz", "max_intra_product_similarity",
)


# ---------------------------------------------------------------------------
# Text features
# ---------------------------------------------------------------------------
def add_text_stats(df: pd.DataFrame, text_col: str = "review_text") -> pd.DataFrame:
    """Add simple stylometric counts/ratios (length, punctuation, case, digits)."""
    df = df.copy()
    texts = df[text_col].fillna("").astype("string")
    values = [str(t) for t in texts]
    df["char_count"] = [len(s) for s in values]
    df["word_count"] = [len(s.split()) for s in values]
    df["avg_word_len"] = [
        (sum(len(w) for w in s.split()) / len(s.split())) if s.split() else 0.0 for s in values
    ]
    df["exclaim_count"] = [s.count("!") for s in values]
    df["question_count"] = [s.count("?") for s in values]
    df["upper_ratio"] = [
        (sum(1 for ch in s if ch.isupper()) / len(s)) if s else 0.0 for s in values
    ]
    df["digit_ratio"] = [
        (sum(1 for ch in s if ch.isdigit()) / len(s)) if s else 0.0 for s in values
    ]
    return df


_VADER = None


def _get_vader():
    """Lazily build a single VADER analyzer (requires the ``vader_lexicon`` corpus)."""
    global _VADER
    if _VADER is None:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        _VADER = SentimentIntensityAnalyzer()
    return _VADER


def add_sentiment(df: pd.DataFrame, text_col: str = "review_text") -> pd.DataFrame:
    """Add VADER ``sentiment_{compound,pos,neg,neu}`` columns.

    The compound score is a useful "over-positivity" signal: a flood of maximal
    positivity is one (weak) marker of incentivized reviews.
    """
    df = df.copy()
    sia = _get_vader()
    scores = [sia.polarity_scores(str(t) if t is not None else "") for t in df[text_col].fillna("")]
    df["sentiment_compound"] = [s["compound"] for s in scores]
    df["sentiment_pos"] = [s["pos"] for s in scores]
    df["sentiment_neg"] = [s["neg"] for s in scores]
    df["sentiment_neu"] = [s["neu"] for s in scores]
    return df


def _text_cfg(config: dict | None) -> dict:
    return (config or utils.load_config())["features"]["text"]


def build_tfidf(
    texts: Iterable[str],
    config: dict | None = None,
    *,
    ngram_range: tuple[int, int] | None = None,
    max_features: int | None = None,
    min_df: int | float | None = None,
    max_df: int | float | None = None,
):
    """Fit a TF-IDF vector-space model; returns ``(sparse_matrix, vectorizer)``.

    Defaults come from ``config['features']['text']``; override per-call (handy
    for tiny fixtures where the configured ``min_df`` would empty the vocabulary).
    """
    cfg = _text_cfg(config)
    vec = TfidfVectorizer(
        ngram_range=tuple(ngram_range or cfg["ngram_range"]),
        max_features=max_features or cfg["max_features"],
        min_df=min_df if min_df is not None else cfg["tfidf_min_df"],
        max_df=max_df if max_df is not None else cfg["tfidf_max_df"],
        lowercase=cfg.get("lowercase", True),
        stop_words="english",
    )
    return vec.fit_transform(list(texts)), vec


def build_bow(
    texts: Iterable[str],
    config: dict | None = None,
    *,
    ngram_range: tuple[int, int] | None = None,
    min_df: int | float | None = None,
):
    """Fit a bag-of-words (count) model; returns ``(sparse_matrix, vectorizer)``."""
    cfg = _text_cfg(config)
    vec = CountVectorizer(
        ngram_range=tuple(ngram_range or cfg["ngram_range"]),
        min_df=min_df if min_df is not None else cfg["tfidf_min_df"],
        lowercase=cfg.get("lowercase", True),
        stop_words="english",
    )
    return vec.fit_transform(list(texts)), vec


# ---------------------------------------------------------------------------
# Behavioral features
# ---------------------------------------------------------------------------
def add_reviewer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-reviewer aggregates broadcast back to each row (history breadth, etc.)."""
    df = df.copy()
    g = df.groupby("reviewer_id")
    df["reviewer_review_count"] = g["review_id"].transform("size").astype("int64")
    df["reviewer_product_count"] = g["product_id"].transform("nunique").astype("int64")
    df["reviewer_category_breadth"] = g["product_category"].transform("nunique").astype("int64")
    df["reviewer_mean_rating"] = g["rating"].transform("mean").astype("float64")
    df["reviewer_rating_std"] = g["rating"].transform("std").fillna(0.0).astype("float64")
    return df


def add_product_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-product aggregates: counts, rating mean/std/skew, verified-proxy rate."""
    from scipy.stats import skew

    df = df.copy()
    g = df.groupby("product_id")
    df["product_review_count"] = g["review_id"].transform("size").astype("int64")
    df["product_mean_rating"] = g["rating"].transform("mean").astype("float64")
    df["product_rating_std"] = g["rating"].transform("std").fillna(0.0).astype("float64")
    df["product_rating_skew"] = g["rating"].transform(
        lambda s: float(skew(s, bias=False)) if (s.notna().sum() > 2 and s.std(ddof=0) > 0) else 0.0
    ).astype("float64")
    vp = df["verified_purchase"].map({True: 1.0, False: 0.0})  # nullable -> NaN for NA
    df["product_verified_ratio"] = vp.groupby(df["product_id"]).transform("mean").astype("float64")
    return df


def _max_window_count(times_seconds: Sequence[float], window_seconds: float) -> int:
    """Max number of timestamps falling within any ``window_seconds``-wide window."""
    ts = sorted(t for t in times_seconds if t == t)  # drop NaN
    if not ts:
        return 0
    left = 0
    best = 1
    for right in range(len(ts)):
        while ts[right] - ts[left] > window_seconds:
            left += 1
        best = max(best, right - left + 1)
    return best


def add_burst_features(
    df: pd.DataFrame, window_days: float | None = None, config: dict | None = None
) -> pd.DataFrame:
    """Add ``reviewer_burst_max`` / ``product_burst_max`` (rolling-window peaks)."""
    df = df.copy()
    w = window_days if window_days is not None else (
        (config or utils.load_config())["features"]["behavioral"]["burst_window_days"])
    window_seconds = float(w) * 86_400.0
    ts = df["review_date"].map(lambda t: t.timestamp() if pd.notna(t) else float("nan"))
    df["_ts"] = ts.to_numpy()
    for key, out in (("reviewer_id", "reviewer_burst_max"), ("product_id", "product_burst_max")):
        df[out] = df.groupby(key)["_ts"].transform(
            lambda s: _max_window_count(list(s), window_seconds)
        ).astype("int64")
    return df.drop(columns="_ts")


def add_helpful_ratio(
    df: pd.DataFrame, smoothing: int | None = None, config: dict | None = None
) -> pd.DataFrame:
    """Add Laplace-smoothed ``helpful_ratio`` + within-product z-score of votes.

    ``helpful_ratio`` needs ``total_votes`` (present in the 2014 edition; ``NaN``
    where unavailable). ``helpful_votes_pz`` normalizes raw up-votes within each
    product so a review's helpfulness is judged against its peers.
    """
    df = df.copy()
    sm = smoothing if smoothing is not None else (
        (config or utils.load_config())["features"]["behavioral"]["helpful_vote_smoothing"])
    hv = pd.to_numeric(df["helpful_votes"], errors="coerce").astype("float64")
    tv = pd.to_numeric(df.get("total_votes"), errors="coerce").astype("float64")
    df["helpful_ratio"] = (hv + sm) / (tv + 2 * sm)
    g = df.groupby("product_id")["helpful_votes"]
    mean = g.transform("mean")
    std = g.transform("std").replace(0.0, np.nan)
    df["helpful_votes_pz"] = ((hv - mean) / std).astype("float64").fillna(0.0)
    return df


def add_near_duplicate_features(
    df: pd.DataFrame,
    threshold: float | None = None,
    config: dict | None = None,
    text_col: str = "review_text",
) -> pd.DataFrame:
    """Per product, add max cosine TF-IDF similarity to any sibling review.

    O(n^2) within each product group — intended for course-scale data. Reviews on
    products with <2 reviews get similarity 0.
    """
    df = df.copy().reset_index(drop=True)
    th = threshold if threshold is not None else (
        (config or utils.load_config())["features"]["behavioral"]["near_duplicate_threshold"])
    sims = np.zeros(len(df), dtype="float64")
    for _, idx in df.groupby("product_id").groups.items():
        idx = list(idx)
        texts = [str(t) for t in df.loc[idx, text_col].fillna("")]
        if len(idx) < 2 or all(not t.strip() for t in texts):
            continue
        try:
            matrix = TfidfVectorizer().fit_transform(texts)
        except ValueError:  # empty vocabulary (all stop words / empty)
            continue
        similarity = cosine_similarity(matrix)
        np.fill_diagonal(similarity, 0.0)
        row_max = similarity.max(axis=1)
        for local, global_i in enumerate(idx):
            sims[global_i] = row_max[local]
    df["max_intra_product_similarity"] = sims
    df["is_near_duplicate"] = df["max_intra_product_similarity"] >= th
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_feature_table(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Run every feature builder in order; returns the enriched DataFrame."""
    cfg = config or utils.load_config()
    out = add_text_stats(df)
    out = add_sentiment(out)
    out = add_reviewer_features(out)
    out = add_product_features(out)
    out = add_burst_features(out, config=cfg)
    out = add_helpful_ratio(out, config=cfg)
    out = add_near_duplicate_features(out, config=cfg)
    LOGGER.info("Built feature table: %d rows × %d columns.", len(out), out.shape[1])
    return out


def assemble_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select the numeric design matrix for modeling (NaN-filled with 0.0)."""
    cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    numeric = df[cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
    return numeric.fillna(0.0)
