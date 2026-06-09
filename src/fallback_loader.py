"""Fallback loader: McAuley/UCSD + Hou et al. 2024 Amazon Reviews → unified schema.

This is the documented, ToS-safe data path. It reads the research dataset from
``data/raw/`` and emits the **same unified review schema** the scraper does, so
everything downstream is source-agnostic.

THE UNIFIED REVIEW SCHEMA (canonical definition)
------------------------------------------------
Both :mod:`src.scrape` and this module normalize to these columns. Downstream
modules depend ONLY on this contract.

Per-review columns
    review_id          : str       stable id (hashed if the source lacks one)
    product_id         : str       join key (parent_asin / asin)
    product_category   : str       file-level category, e.g. "Electronics"
    reviewer_id        : str       pseudonymous reviewer id
    rating             : float      star rating in [1, 5]
    review_title       : str       short title / summary (may be empty)
    review_text        : str       full review body
    verified_purchase  : bool|None  WEAK PROXY LABEL — purchase verification, NOT
                                     deception. ``None`` when the edition lacks it
                                     (the 2014 McAuley edition has no such flag).
    review_date        : datetime   timezone-aware UTC
    helpful_votes      : int        up-votes ("found helpful")
    total_votes        : int|None   total votes where available (2014 only)
    source             : str        "scrape" | "fallback" | "groundtruth"

Joined product-metadata columns (JOIN source #2, merged on ``product_id``)
    product_title      : str
    price              : float|None USD where parseable
    brand              : str|None
    main_category      : str|None

Reviewer aggregates (DERIVED later in :mod:`src.features`, not here).

Edition note
------------
The PRIMARY proxy label (``verified_purchase``) exists in the **2023** edition
and in live scraping, but NOT in the 2014 edition — hence the default
``fallback.dataset: amazon_reviews_2023``. Field names below follow the
published formats; the real download is a gated action (see README/Step 2) at
which point the exact format is reconfirmed before bulk use.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from . import utils

LOGGER = utils.get_logger("review_deception.fallback")

SOURCE = "fallback"

REVIEW_COLUMNS: tuple[str, ...] = (
    "review_id",
    "product_id",
    "product_category",
    "reviewer_id",
    "rating",
    "review_title",
    "review_text",
    "verified_purchase",
    "review_date",
    "helpful_votes",
    "total_votes",
    "product_title",
    "price",
    "brand",
    "main_category",
    "source",
)


# ---------------------------------------------------------------------------
# Small pure parsers (stdlib only, unit-testable)
# ---------------------------------------------------------------------------
def parse_price(raw: Any) -> float | None:
    """Coerce assorted price encodings ("$12.99", "1,234.50", 12.99) to float."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip().replace("$", "").replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_epoch(raw: Any, unit: str = "ms") -> datetime | None:
    """Convert an epoch to a UTC ``datetime``. ``unit`` = 'ms' | 's' | 'auto'.

    'auto' picks milliseconds when the value looks like ms (> 1e12), else seconds
    — robust to the 2023 edition's millisecond timestamps without assuming a unit.
    """
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if unit == "auto":
        unit = "ms" if abs(value) > 1e12 else "s"
    if unit == "ms":
        value /= 1000.0
    try:
        return datetime.fromtimestamp(value, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _as_bool(raw: Any) -> bool | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"true", "1", "yes", "y"}


# ---------------------------------------------------------------------------
# Edition-specific review/metadata normalizers (raw dict -> unified dict)
# ---------------------------------------------------------------------------
def normalize_review_2023(obj: dict[str, Any], category: str) -> dict[str, Any]:
    """Map an Amazon-Reviews-2023 (Hou et al.) review object to the unified schema."""
    product_id = obj.get("parent_asin") or obj.get("asin") or ""
    reviewer_id = obj.get("user_id") or ""
    text = obj.get("text") or ""
    return {
        "review_id": obj.get("review_id") or utils.stable_id(product_id, reviewer_id, text),
        "product_id": str(product_id),
        "product_category": category,
        "reviewer_id": str(reviewer_id),
        "rating": float(obj["rating"]) if obj.get("rating") is not None else None,
        "review_title": obj.get("title") or "",
        "review_text": text,
        "verified_purchase": _as_bool(obj.get("verified_purchase")),
        "review_date": parse_epoch(obj.get("timestamp"), unit="auto"),
        # field name varies across edition/mirror ("helpful_vote" vs "helpful_votes")
        "helpful_votes": int(obj.get("helpful_vote") or obj.get("helpful_votes") or 0),
        "total_votes": None,  # not provided in the 2023 edition
        "source": SOURCE,
    }


def normalize_review_2014(obj: dict[str, Any], category: str) -> dict[str, Any]:
    """Map a 2014 McAuley review object to the unified schema.

    NOTE: the 2014 edition has NO verified-purchase flag, so ``verified_purchase``
    is ``None`` (the weak proxy label is simply unavailable for this edition).
    """
    product_id = obj.get("asin") or ""
    reviewer_id = obj.get("reviewerID") or ""
    text = obj.get("reviewText") or ""
    helpful = obj.get("helpful") or [0, 0]
    up, total = (helpful + [0, 0])[:2]
    return {
        "review_id": utils.stable_id(product_id, reviewer_id, text),
        "product_id": str(product_id),
        "product_category": category,
        "reviewer_id": str(reviewer_id),
        "rating": float(obj["overall"]) if obj.get("overall") is not None else None,
        "review_title": obj.get("summary") or "",
        "review_text": text,
        "verified_purchase": None,
        "review_date": parse_epoch(obj.get("unixReviewTime"), unit="s"),
        "helpful_votes": int(up or 0),
        "total_votes": int(total) if total else None,
        "source": SOURCE,
    }


def normalize_meta_2023(obj: dict[str, Any]) -> dict[str, Any]:
    return {
        "product_id": str(obj.get("parent_asin") or obj.get("asin") or ""),
        "product_title": obj.get("title") or "",
        "price": parse_price(obj.get("price")),
        "brand": obj.get("store"),  # 2023 uses "store" for the seller/brand
        "main_category": obj.get("main_category"),
    }


def normalize_meta_2014(obj: dict[str, Any]) -> dict[str, Any]:
    categories = obj.get("categories") or []
    main = categories[0][0] if categories and categories[0] else None
    return {
        "product_id": str(obj.get("asin") or ""),
        "product_title": obj.get("title") or "",
        "price": parse_price(obj.get("price")),
        "brand": obj.get("brand"),
        "main_category": main,
    }


_REVIEW_NORMALIZERS = {
    "amazon_reviews_2023": normalize_review_2023,
    "amazon_reviews_2014": normalize_review_2014,
}
_META_NORMALIZERS = {
    "amazon_reviews_2023": normalize_meta_2023,
    "amazon_reviews_2014": normalize_meta_2014,
}


# ---------------------------------------------------------------------------
# I/O + join
# ---------------------------------------------------------------------------
def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file, transparently handling ``.gz``."""
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _resolve_file(raw_dir: Path, name: str) -> Path | None:
    """Return the first existing path among ``name`` and ``name + '.gz'`` (else None).

    Real 2023 files are gzipped (``Subscription_Boxes.jsonl.gz``); synthetic test
    samples are plain ``.jsonl`` — this accepts either.
    """
    for candidate in (raw_dir / name, raw_dir / f"{name}.gz"):
        if candidate.is_file():
            return candidate
    return None


def join_reviews_meta(
    reviews: Iterable[dict[str, Any]], meta: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Left-join normalized reviews with normalized product metadata on product_id."""
    meta_index = {m["product_id"]: m for m in meta}
    empty = {"product_title": "", "price": None, "brand": None, "main_category": None}
    joined: list[dict[str, Any]] = []
    for r in reviews:
        extra = meta_index.get(r["product_id"], empty)
        joined.append({**r, **{k: extra.get(k) for k in empty}})
    return joined


def load_category(
    config: dict[str, Any], category: str, raw_dir: str | Path | None = None
) -> list[dict[str, Any]]:
    """Load one category's reviews, normalize, and join product metadata.

    Returns a list of unified-schema dicts (use :func:`utils.records_to_frame`
    to get a DataFrame when pandas is available).
    """
    fb = config["fallback"]
    edition = fb["dataset"]
    if edition not in _REVIEW_NORMALIZERS:
        raise ValueError(f"Unknown fallback.dataset {edition!r}; expected one of {list(_REVIEW_NORMALIZERS)}")
    root = utils.project_root()
    raw = Path(raw_dir) if raw_dir else root / config["paths"]["data_raw"]
    reviews_name = fb["reviews_filename_template"].format(category=category)
    meta_name = fb["meta_filename_template"].format(category=category)
    reviews_path = _resolve_file(raw, reviews_name)
    meta_path = _resolve_file(raw, meta_name)
    if reviews_path is None:
        raise FileNotFoundError(
            f"Reviews file not found for {category!r} in {raw} (looked for "
            f"{reviews_name}[.gz]). Download the fallback dataset first "
            f"(see README; this loader never downloads automatically)."
        )

    review_norm = _REVIEW_NORMALIZERS[edition]
    meta_norm = _META_NORMALIZERS[edition]
    reviews = [review_norm(o, category) for o in iter_jsonl(reviews_path)]
    meta = [meta_norm(o) for o in iter_jsonl(meta_path)] if meta_path else []
    if not meta:
        LOGGER.warning("No metadata file at %s; join columns will be empty.", meta_path)
    rows = join_reviews_meta(reviews, meta)

    sample = fb.get("sample_rows")
    if sample and len(rows) > sample:
        rng = random.Random(config["project"]["random_seed"])
        rows = rng.sample(rows, sample)
    LOGGER.info("Loaded %d reviews for category %r from %s", len(rows), category, reviews_path.name)
    return rows


def load_all(config: dict[str, Any], raw_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Load and concatenate every category in ``config['fallback']['categories']``."""
    rows: list[dict[str, Any]] = []
    for category in config["fallback"]["categories"]:
        try:
            rows.extend(load_category(config, category, raw_dir=raw_dir))
        except FileNotFoundError as exc:
            LOGGER.warning("Skipping %r: %s", category, exc)
    return rows


# ---------------------------------------------------------------------------
# Synthetic sample (dry runs + Step-3 test fixtures). Deterministic for a seed.
# ---------------------------------------------------------------------------
def make_synthetic_sample(
    out_dir: str | Path,
    categories: Sequence[str] = ("Electronics",),
    n_products: int = 5,
    reviews_per_product: int = 6,
    seed: int = 42,
) -> dict[str, list[Path]]:
    """Write tiny 2023-format ``{category}.jsonl`` + ``meta_{category}.jsonl`` files.

    The data intentionally embeds patterns later analyses look for: a verified vs.
    unverified mix, a near-duplicate "generic praise" template, a review burst,
    and skewed ratings on one product. Returns ``{"reviews": [...], "meta": [...]}``.
    """
    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    generic = "Great product works as expected highly recommend to everyone"
    base_ts = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    day_ms = 86_400_000
    written: dict[str, list[Path]] = {"reviews": [], "meta": []}

    for category in categories:
        reviews: list[dict[str, Any]] = []
        metas: list[dict[str, Any]] = []
        for p in range(n_products):
            asin = f"B{seed:03d}{p:04d}"
            suspicious = p == 0  # product 0 gets a burst of generic 5-star unverified reviews
            metas.append({
                "parent_asin": asin,
                "title": f"{category} Item {p}",
                "price": round(rng.uniform(8, 120), 2),
                "store": rng.choice(["Acme", "Globex", "Initech", "Umbrella"]),
                "main_category": category,
            })
            for j in range(reviews_per_product):
                if suspicious:
                    rating, verified = 5.0, False
                    text = generic if j % 2 == 0 else generic + " love it"
                    ts = base_ts + j * 3_600_000  # burst within hours
                else:
                    rating = float(rng.choice([1, 3, 4, 5, 5]))
                    verified = rng.random() < 0.7
                    text = rng.choice([
                        "Solid build quality and the battery lasts all day during travel.",
                        "Stopped working after two weeks, disappointed with the durability.",
                        "Exactly what I needed, detailed setup guide made install painless.",
                        generic,
                    ])
                    ts = base_ts + (p * reviews_per_product + j) * day_ms
                reviews.append({
                    "parent_asin": asin,
                    "asin": asin,
                    "user_id": f"U{rng.randint(0, 40):03d}" if not suspicious else f"S{j:03d}",
                    "rating": rating,
                    "title": "Five Stars" if rating >= 4 else "Not great",
                    "text": text,
                    "verified_purchase": verified,
                    "timestamp": ts,
                    "helpful_vote": rng.randint(0, 25),
                })
        reviews_path = out / f"{category}.jsonl"
        meta_path = out / f"meta_{category}.jsonl"
        with reviews_path.open("w", encoding="utf-8") as fh:
            for r in reviews:
                fh.write(json.dumps(r) + "\n")
        with meta_path.open("w", encoding="utf-8") as fh:
            for m in metas:
                fh.write(json.dumps(m) + "\n")
        written["reviews"].append(reviews_path)
        written["meta"].append(meta_path)
        LOGGER.info("Wrote synthetic sample: %s (%d reviews), %s (%d products)",
                    reviews_path.name, len(reviews), meta_path.name, len(metas))
    return written


def _demo() -> None:
    """`python -m src.fallback_loader --demo`: synthesize + load a tiny sample."""
    parser = argparse.ArgumentParser(description="Fallback loader demo (no download).")
    parser.add_argument("--demo", action="store_true", help="run a synthetic-sample demo")
    parser.add_argument("--n-products", type=int, default=5)
    args = parser.parse_args()

    config = utils.load_config()
    utils.set_seeds(config["project"]["random_seed"])
    sample_dir = utils.project_root() / config["paths"]["data_raw"] / "_dryrun_fallback"
    make_synthetic_sample(sample_dir, categories=("Electronics",), n_products=args.n_products)
    rows = load_category(config, "Electronics", raw_dir=sample_dir)
    LOGGER.info("Unified schema columns: %s", ", ".join(REVIEW_COLUMNS))
    print(utils.preview_table(rows, columns=[
        "review_id", "product_id", "reviewer_id", "rating",
        "verified_purchase", "helpful_votes", "price", "review_text",
    ]))


if __name__ == "__main__":
    _demo()
