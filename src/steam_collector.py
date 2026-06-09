"""Steam reviews collector — a self-collected, ToS-friendly cross-platform source.

Amazon blocks polite scraping behind a JS bot wall, so the *self-collected* data
for this project comes from Steam's **public** review API, which maps onto our
thesis almost one-to-one and emits the same unified schema.

Why Steam fits
--------------
* ``steam_purchase`` (bought on Steam) is a direct analog of Amazon's "Verified
  Purchase" weak proxy → mapped to ``verified_purchase``.
* ``received_for_free`` flags *incentivized* reviews (carried as an extra column)
  — a second, sharper trust signal.
* Rich behavioral fields: author ``num_reviews`` / ``num_games_owned`` /
  ``playtime_forever``, ``votes_up`` (helpful), ``weighted_vote_score``, and
  timestamps for bursts.
* ``appdetails`` provides a genre/price/developer **metadata join** (multi-source).

Engineering reuses the robust pieces from :mod:`src.scrape` (rate limiting,
retry-with-backoff, gzipped on-disk cache of raw JSON, checkpointing) — polite by
construction. The API is public and intended for app integrations.

Confirmed field names (2026-06, from a live probe — not hardcoded from memory):
review: recommendationid, author{steamid,num_games_owned,num_reviews,
playtime_forever,...}, review, timestamp_created, voted_up, votes_up, votes_funny,
weighted_vote_score, comment_count, steam_purchase, received_for_free,
written_during_early_access; appdetails.data: name, genres[].description,
price_overview.final, developers.
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

from . import fallback_loader, scrape, utils
from .fallback_loader import REVIEW_COLUMNS

LOGGER = utils.get_logger("review_deception.steam")

SOURCE = "steam"
_REVIEWS_BASE = "https://store.steampowered.com/appreviews/{appid}"
_DETAILS_BASE = "https://store.steampowered.com/api/appdetails"
_RETRY = {"max_attempts": 4, "backoff_factor": 2.0, "backoff_base_seconds": 1.0, "backoff_max_seconds": 30}

# Extra (Steam-specific) columns carried alongside the unified schema.
STEAM_EXTRA_COLUMNS = (
    "received_for_free", "author_num_reviews", "author_num_games_owned",
    "author_playtime_forever", "weighted_vote_score", "votes_funny",
    "comment_count", "early_access",
)


# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------
def reviews_url(appid: int | str, cursor: str = "*", num: int = 100,
                language: str = "english", filter_: str = "recent",
                purchase_type: str = "all") -> str:
    query = urllib.parse.urlencode({
        "json": 1, "num_per_page": num, "language": language,
        "filter": filter_, "purchase_type": purchase_type, "cursor": cursor,
    })
    return f"{_REVIEWS_BASE.format(appid=appid)}?{query}"


def appdetails_url(appid: int | str) -> str:
    return f"{_DETAILS_BASE}?{urllib.parse.urlencode({'appids': appid})}"


# ---------------------------------------------------------------------------
# Transport (reuses the scraper's retry/backoff + Fetcher contract)
# ---------------------------------------------------------------------------
class SteamApiFetcher(scrape.Fetcher):
    """Polite JSON-over-HTTP fetcher; raises the scraper's Transient/Permanent errors."""

    def __init__(self, retry_on_status=(429, 500, 502, 503, 504)) -> None:
        if requests is None:  # pragma: no cover
            raise RuntimeError("requests is required to collect from the Steam API.")
        self.retry_on_status = set(retry_on_status)
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "review-deception-nlp/0.1 (educational research)"})

    def get(self, url: str) -> scrape.FetchResult:
        try:
            r = self.session.get(url, timeout=25)
        except requests.exceptions.RequestException as exc:  # type: ignore[union-attr]
            raise scrape.TransientFetchError(str(exc)) from exc
        if r.status_code in self.retry_on_status:
            raise scrape.TransientFetchError(f"status {r.status_code}")
        if r.status_code >= 400:
            raise scrape.PermanentFetchError(f"status {r.status_code}")
        return scrape.FetchResult(url=url, text=r.text, status=r.status_code)


# ---------------------------------------------------------------------------
# Normalizers (pure)
# ---------------------------------------------------------------------------
def normalize_meta(data: Mapping[str, Any], appid: int | str) -> dict[str, Any]:
    genres = [g.get("description") for g in (data.get("genres") or [])]
    price_cents = (data.get("price_overview") or {}).get("final")
    return {
        "product_id": str(appid),
        "product_title": data.get("name") or "",
        "price": (price_cents / 100.0) if price_cents else None,
        "brand": (data.get("developers") or [None])[0],
        "main_category": genres[0] if genres else "Steam",
    }


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_review(obj: Mapping[str, Any], appid: int | str,
                     meta: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Map a Steam review to the unified schema + Steam-specific extras."""
    meta = meta or {}
    author = obj.get("author") or {}
    text = obj.get("review") or ""
    return {
        "review_id": str(obj.get("recommendationid") or utils.stable_id(appid, author.get("steamid"), text)),
        "product_id": str(appid),
        "product_category": meta.get("main_category") or "Steam",
        "reviewer_id": str(author.get("steamid") or ""),
        # Steam reviews are recommend / not-recommend; binarize to a coarse rating.
        "rating": 5.0 if obj.get("voted_up") else 1.0,
        "review_title": "",
        "review_text": text,
        "verified_purchase": bool(obj.get("steam_purchase")),  # PROXY (analog of Amazon's flag)
        "review_date": fallback_loader.parse_epoch(obj.get("timestamp_created"), "s"),
        "helpful_votes": int(obj.get("votes_up") or 0),
        "total_votes": None,
        "product_title": meta.get("product_title") or "",
        "price": meta.get("price"),
        "brand": meta.get("brand"),
        "main_category": meta.get("main_category"),
        "source": SOURCE,
        # --- Steam extras ---
        "received_for_free": bool(obj.get("received_for_free")),
        "author_num_reviews": int(author.get("num_reviews") or 0),
        "author_num_games_owned": int(author.get("num_games_owned") or 0),
        "author_playtime_forever": int(author.get("playtime_forever") or 0),
        "weighted_vote_score": _f(obj.get("weighted_vote_score")),
        "votes_funny": int(obj.get("votes_funny") or 0),
        "comment_count": int(obj.get("comment_count") or 0),
        "early_access": bool(obj.get("written_during_early_access")),
    }


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------
def _fetch_json(url: str, fetcher, cache, rate_limiter) -> dict[str, Any]:
    """Cache-first fetch (raw JSON persisted before parse) with retry/backoff."""
    text = cache.get(url)
    if text is None:
        text = scrape.fetch_with_retry(fetcher, url, _RETRY, rate_limiter, LOGGER).text
        cache.set(url, text)
    return json.loads(text)


def fetch_app_meta(appid, fetcher, cache, rate_limiter) -> dict[str, Any]:
    try:
        payload = _fetch_json(appdetails_url(appid), fetcher, cache, rate_limiter)
    except scrape.FetchError as exc:
        LOGGER.warning("appdetails failed for %s: %s", appid, exc)
        return {}
    node = payload.get(str(appid), {})
    return normalize_meta(node.get("data", {}), appid) if node.get("success") else {}


def collect_app(appid, max_reviews, fetcher, cache, rate_limiter,
                language="english", filter_="recent", purchase_type="all") -> list[dict[str, Any]]:
    """Paginate one app's reviews via cursor, enriched with the app metadata join."""
    meta = fetch_app_meta(appid, fetcher, cache, rate_limiter)
    records: list[dict[str, Any]] = []
    cursor, seen = "*", set()
    while len(records) < max_reviews:
        url = reviews_url(appid, cursor=cursor, num=min(100, max_reviews - len(records)),
                          language=language, filter_=filter_, purchase_type=purchase_type)
        data = _fetch_json(url, fetcher, cache, rate_limiter)
        if not data.get("success"):
            break
        reviews = data.get("reviews") or []
        if not reviews:
            break
        records.extend(normalize_review(o, appid, meta) for o in reviews)
        cursor = data.get("cursor")
        if not cursor or cursor in seen:
            break
        seen.add(cursor)
    return records[:max_reviews]


def collect(config: Mapping, appids=None, max_reviews_per_app=None) -> list[dict[str, Any]]:
    """Collect reviews across the configured apps (cache-first; checkpointed)."""
    sc = config["steam"]
    appids = appids or sc["appids"]
    mpa = max_reviews_per_app or sc["max_reviews_per_app"]
    root = utils.project_root()
    cache = scrape.DiskCache(root / config["paths"]["cache"] / "steam", ttl_days=30, enabled=True)
    rate_limiter = scrape.RateLimiter(sc.get("min_delay_seconds", 1.0), sc.get("max_delay_seconds", 2.5), True)
    checkpoint = scrape.CheckpointManager(
        root / config["paths"]["checkpoints"] / "steam_checkpoint.json", enabled=True)
    fetcher = SteamApiFetcher()

    records: list[dict[str, Any]] = []
    for appid in appids:
        recs = collect_app(appid, mpa, fetcher, cache, rate_limiter,
                           language=sc.get("language", "english"),
                           filter_=sc.get("filter", "recent"),
                           purchase_type=sc.get("purchase_type", "all"))
        records.extend(recs)
        checkpoint.mark_done(str(appid), len(recs))
        checkpoint.save()
        LOGGER.info("collected app %s: %d reviews (steam_purchase=%.2f, free=%.3f)", appid, len(recs),
                    sum(r["verified_purchase"] for r in recs) / max(len(recs), 1),
                    sum(r["received_for_free"] for r in recs) / max(len(recs), 1))
    return records


def collect_and_save(config: Mapping, **kwargs) -> pd.DataFrame:
    """Collect, build a DataFrame, and persist to ``data/interim/steam_reviews.pkl``."""
    records = collect(config, **kwargs)
    df = pd.DataFrame(records)[list(REVIEW_COLUMNS) + list(STEAM_EXTRA_COLUMNS)]
    out = utils.project_root() / config["paths"]["data_interim"] / "steam_reviews.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out)
    LOGGER.info("saved %d Steam reviews -> %s", len(df), out)
    return df


def load_steam(config: Mapping) -> pd.DataFrame:
    """Load the previously collected Steam table (raises if not collected yet)."""
    path = utils.project_root() / config["paths"]["data_interim"] / "steam_reviews.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.steam_collector --collect` first.")
    return pd.read_pickle(path)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Collect Steam reviews (public API).")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--max", type=int, default=None, help="max reviews per app")
    parser.add_argument("--appids", type=int, nargs="*", default=None)
    args = parser.parse_args()
    config = utils.load_config()
    utils.set_seeds(config["project"]["random_seed"])
    if not args.collect:
        print("Pass --collect to fetch. Configured apps:", config["steam"]["appids"])
        return
    df = collect_and_save(config, appids=args.appids, max_reviews_per_app=args.max)
    print(f"\ncollected {len(df)} reviews across {df['product_id'].nunique()} apps")
    print("steam_purchase (verified) rate:", round((df["verified_purchase"] == True).mean(), 3))
    print("received_for_free rate:", round(df["received_for_free"].mean(), 3))
    print(df[["product_title", "rating", "verified_purchase", "received_for_free",
              "helpful_votes", "author_num_reviews"]].head().to_string(index=False))


if __name__ == "__main__":
    _main()
