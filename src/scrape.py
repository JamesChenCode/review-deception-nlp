"""Polite, robust Amazon review scraper — the PRIMARY data-collection path.

Architecture
------------
``Fetcher``            : pluggable transport.
    - ``HttpFetcher``  : live HTTP via ``requests`` (or stdlib ``urllib`` fallback),
                         with user-agent rotation.
    - ``FixtureFetcher``: offline transport that serves local fixture files, so the
                         entire pipeline (cache, checkpoint, retry, parse) runs with
                         NO network — used for the dry run and Step-3 tests.
``RateLimiter``        : randomized inter-request delays (politeness).
``fetch_with_retry``   : manual exponential backoff + jitter on transient failures.
``DiskCache``          : gzipped on-disk cache; RAW responses are persisted BEFORE
                         parsing (cache-first; never re-fetch what we already have).
``CheckpointManager``  : resume an interrupted run, skipping completed products.
``ReviewParser``       : raw HTML -> unified-schema review dicts (JSON-island path
                         works without bs4; bs4 selector path used for live Amazon).
``AmazonReviewScraper``: orchestration + GRACEFUL DEGRADATION to the fallback
                         dataset when scraping repeatedly fails.

Label & ethics
--------------
The ``verified_purchase`` flag captured here is a WEAK PROXY for trustworthiness
(it measures purchase verification, not deception). Amazon's ToS restricts
scraping and it actively blocks scrapers; this module is rate-limited and
cache-first by design (see README "Ethics & Legality"). Live scraping is gated:
the CLI refuses to run live without an explicit authorization flag; the offline
``--dry-run`` is the default.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib import robotparser

from . import fallback_loader, utils

# --- optional deps (degrade gracefully) ------------------------------------
try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover
    requests = None

try:
    from bs4 import BeautifulSoup  # type: ignore
except ImportError:  # pragma: no cover
    BeautifulSoup = None

LOGGER = utils.get_logger("review_deception.scrape")

SOURCE = "scrape"
_JSON_ISLAND = re.compile(
    r'<script[^>]*id="(?P<id>[\w-]+)"[^>]*type="application/json"[^>]*>(?P<body>.*?)</script>',
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Errors + result type
# ---------------------------------------------------------------------------
class FetchError(Exception):
    """Base fetch error."""


class TransientFetchError(FetchError):
    """Retryable (timeout, 429, 5xx, connection reset)."""


class PermanentFetchError(FetchError):
    """Non-retryable (404, 403 block, malformed)."""


@dataclass
class FetchResult:
    url: str
    text: str
    status: int = 200
    from_cache: bool = False
    attempts: int = 1


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------
def review_page_url(base_url: str, product_id: str, page: int = 1) -> str:
    return f"{base_url.rstrip('/')}/product-reviews/{product_id}/?pageNumber={page}"


def product_page_url(base_url: str, product_id: str) -> str:
    return f"{base_url.rstrip('/')}/dp/{product_id}"


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
class RateLimiter:
    """Sleep a randomized delay between requests to stay polite."""

    def __init__(self, min_delay: float, max_delay: float, jitter: bool = True) -> None:
        self.min_delay = float(min_delay)
        self.max_delay = float(max_delay)
        self.jitter = jitter

    def sleep(self) -> float:
        delay = random.uniform(self.min_delay, self.max_delay) if self.jitter else self.min_delay
        if delay > 0:
            time.sleep(delay)
        return delay


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------
class Fetcher:
    """Transport interface: ``get(url) -> FetchResult`` or raise a FetchError."""

    def get(self, url: str) -> FetchResult:  # pragma: no cover - interface
        raise NotImplementedError


class HttpFetcher(Fetcher):
    """Live HTTP transport with user-agent rotation (requests, else urllib)."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        sc = config["scraper"]
        self.user_agents: list[str] = list(sc["user_agents"])
        self.retry_on_status = set(sc["retry"]["retry_on_status"])
        self.timeout = 15
        self._session = requests.Session() if requests is not None else None

    def _headers(self) -> dict[str, str]:
        return {
            "User-Agent": random.choice(self.user_agents),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml",
        }

    def get(self, url: str) -> FetchResult:
        if self._session is not None:
            return self._get_requests(url)
        return self._get_urllib(url)

    def _get_requests(self, url: str) -> FetchResult:
        try:
            resp = self._session.get(url, headers=self._headers(), timeout=self.timeout)
        except requests.exceptions.RequestException as exc:  # type: ignore[union-attr]
            raise TransientFetchError(str(exc)) from exc
        if resp.status_code in self.retry_on_status:
            raise TransientFetchError(f"retryable status {resp.status_code}")
        if resp.status_code >= 400:
            raise PermanentFetchError(f"status {resp.status_code}")
        return FetchResult(url=url, text=resp.text, status=resp.status_code)

    def _get_urllib(self, url: str) -> FetchResult:
        req = urllib.request.Request(url, headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                charset = resp.headers.get_content_charset() or "utf-8"
                return FetchResult(url=url, text=resp.read().decode(charset, "replace"), status=resp.status)
        except urllib.error.HTTPError as exc:
            if exc.code in self.retry_on_status:
                raise TransientFetchError(f"retryable status {exc.code}") from exc
            raise PermanentFetchError(f"status {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise TransientFetchError(str(exc)) from exc


class FixtureFetcher(Fetcher):
    """Offline transport: serve local fixture files; optionally simulate failures.

    ``fail_once`` is a set of URLs that raise :class:`TransientFetchError` on their
    FIRST request only, exercising the retry/backoff path deterministically.
    ``always_fail`` forces every request to fail (to demo graceful fallback).
    """

    def __init__(
        self,
        manifest: Mapping[str, Path],
        fail_once: Iterable[str] = (),
        always_fail: bool = False,
    ) -> None:
        self.manifest = dict(manifest)
        self._fail_once = set(fail_once)
        self._already_failed: set[str] = set()
        self.always_fail = always_fail

    def get(self, url: str) -> FetchResult:
        if self.always_fail:
            raise TransientFetchError(f"simulated outage for {url}")
        if url in self._fail_once and url not in self._already_failed:
            self._already_failed.add(url)
            raise TransientFetchError(f"simulated transient failure for {url}")
        path = self.manifest.get(url)
        if path is None:
            raise PermanentFetchError(f"no fixture for {url}")
        return FetchResult(url=url, text=Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Retry with exponential backoff (manual; no third-party dependency)
# ---------------------------------------------------------------------------
def fetch_with_retry(
    fetcher: Fetcher,
    url: str,
    retry_cfg: Mapping[str, Any],
    rate_limiter: RateLimiter,
    logger=LOGGER,
) -> FetchResult:
    """Fetch ``url``, retrying transient failures with capped exponential backoff."""
    max_attempts = int(retry_cfg.get("max_attempts", 4))
    factor = float(retry_cfg.get("backoff_factor", 2.0))
    base = float(retry_cfg.get("backoff_base_seconds", 1.0))
    cap = float(retry_cfg.get("backoff_max_seconds", 60))
    attempt = 0
    while True:
        attempt += 1
        rate_limiter.sleep()  # politeness delay before every request
        try:
            result = fetcher.get(url)
            result.attempts = attempt
            return result
        except TransientFetchError as exc:
            if attempt >= max_attempts:
                logger.warning("Giving up on %s after %d attempts: %s", url, attempt, exc)
                raise
            backoff = min(cap, base * (factor ** (attempt - 1)))
            backoff += random.uniform(0, base)  # jitter
            logger.info("Transient failure (attempt %d/%d) for %s; backing off %.2fs: %s",
                        attempt, max_attempts, url, backoff, exc)
            time.sleep(backoff)


# ---------------------------------------------------------------------------
# On-disk cache (gzip). Raw responses are persisted BEFORE parsing.
# ---------------------------------------------------------------------------
class DiskCache:
    def __init__(self, cache_dir: str | Path, ttl_days: float = 30, enabled: bool = True) -> None:
        self.dir = Path(cache_dir)
        self.ttl_seconds = ttl_days * 86_400
        self.enabled = enabled
        if enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, url: str) -> Path:
        return self.dir / f"{utils.stable_id(url, length=24)}.html.gz"

    def get(self, url: str) -> str | None:
        if not self.enabled:
            return None
        path = self._path(url)
        if not path.is_file():
            return None
        if self.ttl_seconds and (time.time() - path.stat().st_mtime) > self.ttl_seconds:
            return None
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return fh.read()

    def set(self, url: str, text: str, status: int = 200) -> None:
        if not self.enabled:
            return
        path = self._path(url)
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(text)
        # human-readable sidecar so a cache dir is auditable (which url, when, status)
        path.with_suffix(".meta.json").write_text(
            json.dumps({"url": url, "status": status, "fetched_at": datetime.now(timezone.utc).isoformat()}),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
class CheckpointManager:
    def __init__(self, path: str | Path, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self.done_products: set[str] = set()
        self.review_count = 0
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.load()

    def load(self) -> None:
        if self.path.is_file():
            state = json.loads(self.path.read_text(encoding="utf-8"))
            self.done_products = set(state.get("done_products", []))
            self.review_count = int(state.get("review_count", 0))
            LOGGER.info("Loaded checkpoint: %d products done, %d reviews so far",
                        len(self.done_products), self.review_count)

    def is_done(self, product_id: str) -> bool:
        return product_id in self.done_products

    def mark_done(self, product_id: str, n_reviews: int) -> None:
        self.done_products.add(product_id)
        self.review_count += n_reviews

    def save(self) -> None:
        if not self.enabled:
            return
        self.path.write_text(json.dumps({
            "done_products": sorted(self.done_products),
            "review_count": self.review_count,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
class ReviewParser:
    """Raw HTML -> unified-schema review dicts (source='scrape')."""

    def parse_reviews(self, html: str, product_id: str, category: str) -> list[dict[str, Any]]:
        raw = self._json_island(html, "reviews-data")
        if raw is not None:
            return [self._normalize(o, product_id, category) for o in raw]
        if BeautifulSoup is not None:
            return self._parse_bs4(html, product_id, category)
        LOGGER.warning("No JSON island and bs4 not installed; cannot parse %s", product_id)
        return []

    def parse_product_meta(self, html: str, product_id: str) -> dict[str, Any]:
        raw = self._json_island(html, "product-meta")
        if isinstance(raw, dict):
            return {
                "product_id": product_id,
                "product_title": raw.get("title") or "",
                "price": fallback_loader.parse_price(raw.get("price")),
                "brand": raw.get("brand") or raw.get("store"),
                "main_category": raw.get("main_category"),
            }
        return {"product_id": product_id, "product_title": "", "price": None, "brand": None, "main_category": None}

    @staticmethod
    def _json_island(html: str, island_id: str) -> Any | None:
        for match in _JSON_ISLAND.finditer(html):
            if match.group("id") == island_id:
                return json.loads(match.group("body"))
        return None

    @staticmethod
    def _normalize(obj: Mapping[str, Any], product_id: str, category: str) -> dict[str, Any]:
        text = obj.get("text") or ""
        reviewer_id = obj.get("user_id") or ""
        date_raw = obj.get("date")
        review_date = None
        if date_raw:
            try:
                review_date = datetime.fromisoformat(str(date_raw)).replace(tzinfo=timezone.utc)
            except ValueError:
                review_date = None
        return {
            "review_id": obj.get("id") or utils.stable_id(product_id, reviewer_id, text),
            "product_id": product_id,
            "product_category": category,
            "reviewer_id": str(reviewer_id),
            "rating": float(obj["rating"]) if obj.get("rating") is not None else None,
            "review_title": obj.get("title") or "",
            "review_text": text,
            "verified_purchase": _coerce_bool(obj.get("verified_purchase")),
            "review_date": review_date,
            "helpful_votes": int(obj.get("helpful_vote") or 0),
            "total_votes": None,
            "source": SOURCE,
        }

    def _parse_bs4(self, html: str, product_id: str, category: str) -> list[dict[str, Any]]:
        """Best-effort selector parse for live Amazon HTML (used when bs4 present)."""
        soup = BeautifulSoup(html, "html.parser")
        rows: list[dict[str, Any]] = []
        for block in soup.select('[data-hook="review"]'):
            rating_el = block.select_one('[data-hook="review-star-rating"], [data-hook="cmps-review-star-rating"]')
            rating = None
            if rating_el and rating_el.get_text():
                m = re.search(r"([0-5](?:\.\d)?)", rating_el.get_text())
                rating = float(m.group(1)) if m else None
            text_el = block.select_one('[data-hook="review-body"]')
            title_el = block.select_one('[data-hook="review-title"]')
            verified = block.select_one('[data-hook="avp-badge"]') is not None
            author = block.select_one(".a-profile-name")
            rows.append(self._normalize({
                "rating": rating,
                "title": title_el.get_text(strip=True) if title_el else "",
                "text": text_el.get_text(strip=True) if text_el else "",
                "verified_purchase": verified,
                "user_id": author.get_text(strip=True) if author else "",
            }, product_id, category))
        return rows


def _coerce_bool(raw: Any) -> bool | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"true", "1", "yes"}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class AmazonReviewScraper:
    """Drive collection across categories/products with politeness + resumability."""

    def __init__(self, config: Mapping[str, Any], fetcher: Fetcher | None = None) -> None:
        self.config = config
        sc = config["scraper"]
        rl = sc["rate_limit"]
        self.base_url = sc["base_url"]
        self.retry_cfg = dict(sc["retry"])
        self.max_reviews = int(sc["max_reviews_per_product"])
        self.rate_limiter = RateLimiter(rl["min_delay_seconds"], rl["max_delay_seconds"], rl["jitter"])
        self.fetcher = fetcher or HttpFetcher(config)
        self.offline = isinstance(self.fetcher, FixtureFetcher)
        root = utils.project_root()
        self.cache = DiskCache(root / config["paths"]["cache"],
                               ttl_days=sc["caching"]["ttl_days"], enabled=sc["caching"]["enabled"])
        self.checkpoint = CheckpointManager(
            root / config["paths"]["checkpoints"] / "scrape_checkpoint.json",
            enabled=sc["checkpointing"]["enabled"])
        self.parser = ReviewParser()
        self._robots: robotparser.RobotFileParser | None = None
        # Optional override for where _run_fallback reads from (used by the dry run
        # so it loads a scratch sample instead of real data/raw). None -> data/raw.
        self.fallback_raw_dir: str | Path | None = None

    # -- fetch (cache-first) ------------------------------------------------
    def _fetch(self, url: str) -> str:
        cached = self.cache.get(url)
        if cached is not None:
            LOGGER.debug("cache hit %s", url)
            return cached
        if not self._robots_ok(url):
            raise PermanentFetchError(f"blocked by robots.txt: {url}")
        result = fetch_with_retry(self.fetcher, url, self.retry_cfg, self.rate_limiter)
        self.cache.set(url, result.text, status=result.status)  # persist RAW before parse
        return result.text

    def _robots_ok(self, url: str) -> bool:
        if self.offline or not self.config["scraper"].get("respect_robots_txt", True):
            return True
        if self._robots is None:
            self._robots = robotparser.RobotFileParser()
            self._robots.set_url(f"{self.base_url.rstrip('/')}/robots.txt")
            try:
                self._robots.read()
            except Exception as exc:  # network/parse issues -> fail open but log
                LOGGER.warning("Could not read robots.txt (%s); proceeding cautiously", exc)
                return True
        return self._robots.can_fetch("*", url)

    # -- per product --------------------------------------------------------
    def scrape_product(self, product_id: str, category: str) -> list[dict[str, Any]]:
        """Fetch + parse reviews for one product, enriched with product metadata."""
        reviews: list[dict[str, Any]] = []
        page = 1
        while len(reviews) < self.max_reviews:
            url = review_page_url(self.base_url, product_id, page)
            html = self._fetch(url)
            page_reviews = self.parser.parse_reviews(html, product_id, category)
            if not page_reviews:
                break
            reviews.extend(page_reviews)
            page += 1
            if self.offline:  # fixtures are single-page
                break
        # JOIN with product metadata (second source/endpoint -> multi-source data)
        try:
            meta_html = self._fetch(product_page_url(self.base_url, product_id))
            meta = self.parser.parse_product_meta(meta_html, product_id)
        except FetchError as exc:
            LOGGER.warning("No metadata for %s: %s", product_id, exc)
            meta = {"product_title": "", "price": None, "brand": None, "main_category": None}
        for r in reviews:
            r.update({k: meta.get(k) for k in ("product_title", "price", "brand", "main_category")})
        return reviews[: self.max_reviews]

    def discover_products(self, category: str) -> list[str]:
        """Find product ids for a category (live: search page). Offline supplies ids."""
        if self.offline:
            return []  # dry run injects products explicitly via run(products_by_category=...)
        url = f"{self.base_url.rstrip('/')}/s?k={category}"
        try:
            html = self._fetch(url)
        except FetchError as exc:
            LOGGER.warning("discovery failed for %r: %s", category, exc)
            return []
        return list(dict.fromkeys(re.findall(r"/dp/([A-Z0-9]{10})", html)))

    # -- run ----------------------------------------------------------------
    def run(
        self,
        categories: Iterable[str] | None = None,
        max_products: int | None = None,
        products_by_category: Mapping[str, list[str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Collect reviews; degrade to the fallback dataset on repeated failure."""
        utils.set_seeds(self.config["project"]["random_seed"])
        utils.ensure_dirs(self.config)
        cats = list(categories or self.config["scraper"]["target_categories"])
        per_cat_cap = max_products or self.config["scraper"]["max_products_per_category"]
        flush_every = self.config["scraper"]["checkpointing"]["flush_every"]
        max_failures = self.config["data_source"]["max_failures"]
        auto_fallback = self.config["data_source"]["auto_fallback"]

        all_reviews: list[dict[str, Any]] = []
        consecutive_failures = 0
        processed = 0
        for category in cats:
            products = (products_by_category or {}).get(category) or self.discover_products(category)
            for product_id in products[:per_cat_cap]:
                if self.checkpoint.is_done(product_id):
                    LOGGER.info("skip (checkpointed): %s", product_id)
                    continue
                try:
                    product_reviews = self.scrape_product(product_id, category)
                    all_reviews.extend(product_reviews)
                    self.checkpoint.mark_done(product_id, len(product_reviews))
                    consecutive_failures = 0
                    LOGGER.info("scraped %s: %d reviews", product_id, len(product_reviews))
                except FetchError as exc:
                    consecutive_failures += 1
                    LOGGER.warning("product %s failed (%d/%d): %s",
                                   product_id, consecutive_failures, max_failures, exc)
                    if consecutive_failures >= max_failures and auto_fallback:
                        LOGGER.error("Too many failures; DEGRADING to fallback dataset.")
                        self.checkpoint.save()
                        return self._run_fallback()
                processed += 1
                if processed % flush_every == 0:
                    self.checkpoint.save()
        self.checkpoint.save()
        return all_reviews

    def _run_fallback(self) -> list[dict[str, Any]]:
        """Graceful degradation: load the same unified schema from the fallback set."""
        rows = fallback_loader.load_all(self.config, raw_dir=self.fallback_raw_dir)
        LOGGER.info("Fallback produced %d reviews via %s", len(rows), fallback_loader.SOURCE)
        return rows


# ---------------------------------------------------------------------------
# Fixture generation (dry run + tests)
# ---------------------------------------------------------------------------
def build_fixture_pages(
    out_dir: str | Path,
    base_url: str = "https://www.amazon.com",
    n_products: int = 5,
    reviews_per_product: int = 6,
    seed: int = 42,
    category: str = "electronics",
) -> tuple[dict[str, Path], list[str]]:
    """Write HTML fixtures (review + product pages w/ JSON islands). No network."""
    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Path] = {}
    product_ids: list[str] = []
    generic = "Great product works as expected highly recommend to everyone"

    for p in range(n_products):
        asin = f"DRYP{p:06d}"[:10]
        product_ids.append(asin)
        suspicious = p == 0
        reviews = []
        for j in range(reviews_per_product):
            if suspicious:  # burst of generic 5-star UNVERIFIED reviews
                rating, verified, text = 5.0, False, (generic if j % 2 == 0 else generic + " love it")
                user = f"S{j:03d}"
            else:
                rating = float(rng.choice([1, 3, 4, 5, 5]))
                verified = rng.random() < 0.7
                text = rng.choice([
                    "Solid build quality and the battery lasts all day during travel.",
                    "Stopped working after two weeks, disappointed with durability.",
                    "Exactly what I needed, the detailed setup guide made install painless.",
                    generic,
                ])
                user = f"U{rng.randint(0, 40):03d}"
            reviews.append({
                "id": f"{asin}-r{j}",
                "user_id": user,
                "rating": rating,
                "title": "Five Stars" if rating >= 4 else "Not great",
                "text": text,
                "verified_purchase": verified,
                "date": f"2023-01-{(j % 27) + 1:02d}",
                "helpful_vote": rng.randint(0, 25),
            })
        meta = {"title": f"Electronics Item {p}", "price": round(rng.uniform(8, 120), 2),
                "store": rng.choice(["Acme", "Globex", "Initech"]), "main_category": "Electronics"}

        review_file = out / f"reviews_{asin}.html"
        product_file = out / f"product_{asin}.html"
        review_file.write_text(_html_page(f"reviews-data", reviews, f"Reviews for {asin}"), encoding="utf-8")
        product_file.write_text(_html_page("product-meta", meta, f"Product {asin}"), encoding="utf-8")
        manifest[review_page_url(base_url, asin, 1)] = review_file
        manifest[product_page_url(base_url, asin)] = product_file
    return manifest, product_ids


def _html_page(island_id: str, payload: Any, title: str) -> str:
    return (
        f"<!doctype html><html><head><title>{title}</title></head><body>"
        f'<h1>{title}</h1>'
        f'<script id="{island_id}" type="application/json">{json.dumps(payload)}</script>'
        f"</body></html>"
    )


# ---------------------------------------------------------------------------
# Dry run (offline, ≤5 products, no installs, no network)
# ---------------------------------------------------------------------------
def run_dry_run(n_products: int = 5) -> None:
    config = utils.load_config()
    root = utils.project_root()
    # All dry-run artifacts live under ONE scratch dir we fully own, so the demo
    # is reproducible and never touches real collection state or downloaded data.
    scratch = root / "data/raw/_dryrun"
    if scratch.exists():
        shutil.rmtree(scratch)
    # Isolated scratch cache/checkpoint paths + fast, near-instant polite timings.
    config = utils.deep_merge(config, {
        "paths": {"cache": "data/raw/_dryrun/cache", "checkpoints": "data/raw/_dryrun/checkpoints"},
        "scraper": {
            "rate_limit": {"min_delay_seconds": 0.0, "max_delay_seconds": 0.02, "jitter": True},
            "retry": {"backoff_base_seconds": 0.01},
        },
    })
    utils.set_seeds(config["project"]["random_seed"])
    fixtures_dir = scratch / "fixtures"
    manifest, product_ids = build_fixture_pages(
        fixtures_dir, base_url=config["scraper"]["base_url"], n_products=n_products)

    print("\n========== DRY RUN 1: scrape ≤5 products (offline fixtures) ==========")
    # Make the first product's review page fail ONCE to exercise retry/backoff.
    first_review_url = review_page_url(config["scraper"]["base_url"], product_ids[0], 1)
    fetcher = FixtureFetcher(manifest, fail_once=[first_review_url])
    scraper = AmazonReviewScraper(config, fetcher=fetcher)
    reviews = scraper.run(categories=["electronics"],
                          products_by_category={"electronics": product_ids})
    print(f"\nCollected {len(reviews)} reviews from {len(product_ids)} products.")
    print("\nUnified-schema preview (multi-source: review fields + joined product meta):\n")
    print(utils.preview_table(reviews, columns=[
        "review_id", "product_id", "reviewer_id", "rating",
        "verified_purchase", "helpful_votes", "price", "review_text",
    ], n=6))

    cache_dir = root / config["paths"]["cache"]
    ckpt = root / config["paths"]["checkpoints"] / "scrape_checkpoint.json"
    n_cached = len(list(cache_dir.glob("*.html.gz")))
    verified = sum(1 for r in reviews if r["verified_purchase"])
    print(f"\nRaw responses persisted (gzipped) in cache: {n_cached} files -> {cache_dir}")
    print(f"Checkpoint written: {ckpt} (exists={ckpt.is_file()})")
    print(f"Proxy-label split: verified={verified}  unverified={len(reviews) - verified}  "
          f"(weak proxy — purchase verification, NOT deception)")

    print("\n========== DRY RUN 2: re-run shows checkpoint RESUME (skips all) ==========")
    scraper2 = AmazonReviewScraper(config, fetcher=FixtureFetcher(manifest))
    reviews2 = scraper2.run(categories=["electronics"],
                            products_by_category={"electronics": product_ids})
    print(f"Re-run collected {len(reviews2)} NEW reviews (expected 0 — all checkpointed).")

    print("\n========== DRY RUN 3: GRACEFUL DEGRADATION to fallback dataset ==========")
    # Seed a tiny fallback sample (its own scratch dir), then force the scraper to
    # fail and fall back to it.
    fb_dir = scratch / "fallback"
    fallback_loader.make_synthetic_sample(fb_dir, categories=tuple(config["fallback"]["categories"]),
                                           n_products=3, reviews_per_product=5,
                                           seed=config["project"]["random_seed"])
    # Disable the cache here so the always-fail fetcher is genuinely exercised
    # (otherwise warm cache entries from Dry Run 1 would serve and never fail),
    # and lower max_failures so degradation triggers mid-stream.
    failing_config = utils.deep_merge(config, {
        "data_source": {"max_failures": 3},
        "scraper": {"checkpointing": {"enabled": False}, "caching": {"enabled": False}},
    })
    failing = AmazonReviewScraper(failing_config, fetcher=FixtureFetcher(manifest, always_fail=True))
    failing.fallback_raw_dir = fb_dir
    fb_reviews = failing.run(categories=["electronics"],
                             products_by_category={"electronics": product_ids})
    source = fb_reviews[0]["source"] if fb_reviews else "n/a"
    print(f"After simulated block, fallback produced {len(fb_reviews)} reviews "
          f"(source='{source}') via the SAME unified schema.")
    print("\nDry run complete. No Amazon servers were contacted; no packages installed.\n")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Amazon review scraper.")
    parser.add_argument("--dry-run", action="store_true", help="offline demo on ≤5 fixture products")
    parser.add_argument("--n-products", type=int, default=5)
    parser.add_argument("--i-have-authorization", action="store_true",
                        help="REQUIRED to attempt any live scraping (see README ethics).")
    args = parser.parse_args()

    if args.dry_run or not args.i_have_authorization:
        if not args.dry_run:
            LOGGER.warning("Live scraping is gated. Running the offline --dry-run instead. "
                           "Pass --i-have-authorization to enable live mode.")
        run_dry_run(n_products=args.n_products)
        return
    # Live path (gated behind explicit flag).
    config = utils.load_config()
    scraper = AmazonReviewScraper(config)
    reviews = scraper.run()
    LOGGER.info("Live run collected %d reviews.", len(reviews))


if __name__ == "__main__":
    _main()
