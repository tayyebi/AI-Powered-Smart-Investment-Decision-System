"""
src/data/news/__init__.py
--------------------------
News aggregator – combines results from all registered sources.

Built-in sources (registered by default):
  1. GoogleNewsSource  – no API key required
  2. NewsApiSource     – requires NEWS_API_KEY in .env (optional)

Adding a new source
-------------------
Create a class in a new module under ``src/data/news/`` that inherits from
:class:`~src.data.news.base.NewsSource` and implements ``name`` and
``fetch()``, then register it in ``DEFAULT_SOURCES`` below.

Example::

    from .my_source import MySource
    DEFAULT_SOURCES = [GoogleNewsSource(), NewsApiSource(), MySource()]
"""

from .base import NewsSource
from .google_news import GoogleNewsSource
from .newsapi import NewsApiSource

# ── Default source registry ──────────────────────────────────────────────────
# Sources are queried in order; duplicates are removed by title.
DEFAULT_SOURCES: list[NewsSource] = [
    GoogleNewsSource(),
    NewsApiSource(),
]


def fetch_news(
    query: str,
    max_results: int = 10,
    sources: list[NewsSource] | None = None,
) -> list[dict]:
    """Aggregate news articles from all active sources.

    Args:
        query:       Search query, e.g. ``"TSLA stock"``.
        max_results: Max articles to collect *per source*.
        sources:     Override the default source list.  Pass an explicit list
                     to use only specific providers.

    Returns:
        Deduplicated list of ``{"title": str, "content": str}`` dicts,
        ordered by source priority (first source's articles come first).
    """
    active_sources = sources if sources is not None else DEFAULT_SOURCES
    seen_titles: set[str] = set()
    aggregated: list[dict] = []

    for source in active_sources:
        try:
            articles = source.fetch(query, max_results=max_results)
            for art in articles:
                title = art.get("title", "").strip()
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    aggregated.append(art)
        except Exception as exc:
            print(f"[news] Source '{source.name}' raised an error: {exc}")

    return aggregated


__all__ = [
    "NewsSource",
    "GoogleNewsSource",
    "NewsApiSource",
    "DEFAULT_SOURCES",
    "fetch_news",
]
