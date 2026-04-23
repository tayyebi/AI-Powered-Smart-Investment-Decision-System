"""
src/data/news/google_news.py
----------------------------
Google News RSS feed source – no API key required.

Uses the public Google News RSS endpoint:
  https://news.google.com/rss/search?q=<query>&hl=en&gl=US&ceid=US:en

Parsed with `feedparser`.
"""

import html
import re

import feedparser

from .base import NewsSource

_RSS_URL = "https://news.google.com/rss/search?q={query}&hl=en&gl=US&ceid=US:en"


def _strip_tags(text: str) -> str:
    """Remove HTML tags and decode HTML entities."""
    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(text).strip()


class GoogleNewsSource(NewsSource):
    """Fetches articles from the Google News RSS feed (no API key required)."""

    @property
    def name(self) -> str:
        return "Google News"

    def fetch(self, query: str, max_results: int = 10) -> list[dict]:
        """Return up to *max_results* articles from Google News RSS.

        Args:
            query:       Search query, e.g. ``"TSLA stock"``.
            max_results: Cap on returned articles.

        Returns:
            List of ``{"title": str, "content": str}`` dicts.
        """
        url = _RSS_URL.format(query=query.replace(" ", "+"))
        try:
            feed = feedparser.parse(url)
            articles = []
            for entry in feed.entries[:max_results]:
                title   = _strip_tags(entry.get("title", ""))
                content = _strip_tags(entry.get("summary", "") or entry.get("description", ""))
                if title:
                    articles.append({"title": title, "content": content})
            return articles
        except Exception as exc:
            print(f"[{self.name}] Fetch failed: {exc}")
            return []
