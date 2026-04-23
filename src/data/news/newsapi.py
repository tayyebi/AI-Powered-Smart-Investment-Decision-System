"""
src/data/news/newsapi.py
------------------------
NewsAPI.org source – requires a free API key (NEWS_API_KEY in .env).

Sign up at https://newsapi.org to get a free developer key.
"""

import os

import requests
from dotenv import load_dotenv

from .base import NewsSource

load_dotenv()

_ENDPOINT = "https://newsapi.org/v2/everything"


class NewsApiSource(NewsSource):
    """Fetches articles from NewsAPI.org.

    The ``NEWS_API_KEY`` environment variable must be set; if it is absent
    this source silently returns an empty list.
    """

    @property
    def name(self) -> str:
        return "NewsAPI"

    def fetch(self, query: str, max_results: int = 10) -> list[dict]:
        """Return up to *max_results* articles from NewsAPI.

        Args:
            query:       Search query, e.g. ``"TSLA stock"``.
            max_results: Cap on returned articles (max 100 per API docs).

        Returns:
            List of ``{"title": str, "content": str}`` dicts.
            Empty list when the API key is missing or the request fails.
        """
        api_key = os.getenv("NEWS_API_KEY")
        if not api_key:
            return []

        params = {
            "q":        query,
            "language": "en",
            "sortBy":   "publishedAt",
            "pageSize": min(max_results, 100),
        }
        headers = {"X-Api-Key": api_key}
        try:
            resp = requests.get(_ENDPOINT, params=params, headers=headers, timeout=15)
            resp.raise_for_status()
            return [
                {
                    "title":   art.get("title", ""),
                    "content": art.get("content", "") or art.get("description", ""),
                }
                for art in resp.json().get("articles", [])
                if art.get("title")
            ]
        except Exception as exc:
            print(f"[{self.name}] Fetch failed: {exc}")
            return []
