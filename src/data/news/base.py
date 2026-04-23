"""
src/data/news/base.py
---------------------
Abstract base class that every news source must implement.
"""

from abc import ABC, abstractmethod


class NewsSource(ABC):
    """A single news provider that can fetch articles for a query string."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name shown in logs (e.g. ``"Google News"``)."""

    @abstractmethod
    def fetch(self, query: str, max_results: int = 10) -> list[dict]:
        """Fetch news articles matching *query*.

        Args:
            query:       Search query, e.g. ``"TSLA stock"``.
            max_results: Maximum number of articles to return.

        Returns:
            List of dicts with at minimum the keys:
              - ``"title"``   (str) – article headline
              - ``"content"`` (str) – body text or description (may be empty)
        """
