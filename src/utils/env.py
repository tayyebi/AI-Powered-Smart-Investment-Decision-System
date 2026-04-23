"""
src/utils/env.py
----------------
Environment setup: .env loading and safe variable access.
"""

import logging
import os

from dotenv import load_dotenv


def setup_environment() -> None:
    """Load .env file and configure basic logging."""
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def get_env_variable(key: str, default=None):
    """Return the value of environment variable *key*, or *default*."""
    return os.getenv(key, default)
