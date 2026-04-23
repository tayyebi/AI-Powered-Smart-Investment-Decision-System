import os
import logging
from dotenv import load_dotenv

def setup_environment():
    """Load environment variables and setup basic logging."""
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def get_env_variable(key: str, default=None):
    """Safely get an environment variable."""
    return os.getenv(key, default)
