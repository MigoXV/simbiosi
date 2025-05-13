import logging
import os
import sys

# We need to setup root logger before importing any fairseq libraries.
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logging.getLogger("fairseq.tasks.text_to_speech").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

try:
    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv())
    logger.debug("Environment variables are loaded from .env file.")
except ImportError:
    logger.debug(
        "python-dotenv is not installed, so environment variables are not loaded from .env file."
    )
