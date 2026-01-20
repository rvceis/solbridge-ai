# ML Service package initialization
__version__ = "1.0.0"
__author__ = "Solar Energy Team"
__description__ = "AI/ML Service for Community Energy Sharing Platform"

from src.config import get_settings
from src.utils import get_logger

logger = get_logger(__name__)
logger.info(f"Initializing {__description__} v{__version__}")
