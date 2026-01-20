import logging
import logging.handlers
import json
from pathlib import Path
from pythonjsonlogger import jsonlogger
from datetime import datetime
from src.config.settings import get_settings

settings = get_settings()


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional context"""
    
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['service'] = settings.SERVICE_NAME
        log_record['version'] = settings.SERVICE_VERSION
        log_record['environment'] = settings.ENVIRONMENT
        log_record['level'] = record.levelname
        log_record['logger'] = record.name


class LoggerFactory:
    """Factory for creating configured loggers"""
    
    _loggers = {}
    
    @staticmethod
    def setup_logging():
        """Initialize logging configuration"""
        # Create logs directory
        Path(settings.LOG_DIR).mkdir(parents=True, exist_ok=True)
        
        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(settings.LOG_LEVEL)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # File handler (rotating)
        file_handler = logging.handlers.RotatingFileHandler(
            settings.LOG_FILE,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=10
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        
        # Formatter
        if settings.LOG_FORMAT == "json":
            formatter = CustomJsonFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        return root_logger
    
    @staticmethod
    def get_logger(name: str):
        """Get or create logger"""
        if name not in LoggerFactory._loggers:
            LoggerFactory._loggers[name] = logging.getLogger(name)
        return LoggerFactory._loggers[name]


# Initialize logging on import
LoggerFactory.setup_logging()


def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Configured logger instance
    """
    return LoggerFactory.get_logger(name)
