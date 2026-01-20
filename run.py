#!/usr/bin/env python3
"""
ML Service Deployment Runner
Handles startup, logging, error handling, and graceful shutdown
"""

import sys
import os
import signal
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import after path is set
from src.config.settings import get_settings
from src.utils.logger import LoggerFactory

# Initialize logging
LoggerFactory.setup_logging()
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()


class MLServiceRunner:
    """Manages ML Service lifecycle"""
    
    def __init__(self):
        self.app = None
        self.server = None
        self.running = False
        
    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown()
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    def load_models(self):
        """Pre-load ML models to verify they work before serving"""
        logger.info("=" * 80)
        logger.info("LOADING ML MODELS")
        logger.info("=" * 80)
        
        models_dir = PROJECT_ROOT / "models"
        
        try:
            from src.models.solar_forecast import SolarXGBoostModel
            logger.info("Loading Solar XGBoost Model...")
            solar_model = SolarXGBoostModel()
            solar_model_path = models_dir / "solar_xgboost_model.pkl"
            if solar_model_path.exists():
                solar_model.load(str(solar_model_path))
                logger.info("✓ Solar XGBoost Model loaded successfully")
            else:
                logger.warning(f"⚠ Solar XGBoost Model not found at {solar_model_path} - will use default")
                
        except Exception as e:
            logger.error(f"✗ Error loading Solar XGBoost Model: {e}", exc_info=True)
            # Don't raise - allow service to start with default models
            
        try:
            from src.models.demand_forecast import DemandXGBoostModel
            logger.info("Loading Demand XGBoost Model...")
            demand_model = DemandXGBoostModel()
            demand_model_path = models_dir / "demand_xgboost_model.pkl"
            if demand_model_path.exists():
                demand_model.load(str(demand_model_path))
                logger.info("✓ Demand XGBoost Model loaded successfully")
            else:
                logger.warning(f"⚠ Demand XGBoost Model not found at {demand_model_path} - will use default")
                
        except Exception as e:
            logger.error(f"✗ Error loading Demand XGBoost Model: {e}", exc_info=True)
            # Don't raise - allow service to start with default models
            
        logger.info("=" * 80)
        
    def start(self):
        """Start the ML Service"""
        logger.info("=" * 80)
        logger.info("SOLAR SHARING ML SERVICE - STARTING")
        logger.info("=" * 80)
        logger.info(f"Environment: {settings.ENVIRONMENT}")
        logger.info(f"Service Name: {settings.SERVICE_NAME}")
        logger.info(f"Version: {settings.SERVICE_VERSION}")
        logger.info(f"Log Level: {settings.LOG_LEVEL}")
        logger.info(f"Log Directory: {settings.LOG_DIR}")
        logger.info(f"Host: {settings.HOST}")
        logger.info(f"Port: {settings.PORT}")
        logger.info("=" * 80)
        
        self.setup_signal_handlers()
        
        try:
            # Load and verify models
            self.load_models()
            
            # Import FastAPI app
            logger.info("Initializing FastAPI application...")
            from src.api.main import app
            self.app = app
            
            # Import uvicorn
            import uvicorn
            
            # Configure uvicorn
            config = uvicorn.Config(
                app=self.app,
                host=settings.HOST,
                port=settings.PORT,
                log_level=settings.LOG_LEVEL.lower(),
                access_log=True,
                reload=settings.ENVIRONMENT == 'development',
            )
            
            # Create server
            self.server = uvicorn.Server(config)
            self.running = True
            
            logger.info("=" * 80)
            logger.info(f"✓ ML SERVICE STARTED SUCCESSFULLY")
            logger.info(f"✓ API available at: http://{settings.HOST}:{settings.PORT}")
            logger.info(f"✓ Documentation at: http://{settings.HOST}:{settings.PORT}/docs")
            logger.info("=" * 80)
            
            # Start server (blocking)
            return self.server.run()
            
        except ImportError as e:
            logger.error(f"✗ Import Error: {e}", exc_info=True)
            logger.error("Make sure all dependencies are installed: pip install -r requirements.txt")
            sys.exit(1)
            
        except Exception as e:
            logger.error(f"✗ Failed to start ML Service: {e}", exc_info=True)
            sys.exit(1)
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("=" * 80)
        logger.info("SHUTTING DOWN ML SERVICE")
        logger.info("=" * 80)
        self.running = False
        
        if self.server:
            try:
                self.server.should_exit = True
                logger.info("✓ Server shutdown initiated")
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
                
        logger.info("=" * 80)
        logger.info("Service stopped.")
        logger.info("=" * 80)


def main():
    """Main entry point"""
    runner = MLServiceRunner()
    
    try:
        runner.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        runner.shutdown()
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Critical error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
