
import logging
import os
from pipelines.reddit_price_pipeline import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

if __name__ == "__main__":
    config_path = os.getenv("CONFIG_PATH", "config/config.yml")

    
    run_pipeline(config_path)
