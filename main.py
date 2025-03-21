
import logging
import os
import argparse

from pipelines.reddit_pipeline import run_pipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Reddit Data Ingestion Pipeline")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml", 
        help="Path to configuration YAML file"
    )
    args = parser.parse_args()

    config_path = os.getenv("CONFIG_PATH", "config/config.yml")

    run_pipeline(config_path)
