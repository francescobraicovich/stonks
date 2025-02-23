import os
import logging
import yaml
import pandas as pd
from dotenv import load_dotenv  # Import dotenv
from data_ingestion.reddit_api import RedditIngestor

# Load environment variables from a .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = "data"

def ensure_data_directory():
    """Ensures that the 'data' directory exists."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def load_existing_data(subreddit: str):
    """Loads existing Parquet data for the subreddit if available."""
    file_path = os.path.join(DATA_DIR, f"{subreddit}.parquet")
    if os.path.exists(file_path):
        try:
            df = pd.read_parquet(file_path)
            logger.info(f"Loaded {len(df)} existing posts from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def save_to_parquet(df: pd.DataFrame, subreddit: str):
    """Saves the DataFrame to a Parquet file in the data directory."""
    if df.empty:
        logger.warning(f"No new data to save for r/{subreddit}. Skipping.")
        return
    
    file_path = os.path.join(DATA_DIR, f"{subreddit}.parquet")
    df.to_parquet(file_path, index=False)
    logger.info(f"Saved {len(df)} unique posts from r/{subreddit} to {file_path}")

def run_pipeline(config_path: str):
    """
    Runs the pipeline to fetch Reddit posts and store them in local Parquet files without duplicates.
    """
    ensure_data_directory()

    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load Reddit API credentials (fallback to .env if not in YAML)
    reddit_client_id = os.getenv("REDDIT_CLIENT_ID", config["reddit_api"].get("client_id"))
    reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET", config["reddit_api"].get("client_secret"))
    reddit_username = os.getenv("REDDIT_USERNAME", config["reddit_api"].get("username"))
    reddit_password = os.getenv("REDDIT_PASSWORD", config["reddit_api"].get("password"))
    user_agent = config["reddit_api"]["user_agent"]
    subreddits = config["reddit_api"]["subreddits"]
    days = config["reddit_api"].get("days", 30)

    # Initialize Reddit API
    reddit_ingestor = RedditIngestor(
        client_id=reddit_client_id,
        client_secret=reddit_client_secret,
        username=reddit_username,
        password=reddit_password,
        user_agent=user_agent,
    )

    # Fetch and store posts for each subreddit
    for subreddit in subreddits:
        logger.info(f"Starting ingestion for r/{subreddit} (last {days} days)")

        # Fetch new submissions
        new_df = reddit_ingestor.fetch_submissions(subreddit, days=days)

        # Load existing data and merge with new
        existing_df = load_existing_data(subreddit)
        combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=["id"]).reset_index(drop=True)

        # Save updated dataset
        save_to_parquet(combined_df, subreddit)

    logger.info("Pipeline completed successfully.")