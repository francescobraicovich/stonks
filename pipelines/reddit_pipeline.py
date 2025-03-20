import os
import logging
import yaml
import pandas as pd
from typing import Tuple
import time
from dotenv import load_dotenv

# Import our custom classes
from data_ingestion.reddit_ingestor import RedditIngestor
from data_ingestion.data_cleaner import DataCleaner

# Load environment variables from a .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("reddit_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "data"
SUBMISSIONS_FILE = "submissions_latest.parquet"
COMMENTS_FILE = "comments_latest.parquet"

def ensure_data_directory():
    """Ensures that the data directory exists."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logger.info(f"Created data directory at {DATA_DIR}")

def save_to_parquet(submissions_df: pd.DataFrame, comments_df: pd.DataFrame):
    """
    Saves submissions and comments to separate Parquet files.
    
    Args:
        submissions_df: DataFrame containing submission data
        comments_df: DataFrame containing comment data
    """
    ensure_data_directory()
    
    submissions_path = os.path.join(DATA_DIR, SUBMISSIONS_FILE)
    comments_path = os.path.join(DATA_DIR, COMMENTS_FILE)
    
    # Save submissions
    if not submissions_df.empty:
        submissions_df.to_parquet(submissions_path, index=False)
        logger.info(f"Saved {len(submissions_df)} submissions to {submissions_path}")
    else:
        logger.warning("No submissions data to save.")
    
    # Save comments
    if not comments_df.empty:
        comments_df.to_parquet(comments_path, index=False)
        logger.info(f"Saved {len(comments_df)} comments to {comments_path}")
    else:
        logger.warning("No comments data to save.")

def fetch_reddit_data(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch Reddit submissions and comments based on the configuration.
    
    Args:
        config: Dictionary containing the configuration
        
    Returns:
        Tuple containing (submissions_df, comments_df)
    """
    # Extract Reddit API credentials
    reddit_client_id = os.getenv("REDDIT_CLIENT_ID", config["reddit_api"].get("client_id"))
    reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET", config["reddit_api"].get("client_secret"))
    reddit_username = os.getenv("REDDIT_USERNAME", config["reddit_api"].get("username"))
    reddit_password = os.getenv("REDDIT_PASSWORD", config["reddit_api"].get("password"))
    user_agent = config["reddit_api"].get("user_agent", "RedditDataCollector/1.0")
    
    # Extract other configuration options
    subreddits = config["reddit_api"]["subreddits"]
    limit_per_subreddit = config["reddit_api"].get("limit_per_subreddit", 100)
    comments_per_submission = config["reddit_api"].get("comments_per_submission", None)
    filter_media = config["reddit_api"].get("filter_media", True)
    
    # Initialize Reddit API
    reddit_ingestor = RedditIngestor(
        client_id=reddit_client_id,
        client_secret=reddit_client_secret,
        username=reddit_username,
        password=reddit_password,
        user_agent=user_agent,
    )
    
    all_submissions = []
    all_comments = []
    
    # Process each subreddit
    for subreddit in subreddits:
        logger.info(f"Processing subreddit: r/{subreddit}")
        
        try:
            # Fetch latest submissions
            subreddit_submissions = reddit_ingestor.fetch_latest_submissions(
                subreddit_name=subreddit,
                limit=limit_per_subreddit,
                filter_media=filter_media
            )
            
            if subreddit_submissions.empty:
                logger.warning(f"No submissions found for r/{subreddit}")
                continue
                
            # Clean submission text
            subreddit_submissions = DataCleaner.clean_submission_text(subreddit_submissions)
            
            # Fetch comments for these submissions
            subreddit_comments = reddit_ingestor.fetch_comments_for_submissions(
                submissions_df=subreddit_submissions,
                top_n=comments_per_submission
            )
            
            # Clean comment text
            if 'body' in subreddit_comments.columns:
                subreddit_comments['body'] = subreddit_comments['body'].apply(DataCleaner._clean_text)
            
            # Add to our collection
            all_submissions.append(subreddit_submissions)
            all_comments.append(subreddit_comments)
            
            logger.info(f"Completed processing r/{subreddit}")
            
        except Exception as e:
            logger.error(f"Error processing subreddit r/{subreddit}: {e}")
        
        # Sleep between subreddits to avoid rate limits
        time.sleep(1)
    
    # Combine all subreddits
    if all_submissions:
        submissions_df = pd.concat(all_submissions, ignore_index=True)
        logger.info(f"Total submissions collected: {len(submissions_df)}")
    else:
        submissions_df = pd.DataFrame()
        logger.warning("No submissions collected from any subreddit.")
    
    if all_comments:
        comments_df = pd.concat(all_comments, ignore_index=True)
        logger.info(f"Total comments collected: {len(comments_df)}")
    else:
        comments_df = pd.DataFrame()
        logger.warning("No comments collected from any subreddit.")
    
    return submissions_df, comments_df

def run_pipeline(config_path: str):
    """
    Run the Reddit data pipeline.
    
    Args:
        config_path: Path to the YAML configuration file
    """
    try:
        # Load configuration
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        
        # Fetch Reddit data
        submissions_df, comments_df = fetch_reddit_data(config)

        print(submissions_df.head())
        print(comments_df.head())
        
        # Save data to Parquet files
        save_to_parquet(submissions_df, comments_df)
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
