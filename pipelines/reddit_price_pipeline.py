# pipelines/reddit_price_pipeline.py

import logging
import yaml
from datetime import datetime
from data_ingestion.pushshift_reddit import PushshiftRedditIngestor
from data_ingestion.yahoo_finance import YahooFinanceIngestor
from data_ingestion.data_cleaner import DataCleaner
from data_storage.database_manager import MongoDBManager

logger = logging.getLogger(__name__)

def run_pipeline(config_path: str):
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    pushshift_url = config["reddit_api"]["pushshift_url"]
    subreddits = config["reddit_api"]["subreddits"]
    db_uri = config["database"]["uri"]
    db_name = config["database"]["db_name"]
    tickers = config["yahoo_finance"]["tickers"]

    # Initialize ingestion and storage classes
    reddit_ingestor = PushshiftRedditIngestor(base_url=pushshift_url)
    yahoo_ingestor = YahooFinanceIngestor(tickers=tickers)
    db_manager = MongoDBManager(uri=db_uri, db_name=db_name)

    # Define date ranges (example: data for 2021)
    start_date_unix = int(datetime(2021, 1, 1).timestamp())
    end_date_unix = int(datetime(2022, 1, 1).timestamp())

    # Loop through each subreddit and ingest data
    all_submissions = []
    for subreddit in subreddits:
        logger.info(f"Starting ingestion for r/{subreddit}")
        submissions_df = reddit_ingestor.fetch_submissions(subreddit, start_date_unix, end_date_unix)
        submissions_df = DataCleaner.clean_submission_text(submissions_df)
        logger.info(f"Total records for r/{subreddit}: {len(submissions_df)}")
        # Optionally, you could store each subreddit separately by setting collection_name=f"reddit_{subreddit}"
        db_manager.store_submissions(submissions_df, collection_name="reddit_submissions")
        all_submissions.append(submissions_df)

    # Optionally, combine all submissions for further processing
    # combined_df = pd.concat(all_submissions, ignore_index=True)

    # Fetch historical price data (example: for 2021)
    yahoo_start = "2021-01-01"
    yahoo_end = "2022-01-01"
    price_df = yahoo_ingestor.fetch_historical_data(yahoo_start, yahoo_end)
    db_manager.store_prices(price_df, collection_name="ticker_prices")

    logger.info("Pipeline run completed successfully.")
