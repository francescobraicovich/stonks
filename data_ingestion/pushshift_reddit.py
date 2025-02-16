# data_ingestion/pushshift_reddit.py

import requests
import time
import logging
import pandas as pd
from typing import List, Dict

logger = logging.getLogger(__name__)

class PushshiftRedditIngestor:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def fetch_submissions(
        self, 
        subreddit: str, 
        start_date: int, 
        end_date: int, 
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch submissions from Pushshift for a specific subreddit
        within a date range (UNIX timestamps).

        Args:
            subreddit: The subreddit name (e.g., "wallstreetbets").
            start_date: Start UNIX timestamp.
            end_date: End UNIX timestamp.
            limit: Number of posts per request.
        Returns:
            A pandas DataFrame with the submissions.
        """
        all_data = []
        last_timestamp = end_date

        while True:
            # Construct URL
            url = (
                f"{self.base_url}/submission/"
                f"?subreddit={subreddit}"
                f"&size={limit}"
                f"&before={last_timestamp}"
                f"&after={start_date}"
            )
            resp = requests.get(url)
            
            if resp.status_code != 200:
                logger.error(f"Pushshift request failed with status {resp.status_code}")
                break

            data = resp.json().get("data", [])

            if not data:
                # No more data
                break

            # Convert JSON to list of dictionaries
            for submission in data:
                all_data.append(submission)

            # Update the last timestamp to fetch older posts
            last_timestamp = data[-1]["created_utc"]
            # Sleep to respect rate limits
            time.sleep(1)

        df = pd.DataFrame(all_data)
        logger.info(f"Fetched {len(df)} submissions from r/{subreddit}")
        return df
