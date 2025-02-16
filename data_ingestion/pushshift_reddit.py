# data_ingestion/pushshift_reddit.py

import requests
import time
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class PushshiftRedditIngestor:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def fetch_submissions(self, subreddit: str, start_date: int, end_date: int, limit: int = 100) -> pd.DataFrame:
        """
        Fetch submissions from Pushshift for a given subreddit and date range.

        Args:
            subreddit: The subreddit name.
            start_date: Start UNIX timestamp.
            end_date: End UNIX timestamp.
            limit: Number of posts per request.
        Returns:
            DataFrame with the submissions.
        """
        all_data = []
        last_timestamp = end_date

        while True:
            url = (
                f"{self.base_url}/submission/"
                f"?subreddit={subreddit}"
                f"&size={limit}"
                f"&before={last_timestamp}"
                f"&after={start_date}"
            )
            resp = requests.get(url)
            if resp.status_code != 200:
                logger.error(f"Pushshift request failed for {subreddit} with status {resp.status_code}")
                break

            data = resp.json().get("data", [])
            if not data:
                break

            all_data.extend(data)
            last_timestamp = data[-1]["created_utc"]
            time.sleep(1)  # Respect rate limits

        df = pd.DataFrame(all_data)
        if not df.empty:
            # Ensure the subreddit field is set (Pushshift sometimes includes it, but we can enforce it)
            df["subreddit"] = subreddit
        logger.info(f"Fetched {len(df)} submissions from r/{subreddit}")
        return df
