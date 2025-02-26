import praw
import requests
import time
import logging
import pandas as pd
from datetime import datetime
from typing import List

logger = logging.getLogger(__name__)

class RedditIngestor:
    def __init__(
        self, 
        client_id: str, 
        client_secret: str, 
        username: str, 
        password: str, 
        user_agent: str
    ):
        """
        Initialize Reddit API connection using PRAW (Python Reddit API Wrapper).
        """
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            username=username,
            password=password,
            user_agent=user_agent,
        )

    def fetch_submissions_via_praw(self, subreddit_name: str, days: int = 30) -> pd.DataFrame:
        """
        Example method: Fetch the most recent X days of submissions using PRAW's built-in .new().
        (Unchanged from your original example, limited to last X days, no exact date range.)
        """
        logger.info(f"Fetching posts from r/{subreddit_name} for the last {days} days")

        # Your existing method remains the same
        # ... code omitted for brevity ...
        return pd.DataFrame()  # Stub return

    def fetch_submissions_by_date_range(
        self, 
        subreddit_name: str, 
        start_date: datetime, 
        end_date: datetime, 
        batch_size: int = 100
    ) -> pd.DataFrame:
        """
        Fetch Reddit submissions from a subreddit between start_date and end_date
        using Pushshift to locate the IDs, then pulling detailed data via PRAW.
        
        :param subreddit_name: Subreddit to scrape.
        :param start_date: Datetime object for the earliest post creation time.
        :param end_date: Datetime object for the latest post creation time.
        :param batch_size: Number of submissions to fetch at a time from Pushshift.
        
        :return: DataFrame with all found submissions between the two dates.
        """
        # Convert datetimes to POSIX timestamps
        after = int(start_date.timestamp())
        before = int(end_date.timestamp())

        logger.info(f"Fetching pushshift data for r/{subreddit_name} from {start_date} to {end_date}")

        all_post_ids = []
        pushshift_url = "https://api.pushshift.io/reddit/search/submission"

        # We'll paginate over pushshift in descending order, starting from 'before'
        # to get every submission in the specified window.
        current_before = before

        while True:
            params = {
                "subreddit": subreddit_name,
                "after": after,
                "before": current_before,
                "sort_type": "created_utc",
                "sort": "desc",
                "size": batch_size,
            }

            try:
                resp = requests.get(pushshift_url, params=params, timeout=30)
                resp.raise_for_status()
            except Exception as e:
                logger.error(f"Error querying Pushshift: {e}")
                break  # Exit on error, or implement a retry if desired.

            data = resp.json().get("data", [])
            print('data:', data)

            if not data:
                # No more submissions found in this date window.
                break

            # Collect submission IDs
            for submission in data:
                all_post_ids.append(submission["id"])

            # Prepare for the next iteration
            # Because we sorted descending, we take the last submission's created_utc
            # and make that our new 'before'.
            last_utc = data[-1]["created_utc"]
            current_before = last_utc

            # If the last submission is exactly at the threshold, we might get stuck.
            # We'll move one second lower to avoid duplicates in the next batch.
            current_before -= 1

            # Sleep to avoid rate-limiting
            time.sleep(1)

        logger.info(f"Found {len(all_post_ids)} unique submission IDs via Pushshift.")

        # Now, use PRAW to fetch the full data for each post
        # (Pushshift data can sometimes be incomplete or out-of-date)
        submissions_data = []
        for post_id in all_post_ids:
            try:
                # Use PRAW to get the actual Submission object
                submission = self.reddit.submission(id=post_id)

                # Optionally check the creation time to ensure it's in [start_date, end_date].
                created_dt = datetime.utcfromtimestamp(submission.created_utc)
                if created_dt < start_date or created_dt > end_date:
                    continue

                submissions_data.append({
                    "id": submission.id,
                    "title": submission.title,
                    "selftext": submission.selftext,
                    "score": submission.score,
                    "upvote_ratio": submission.upvote_ratio,
                    "num_comments": submission.num_comments,
                    "created_utc": submission.created_utc,
                    "subreddit": subreddit_name,
                    "author": str(submission.author) if submission.author else None
                })
            except Exception as e:
                logger.warning(f"Skipping post {post_id} due to an error: {e}")

            # Optional rate-limit sleep to avoid hitting Reddit's API too aggressively
            time.sleep(0.01)

        df = pd.DataFrame(submissions_data)
        logger.info(f"Fetched {len(df)} submissions from r/{subreddit_name} via PRAW.")
        return df
