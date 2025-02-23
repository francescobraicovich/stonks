import praw
import logging
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RedditIngestor:
    def __init__(self, client_id, client_secret, username, password, user_agent):
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

    def fetch_submissions(self, subreddit_name: str, days: int = 30) -> pd.DataFrame:
        """
        Fetch Reddit submissions from a subreddit within the last X days.
        """
        logger.info(f"Fetching posts from r/{subreddit_name} for the last {days} days")

        subreddit = self.reddit.subreddit(subreddit_name)
        all_posts = []
        time_threshold = datetime.utcnow() - timedelta(days=days)

        for post in subreddit.new(limit=1000):  # Get the most recent 1000 posts
            post_time = datetime.utcfromtimestamp(post.created_utc)
            if post_time < time_threshold:
                break  # Stop if post is too old

            all_posts.append({
                "id": post.id,
                "title": post.title,
                "selftext": post.selftext,
                "score": post.score,
                "upvote_ratio": post.upvote_ratio,
                "num_comments": post.num_comments,
                "created_utc": post.created_utc,
                "subreddit": subreddit_name,
            })

        df = pd.DataFrame(all_posts)
        logger.info(f"Fetched {len(df)} posts from r/{subreddit_name}")
        return df