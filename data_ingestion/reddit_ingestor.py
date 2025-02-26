import praw
import time
import logging
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class RedditIngestor:
    def __init__(
        self, 
        client_id: str, 
        client_secret: str, 
        username: str = None, 
        password: str = None, 
        user_agent: str = "RedditDataCollector/1.0"
    ):
        """
        Initialize Reddit API connection using PRAW.
        
        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            username: Reddit username (optional for read-only access)
            password: Reddit password (optional for read-only access)
            user_agent: Custom user agent string
        """
        # Configure PRAW with provided credentials
        praw_config = {
            "client_id": client_id,
            "client_secret": client_secret,
            "user_agent": user_agent,
        }
        
        # Include username/password if provided
        if username and password:
            praw_config["username"] = username
            praw_config["password"] = password
        
        try:
            self.reddit = praw.Reddit(**praw_config)
            logger.info(f"PRAW initialized successfully. Read-only: {self.reddit.read_only}")
        except Exception as e:
            logger.error(f"Failed to initialize PRAW: {e}")
            raise

    def fetch_latest_submissions(
        self,
        subreddit_name: str,
        limit: int = 100,
        filter_media: bool = True
    ) -> pd.DataFrame:
        """
        Fetch the latest submissions from a subreddit.
        
        Args:
            subreddit_name: Name of the subreddit
            limit: Maximum number of submissions to fetch
            filter_media: If True, filter out submissions with images or videos
            
        Returns:
            DataFrame containing the submissions
        """
        logger.info(f"Fetching latest {limit} submissions from r/{subreddit_name}")
        
        subreddit = self.reddit.subreddit(subreddit_name)
        submissions_data = []
        
        # Get latest submissions from the subreddit
        for submission in subreddit.new(limit=limit * 2):  # Fetch extra to account for filtered posts
            # Skip if it contains media and filter_media is True
            if filter_media and (submission.is_video or hasattr(submission, 'is_gallery') and submission.is_gallery):
                continue
                
            # Skip if it's an image post (checking common image domains and URL patterns)
            if filter_media and self._is_image_post(submission):
                continue
                
            # Extract submission data
            submission_data = {
                "id": submission.id,
                "title": submission.title,
                "selftext": submission.selftext if hasattr(submission, 'selftext') else "",
                "score": submission.score,
                "upvote_ratio": submission.upvote_ratio,
                "num_comments": submission.num_comments,
                "created_utc": submission.created_utc,
                "created_date": datetime.utcfromtimestamp(submission.created_utc),
                "subreddit": subreddit_name,
                "author": str(submission.author) if submission.author else "[deleted]",
                "permalink": submission.permalink,
                "url": submission.url,
                "is_self": submission.is_self,
            }
            
            submissions_data.append(submission_data)
            
            # Break if we've collected enough submissions
            if len(submissions_data) >= limit:
                break
                
            # Sleep briefly to avoid hitting rate limits
            time.sleep(0.1)
            
        df = pd.DataFrame(submissions_data)
        logger.info(f"Fetched {len(df)} submissions from r/{subreddit_name}")
        return df
    
    def fetch_comments_for_submissions(self, submissions_df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
        all_comments = []
        submission_ids = submissions_df['id'].tolist()

        logger.info(f"Fetching comments for {len(submission_ids)} submissions")

        for submission_id in submission_ids:
            try:
                submission = self.reddit.submission(id=submission_id)
                # Set sort order to 'top'
                submission.comment_sort = "top"
                submission.comments.replace_more(limit=0)
                
                comments = submission.comments.list()
                # Slice top 50 comments directly, as they should be sorted by score
                top_comments = comments[:top_n]

                for comment in top_comments:
                    if not comment or not hasattr(comment, 'id'):
                        continue

                    comment_data = {
                        "comment_id": comment.id,
                        "submission_id": submission_id,
                        "body": comment.body if hasattr(comment, 'body') else "",
                        "score": comment.score if hasattr(comment, 'score') else 0,
                        "created_utc": comment.created_utc if hasattr(comment, 'created_utc') else 0,
                        "created_date": datetime.utcfromtimestamp(comment.created_utc) if hasattr(comment, 'created_utc') else None,
                        "author": str(comment.author) if hasattr(comment, 'author') and comment.author else "[deleted]",
                        "parent_id": comment.parent_id if hasattr(comment, 'parent_id') else None,
                        "is_submitter": comment.is_submitter if hasattr(comment, 'is_submitter') else False,
                    }
                    all_comments.append(comment_data)

                logger.info(f"Fetched {len(top_comments)} comments for submission {submission_id}")
                
            except Exception as e:
                logger.error(f"Error fetching comments for submission {submission_id}: {e}")
            
            time.sleep(0.2)

        comments_df = pd.DataFrame(all_comments)
        logger.info(f"Fetched a total of {len(comments_df)} comments for all submissions")
        return comments_df

    
    def _is_image_post(self, submission) -> bool:
        """
        Check if a submission contains an image.
        
        Args:
            submission: PRAW submission object
            
        Returns:
            True if the submission contains an image, False otherwise
        """
        # Check for common image file extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        url_lower = submission.url.lower()
        
        if any(url_lower.endswith(ext) for ext in image_extensions):
            return True
            
        # Check for common image hosting domains
        image_domains = ['imgur.com', 'i.redd.it', 'i.imgur.com', 'ibb.co']
        if any(domain in url_lower for domain in image_domains):
            return True
            
        # Check for Reddit gallery
        if hasattr(submission, 'is_gallery') and submission.is_gallery:
            return True
            
        return False