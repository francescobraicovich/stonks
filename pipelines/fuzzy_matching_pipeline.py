import os
import logging
import pandas as pd
from typing import Dict, List
from collections import Counter
from rapidfuzz import fuzz, process

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/fuzzy_matching_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "data"
SUBMISSIONS_FILE = os.path.join(DATA_DIR, "historical_submissions.parquet")
COMMENTS_FILE = os.path.join(DATA_DIR, "historical_comments.parquet")


def load_tickers(file_path: str) -> List[str]:
    """
    Load stock tickers from a text file.

    Args:
        file_path: Path to the text file containing tickers (one ticker per line).

    Returns:
        A list of stock tickers.
    """
    try:
        with open(file_path, 'r') as file:
            tickers = [line.strip() for line in file if line.strip()]  # Read tickers from lines
        logger.info(f"Loaded {len(tickers)} tickers from {file_path}")
        return tickers
    except Exception as e:
        logger.error(f"Failed to load tickers: {e}")
        return []


def extract_stock_mentions(texts: List[str], tickers: List[str], threshold: int = 80) -> Counter:
    """
    Extract stock mentions from text data using fuzzy matching.

    Args:
        texts: List of text strings to analyze.
        tickers: List of valid stock tickers.
        threshold: Minimum similarity ratio for considering a match.

    Returns:
        Counter with frequency of stock mentions.
    """
    mentions = Counter()

    for text in texts:
        words = text.split()
        for word in words:
            result = process.extractOne(word.upper(), tickers, scorer=fuzz.ratio)
            if result is not None:
                match, score, _ = result
                if score >= threshold:
                    mentions[match] += 1

    return mentions


def display_common_mentions(mentions: Counter, top_n: int = 10):
    """
    Display the most common stock mentions.

    Args:
        mentions: Counter with stock mention frequencies.
        top_n: Number of top mentions to display.
    """
    logger.info("Most commonly mentioned stocks:")
    for stock, count in mentions.most_common(top_n):
        logger.info(f"{stock}: {count}")


def run_pipeline(ticker_file=None, top_n=10):
    """
    Run the complete stock mention extraction pipeline.
    
    Args:
        ticker_file: Path to the file containing stock tickers.
                    If None, uses default path.
        top_n: Number of top stock mentions to display.
        
    Returns:
        Counter with stock mention frequencies.
    """
    if ticker_file is None:
        ticker_file = os.path.join(DATA_DIR, "tickers.txt")
    
    tickers = load_tickers(ticker_file)

    try:
        submissions_df = pd.read_parquet(SUBMISSIONS_FILE)
        comments_df = pd.read_parquet(COMMENTS_FILE)
        texts = submissions_df['title'].tolist() + comments_df['body'].tolist()
    except Exception as e:
        logger.error(f"Failed to load Parquet files: {e}")
        texts = []

    mentions = extract_stock_mentions(texts, tickers)
    display_common_mentions(mentions, top_n)
    
    return mentions


if __name__ == "__main__":
    run_pipeline()