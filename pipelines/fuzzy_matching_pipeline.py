import os
import logging
import pandas as pd
from typing import List
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
RESULTS_FILE = os.path.join(DATA_DIR, "stock_mentions.parquet")


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


def extract_stock_mentions_with_context(texts: List[str], tickers: List[str], threshold: int = 80):
    """
    Extract stock mentions from text data using fuzzy matching with full context.

    Args:
        texts: List of text strings to analyze.
        tickers: List of valid stock tickers.
        threshold: Minimum similarity ratio for considering a match.

    Returns:
        DataFrame containing all stock mentions with context
    """
    all_mentions = []
    
    for idx, text in enumerate(texts):
        words = text.split()
        for word in words:
            result = process.extractOne(word.upper(), tickers, scorer=fuzz.ratio)
            if result is not None:
                ticker, score, _ = result
                if score >= threshold:
                    mention = {
                        'ticker': ticker,
                        'original_word': word,
                        'match_score': score,
                        'text': text,
                        'text_index': idx
                    }
                    all_mentions.append(mention)
    
    mentions_df = pd.DataFrame(all_mentions)
    frequencies = Counter(mentions_df['ticker'])
    
    return mentions_df, frequencies


def display_mentions_summary(mentions_df, frequencies, top_n=10):
    """
    Display summary of stock mentions.

    Args:
        mentions_df: DataFrame containing all mentions
        frequencies: Counter with mention frequencies
        top_n: Number of top mentions to display
    """
    logger.info("Most commonly mentioned stocks:")
    
    for ticker, count in frequencies.most_common(top_n):
        ticker_mentions = mentions_df[mentions_df['ticker'] == ticker]
        unique_texts = ticker_mentions['text_index'].nunique()
        
        logger.info(f"{ticker}: {count} mentions in {unique_texts} texts")
        
        if not ticker_mentions.empty:
            example = ticker_mentions.iloc[0]
            logger.info(f"  Example: \"{example['text']}\"")
            logger.info(f"    Matched word: \"{example['original_word']}\" (score: {example['match_score']})")


def save_results_to_parquet(mentions_df, frequencies, filepath):
    """
    Save results to Parquet files.

    Args:
        mentions_df: DataFrame of all mentions
        frequencies: Counter of frequencies
        filepath: Path to save the Parquet file
    """
    freq_df = pd.DataFrame({
        'ticker': list(frequencies.keys()),
        'frequency': list(frequencies.values())
    })
    
    mentions_filepath = filepath
    mentions_df.to_parquet(mentions_filepath)
    
    freq_filepath = filepath.replace('.parquet', '_frequencies.parquet')
    freq_df.to_parquet(freq_filepath)
    
    logger.info(f"Saved mentions to {mentions_filepath}")
    logger.info(f"Saved frequencies to {freq_filepath}")


def run_pipeline(ticker_file=None, top_n=10, save_results=True):
    """
    Run the complete stock mention extraction pipeline.
    
    Args:
        ticker_file: Path to the file containing stock tickers.
                    If None, uses default path.
        top_n: Number of top stock mentions to display.
        save_results: Whether to save results to Parquet file.
        
    Returns:
        Tuple of (mentions_df, frequencies)
    """
    if ticker_file is None:
        ticker_file = os.path.join(DATA_DIR, "tickers.txt")
    
    tickers = load_tickers(ticker_file)

    try:
        submissions_df = pd.read_parquet(SUBMISSIONS_FILE)
        comments_df = pd.read_parquet(COMMENTS_FILE)
        texts = submissions_df['title'].tolist() + comments_df['body'].tolist() + submissions_df['selftext'].tolist()
        logger.info(f"Loaded {len(texts)} text samples from {SUBMISSIONS_FILE} and {COMMENTS_FILE}")
    except Exception as e:
        logger.error(f"Failed to load Parquet files: {e}")
        texts = []

    mentions_df, frequencies = extract_stock_mentions_with_context(texts, tickers)
    display_mentions_summary(mentions_df, frequencies, top_n)
    
    if save_results and not mentions_df.empty:
        save_results_to_parquet(mentions_df, frequencies, RESULTS_FILE)
    
    return mentions_df, frequencies


if __name__ == "__main__":
    mentions_df, frequencies = run_pipeline()
    logger.info("Pipeline completed successfully.")
    print(mentions_df.head())