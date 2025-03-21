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
        raise Exception(f"Failed to load tickers: {e}")


def extract_stock_mentions_with_context(text_data, tickers: List[str], threshold: int = 80):
    """
    Extract stock mentions from text data using fuzzy matching with full context.

    Args:
        text_data: List of dictionaries with 'text', 'origin', and 'original_index' keys
        tickers: List of valid stock tickers.
        threshold: Minimum similarity ratio for considering a match.

    Returns:
        DataFrame containing all stock mentions with context
    """
    all_mentions = []
    
    for idx, item in enumerate(text_data):
        text = item['text']
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
                        'text_index': idx,
                        'origin': item['origin'],
                        'original_index': item['original_index']
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
            logger.info(f"Example: {example['original_word']} in '{example['text']}'")
            logger.info(f"Origin: {example['origin']} (Index: {example['original_index']})")
            logger.info(f"Match Score: {example['match_score']}")


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
    mentions_df.to_parquet(mentions_filepath, compression = 'brotli') # approx 70% compression
    
    freq_filepath = filepath.replace('.parquet', '_frequencies.parquet')
    freq_df.to_parquet(freq_filepath, compression = 'brotli')
    
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
        
        text_data = []
        
        for idx, title in enumerate(submissions_df['title']):
            text_data.append({
                'text': title,
                'origin': 'submission_title',
                'original_index': submissions_df.index[idx]
            })
        
        for idx, selftext in enumerate(submissions_df['selftext']):
            text_data.append({
                'text': selftext,
                'origin': 'submission_selftext',
                'original_index': submissions_df.index[idx]
            })
        
        for idx, body in enumerate(comments_df['body']):
            text_data.append({
                'text': body,
                'origin': 'comment',
                'original_index': comments_df.index[idx]
            })
        
        logger.info(f"Loaded {len(text_data)} text samples from {SUBMISSIONS_FILE} and {COMMENTS_FILE}")
    except Exception as e:
        raise Exception(f"Failed to load data: {e}")

    mentions_df, frequencies = extract_stock_mentions_with_context(text_data, tickers)
    display_mentions_summary(mentions_df, frequencies, top_n)
    
    if save_results and not mentions_df.empty:
        save_results_to_parquet(mentions_df, frequencies, RESULTS_FILE)
    
    return mentions_df, frequencies


if __name__ == "__main__":
    mentions_df, frequencies = run_pipeline()
    logger.info("Pipeline completed successfully.")
    print(mentions_df[['ticker', 'original_word', 'match_score', 'origin', 'original_index']].head())