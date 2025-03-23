import os
import logging
import pandas as pd
from typing import Any, Dict, List, Tuple
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


def load_tickers(file_path: str, file_type: str = 'parquet', min_number_letters: int = 3) -> pd.DataFrame:
    """
    Load stock tickers from a file, either .txt or .parquet.

    Args:
        file_path: Path to the file containing tickers.
        file_type: Type of file ('txt' or 'parquet').
        min_number_letters: Minimum number of letters a ticker must have to be considered.

    Returns:
        DataFrame containing stock tickers and names.
    """
    try:
        if file_type == 'txt':
            with open(file_path, 'r') as file:
                tickers = [
                    line.strip() for line in file if line.strip() and len(line.strip()) >= min_number_letters
                ]
            stocks_df = pd.DataFrame({'Ticker': tickers, 'Name': tickers})
            logger.info(f"Loaded {len(stocks_df)} tickers from {file_path} (filtered by length >= {min_number_letters})")
        elif file_type == 'parquet':
            stocks_df = pd.read_parquet(file_path)
            stocks_df = stocks_df[stocks_df['Ticker'].str.len() >= min_number_letters]
            logger.info(f"Loaded {len(stocks_df)} stocks from {file_path} (filtered by length >= {min_number_letters})")
        else:
            raise ValueError("Invalid file_type. Use 'txt' or 'parquet'.")

        return stocks_df
    except Exception as e:
        raise Exception(f"Failed to load stocks: {e}")


def extract_stock_mentions_with_context(
    text_data: List[Dict[str, Any]], 
    stocks_df: pd.DataFrame, 
    ticker_threshold: int = 90, 
    name_threshold: int = 85,
    input_type: str = 'both',
    excluded_tickers: dict = {'ARE', 'HAS', 'ALL', 'NOW', 'TECH', 'KEY', 'LOW', 'DAY', 'WELL', 'FAST', 'COST'}
) -> Tuple[pd.DataFrame, Counter]:
    """
    Extract stock mentions from text data using fuzzy matching with full context.

    Args:
        text_data: List of dictionaries with 'text', 'origin', and 'original_index' keys.
        stocks_df: DataFrame with 'Ticker' and 'Name' columns.
        ticker_threshold: Minimum similarity ratio for considering a ticker match.
        name_threshold: Minimum similarity ratio for considering a name match.
        input_type: Specify whether to match by 'ticker', 'name', or 'both'.
        excluded_tickers: Set of tickers to exclude from matching.

    Returns:
        Tuple of (DataFrame containing all stock mentions with context, Counter of ticker frequencies).
    """
    all_mentions = []
    
    tickers_list = stocks_df['Ticker'].tolist()
    names_list = stocks_df['Name'].tolist()
    ticker_to_name = dict(zip(stocks_df['Ticker'], stocks_df['Name']))
    name_to_ticker = dict(zip(stocks_df['Name'], stocks_df['Ticker']))

    for idx, item in enumerate(text_data):
        text = item['text']
        words = text.split()
        for word in words:
            if input_type in ('ticker', 'both'):
                ticker_result = process.extractOne(word.upper(), tickers_list, scorer=fuzz.ratio)
                if ticker_result and ticker_result[1] >= ticker_threshold and ticker_result[0] not in excluded_tickers:
                    mention = {
                        'ticker': ticker_result[0],
                        'company_name': ticker_to_name.get(ticker_result[0], ''),
                        'original_word': word,
                        'match_type': 'ticker',
                        'match_score': ticker_result[1],
                        'text': text,
                        'text_index': idx,
                        'origin': item['origin'],
                        'original_index': item['original_index']
                    }
                    all_mentions.append(mention)
                    continue

            if input_type in ('name', 'both'):
                name_result = process.extractOne(word, names_list, scorer=fuzz.ratio)
                if name_result and name_result[1] >= name_threshold:
                    mention = {
                        'ticker': name_to_ticker.get(name_result[0], ''),
                        'company_name': name_result[0],
                        'original_word': word,
                        'match_type': 'name',
                        'match_score': name_result[1],
                        'text': text,
                        'text_index': idx,
                        'origin': item['origin'],
                        'original_index': item['original_index']
                    }
                    all_mentions.append(mention)

    mentions_df = pd.DataFrame(all_mentions)
    frequencies = Counter(mentions_df['ticker']) if not mentions_df.empty else Counter()

    logger.info(f"Found {len(mentions_df)} stock mentions in text data")
    logger.info(f"Unique stocks mentioned: {len(frequencies)}")

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
        ticker_file = os.path.join(DATA_DIR, "sp500_names.parquet")
    
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

