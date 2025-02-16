import yfinance as yf
import logging
import pandas as pd
from typing import List, Tuple

logger = logging.getLogger(__name__)

class YahooFinanceIngestor:
    def __init__(self, tickers: List[str]):
        """
        Initialize with the list of tickers you're interested in.
        """
        self.tickers = tickers

    def fetch_historical_data(
        self, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for all tickers in self.tickers.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.
        Returns:
            A pandas DataFrame with multi-index (Ticker, Date).
        """
        logger.info(f"Fetching data for tickers: {self.tickers}")

        data = yf.download(
            tickers=self.tickers,
            start=start_date,
            end=end_date,
            group_by='ticker'
        )

        # Data can be a multi-level DataFrame if multiple tickers
        logger.info("Data fetched successfully from Yahoo Finance.")
        return data
