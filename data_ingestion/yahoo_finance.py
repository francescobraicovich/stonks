# data_ingestion/yahoo_finance.py

import yfinance as yf
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class YahooFinanceIngestor:
    def __init__(self, tickers: list):
        self.tickers = tickers

    def fetch_historical_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for all tickers.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.
        Returns:
            A pandas DataFrame with data for the tickers.
        """
        logger.info(f"Fetching data for tickers: {self.tickers}")
        data = yf.download(
            tickers=self.tickers,
            start=start_date,
            end=end_date,
            group_by='ticker'
        )
        logger.info("Data fetched successfully from Yahoo Finance.")
        return data
