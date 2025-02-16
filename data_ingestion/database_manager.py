# data_storage/database_manager.py

import logging
import pandas as pd
from pymongo import MongoClient

logger = logging.getLogger(__name__)

class MongoDBManager:
    def __init__(self, uri: str, db_name: str = "wsb_db"):
        """
        Args:
            uri: MongoDB connection string (e.g., "mongodb://user:password@localhost:27017")
            db_name: Name of the database to use.
        """
        self.client = MongoClient(uri)
        self.db = self.client[db_name]

    def store_submissions(self, df: pd.DataFrame, collection_name: str = "wsb_submissions"):
        """
        Insert Reddit submission documents into a MongoDB collection.
        """
        try:
            records = df.to_dict("records")
            if records:
                self.db[collection_name].insert_many(records)
                logger.info(f"Inserted {len(records)} submissions into {collection_name}.")
            else:
                logger.warning("No submission records to insert.")
        except Exception as e:
            logger.error(f"Error inserting submissions: {e}")

    def store_prices(self, df: pd.DataFrame, collection_name: str = "ticker_prices"):
        """
        Insert price data into a MongoDB collection.
        """
        try:
            # Reset index in case of a multi-index DataFrame (from yfinance)
            df_reset = df.reset_index()
            records = df_reset.to_dict("records")
            if records:
                self.db[collection_name].insert_many(records)
                logger.info(f"Inserted {len(records)} price records into {collection_name}.")
            else:
                logger.warning("No price records to insert.")
        except Exception as e:
            logger.error(f"Error inserting prices: {e}")
