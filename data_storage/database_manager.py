# data_storage/database_manager.py
import os
import logging
from pymongo import MongoClient
from dotenv import load_dotenv
import yaml

# Load environment variables from .env
load_dotenv()

logger = logging.getLogger(__name__)

class MongoDBManager:
    def __init__(self, config_path: str = "config/config.yml"):
        """
        Connect to MongoDB using environment variables.
        """
        # Load config.yml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Fetch credentials (from .env or config.yml)
        mongo_uri = os.getenv("MONGO_URI", config["database"]["uri"])
        db_name = os.getenv("DB_NAME", config["database"]["db_name"])

        if not mongo_uri:
            raise ValueError("MONGO_URI is not set. Please define it in .env or config.yml")

        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        logger.info(f"Connected to MongoDB database: {db_name}")


    def store_submissions(self, df: pd.DataFrame, collection_name: str = "reddit_submissions"):
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
            df_reset = df.reset_index()
            records = df_reset.to_dict("records")
            if records:
                self.db[collection_name].insert_many(records)
                logger.info(f"Inserted {len(records)} price records into {collection_name}.")
            else:
                logger.warning("No price records to insert.")
        except Exception as e:
            logger.error(f"Error inserting prices: {e}")
