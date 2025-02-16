import logging
import re
import pandas as pd

logger = logging.getLogger(__name__)

class DataCleaner:
    @staticmethod
    def clean_submission_text(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the submission text by removing non-ASCII characters and extra whitespace.
        """
        if "title" in df.columns:
            df["title"] = df["title"].apply(DataCleaner._clean_text)
        if "selftext" in df.columns:
            df["selftext"] = df["selftext"].apply(DataCleaner._clean_text)
        return df

    @staticmethod
    def _clean_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r'[^\x00-\x7F]+',' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
