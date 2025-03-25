import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os

class DatabaseConnector:
    def __init__(self, db_uri=None):
        load_dotenv()
        self.db_uri = db_uri or os.getenv('AIVEN_DB_URI')

        if not self.db_uri:
            raise ValueError("Database URI must be provided either via argument or environment variable.")

        self.connection = None
        self.cursor = None

    def connect(self):
        """Establish the connection to the Aiven database."""
        try:
            self.connection = psycopg2.connect(self.db_uri)
            self.cursor = self.connection.cursor()
            print("Connection to Aiven database established successfully.")
        except Exception as e:
            print(f"Error connecting to database: {e}")

    def execute_query(self, query):
        """Execute a query and fetch results."""
        try:
            self.cursor.execute(query)
            if query.strip().lower().startswith("select"):
                return self.cursor.fetchall()
            else:
                self.connection.commit()  # commit if it's an update/insert
        except Exception as e:
            print(f"Error executing query: {e}")

    def create_db(self, db_name):
        """Create a new database."""
        query = f"CREATE DATABASE {db_name};"
        self.execute_query(query)
        print(f"Database '{db_name}' created successfully.")

    def insert_parquet(self, table_name, parquet_file_path):
        """Insert data from a Parquet file into a specified database table."""
        try:
            df = pd.read_parquet(parquet_file_path)

            for _, row in df.iterrows():
                columns = ', '.join(df.columns)
                values = ', '.join([f"'{v}'" for v in row])
                query = f"INSERT INTO {table_name} ({columns}) VALUES ({values});"
                self.execute_query(query)
            print(f"Data from {parquet_file_path} inserted into {table_name} successfully.")
        except Exception as e:
            print(f"Error inserting data from Parquet file: {e}")

    def close(self):
        """Close the database connection and cursor."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        print("Connection closed.")

if __name__ == "__main__":
    connector = DatabaseConnector()  # None so loaded from environment variable
    connector.connect()

    # Example query 1: Fetch PostgreSQL version
    query_sql = 'SELECT VERSION()'
    version = connector.execute_query(query_sql)
    if version:
        print(version[0][0])  # Print the version

    # # Example query 2: Create a table if it doesnt exist
    # create_table_query = """
    # CREATE TABLE IF NOT EXISTS stock_mentions (
    #     ticker VARCHAR(50),
    #     frequency INTEGER
    # );
    # """
    # connector.execute_query(create_table_query)
    # print("Table 'stock_mentions' created (if not exists).")

    # # Insert data from Parquet file into the table
    # parquet_file_path = 'data/stock_mentions_frequencies.parquet'  # Path to your Parquet file
    # connector.insert_parquet('stock_mentions', parquet_file_path)

    # connector.close()

    # Example query 3: fetch and print all data from the stock_mentions table
    fetch_query = 'SELECT * FROM stock_mentions;'
    results = connector.execute_query(fetch_query)
    if results:
        for row in results:
            print(row)
