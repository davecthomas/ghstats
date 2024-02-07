import os
import pandas as pd
import snowflake.connector
from typing import Dict, Optional


class ContributorStatsStorageManager:
    def __init__(self):
        self.conn: Optional[snowflake.connector.SnowflakeConnection] = None

    def __del__(self):
        """Destructor to ensure the Snowflake connection is closed."""
        try:
            self.close_connection()
        except Exception as e:
            print(f"Error closing Snowflake connection: {e}")

    def close_connection(self):
        """Closes the Snowflake connection if it's open."""
        if self.conn is not None and not self.conn.is_closed():
            self.conn.close()
            self.conn = None

    def get_db_env(self) -> Dict[str, str]:
        """Fetches database environment variables."""
        dict_db_env = {
            "snowflake_user": os.getenv("SNOWFLAKE_USER"),
            "snowflake_password": os.getenv("SNOWFLAKE_PASSWORD"),
            "snowflake_account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "snowflake_warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
            "snowflake_db": os.getenv("SNOWFLAKE_DB"),
            "snowflake_schema": os.getenv("SNOWFLAKE_SCHEMA"),
            "snowflake_table_name": os.getenv("SNOWFLAKE_TABLE_NAME")
        }
        return dict_db_env

    def get_snowflake_connection(self):
        """Establishes a connection to Snowflake."""
        if self.conn is None or self.conn.is_closed():
            dict_db_env = self.get_db_env()
            self.conn = snowflake.connector.connect(
                user=dict_db_env["snowflake_user"],
                password=dict_db_env["snowflake_password"],
                account=dict_db_env["snowflake_account"],
                warehouse=dict_db_env["snowflake_warehouse"],
                database=dict_db_env["snowflake_db"],
                schema=dict_db_env["snowflake_schema"]
            )
        return self.conn

    def run_select_query(self, query: str) -> pd.DataFrame:
        """Executes a SELECT query and returns the results as a pandas DataFrame."""
        conn = self.get_snowflake_connection()

        # Use the connection in a context manager to ensure it's closed properly
        with conn.cursor().execute(query) as cur:
            # Fetch the result set from the cursor object
            df = pd.DataFrame(cur.fetchall(), columns=[
                col[0] for col in cur.description])
        return df

    def save_df_to_csv(self, df: pd.DataFrame, filename: str):
        """Saves the DataFrame to a CSV file."""
        df.to_csv(filename, index=False)

    def save_summary_stats_csv(self, df: pd.DataFrame, filename: str):
        # Generate descriptive statistics
        summary = df.describe()
        summary.to_csv(f"summary_{filename}")

    def store_df_in_snowflake(self, df: pd.DataFrame):
        """Stores the DataFrame in a Snowflake table."""
        conn = self.get_snowflake_connection()
        # Assuming the Snowflake Connector for Python is already installed
        # and you have a DataFrame 'df' to upload.
        success, nchunks, nrows, _ = conn.write_pandas(
            df, self.dict_db_env["snowflake_table_name"].upper())
        print(
            f"Data stored in Snowflake table {self.dict_db_env['snowflake_table_name']}: {nrows} rows in {nchunks} chunks.")
