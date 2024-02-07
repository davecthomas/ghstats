import os
import pandas as pd
import snowflake.connector
from typing import Dict, Optional, List
from snowflake.connector.pandas_tools import write_pandas


class ContributorStatsStorageManager:
    def __init__(self):
        self.dict_db_env = None
        self.conn: Optional[snowflake.connector.SnowflakeConnection] = None
        self.get_db_env()

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
        if self.dict_db_env is None:
            self.dict_db_env = {
                "snowflake_user": os.getenv("SNOWFLAKE_USER"),
                "snowflake_password": os.getenv("SNOWFLAKE_PASSWORD"),
                "snowflake_account": os.getenv("SNOWFLAKE_ACCOUNT"),
                "snowflake_warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
                "snowflake_db": os.getenv("SNOWFLAKE_DB"),
                "snowflake_schema": os.getenv("SNOWFLAKE_SCHEMA"),
                "snowflake_table_name_staging": os.getenv("SNOWFLAKE_TABLE_NAME_STAGING"),
                "snowflake_table_name": os.getenv("SNOWFLAKE_TABLE_NAME")
            }
        return self.dict_db_env

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

    def store_df(self, df: pd.DataFrame, table_name: str):
        """Stores the DataFrame in a Snowflake table."""
        conn = self.get_snowflake_connection()
        # Assuming the Snowflake Connector for Python is already installed
        # and you have a DataFrame 'df' to upload.
        success, nchunks, nrows, _ = write_pandas(conn,
                                                  df, table_name)
        print(
            f"Data stored in Snowflake table {table_name}: {nrows} rows in {nchunks} chunks.")

    def delete_all_rows(self, staging_table: str):
        conn = self.get_snowflake_connection()
        sql: str = f"""delete from "{staging_table}";"""
        conn.cursor().execute(sql)
        print("")
        # conn.close()

    def upsert_dataframe(self, df: pd.DataFrame, target_table: str, staging_table: str):
        """
        Upserts data from a DataFrame into the target table using a staging table.
        This is to avoid us having duplicate records when we re-run similar timeframes for the same
        repos. 

        :param df: DataFrame to upsert.
        :param target_table: Name of the target table for upsert.
        :param staging_table: Name of the staging table for initial DataFrame upload.
        """
        conn = self.get_snowflake_connection()
        # Prereq: this assume staging table is clear!
        self.delete_all_rows(staging_table)

        # Step 1: Upload DataFrame to a staging table
        write_pandas(conn, df, staging_table)

        # Step 2: Merge from staging table to target table
        merge_sql = f"""
        MERGE INTO "{target_table}" AS target
        USING "{staging_table}" AS staging
        ON target."contributor_username" = staging."contributor_username"
        AND target."repo" = staging."repo"
        AND target."stats_beginning" = staging."stats_beginning"
        WHEN NOT MATCHED THEN
            INSERT (
                "repo",
                "contributor_name",
                "contributor_username",
                "curved_score" ,
                "stats_beginning" ,
                "stats_ending" ,
                "contributor_first_commit_date" ,
                "num_workdays" ,
                "commits",
                "prs",
                "review_comments",
                "changed_lines",
                "avg_pr_duration",
                "avg_code_movement_per_pr" ,
                "commits_per_day" ,
                "changed_lines_per_day" ,
                "prs_per_day" ,
                "review_comments_per_day" ,
                "prs_diff_from_mean" , 
                "prs_ntile" ,
                "commits_ntile" ,
                "lines_of_code_ntile" ,
                "review_comments_ntile" ,
                "avg_pr_duration_ntile" ,
                "avg_ntile" 
            )
            VALUES (
                staging."repo",
                staging."contributor_name",
                staging."contributor_username",
                staging."curved_score" ,
                staging."stats_beginning" ,
                staging."stats_ending" ,
                staging."contributor_first_commit_date" ,
                staging."num_workdays" ,
                staging."commits",
                staging."prs",
                staging."review_comments",
                staging."changed_lines",
                staging."avg_pr_duration",
                staging."avg_code_movement_per_pr" ,
                staging."commits_per_day" ,
                staging."changed_lines_per_day" ,
                staging."prs_per_day" ,
                staging."review_comments_per_day" ,
                staging."prs_diff_from_mean" , 
                staging."prs_ntile" ,
                staging."commits_ntile" ,
                staging."lines_of_code_ntile" ,
                staging."review_comments_ntile" ,
                staging."avg_pr_duration_ntile" ,
                staging."avg_ntile"     
            );
        """

        cursor = conn.cursor()
        cursor.execute(merge_sql)
        conn.commit()
        rows_merged: int = cursor.rowcount
        print(f"Merged {rows_merged} into {target_table}")
        conn.close()

    def store_list_dict(self, list_dict_test: List[Dict[str, any]], table_name: str):
        """
        Inserts a list of dict 

        """
        conn = self.get_snowflake_connection()

        # Define a dummy row matching the contributor_stats table schema
        # Adjust the column names and dummy values according to your actual table schema

        df = pd.DataFrame(list_dict_test)
        self.store_df(df, table_name)
