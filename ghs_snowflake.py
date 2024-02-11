from typing import List, Dict
import os
import pandas as pd
import snowflake.connector
from typing import Dict, Optional, List
from snowflake.connector.pandas_tools import write_pandas


class GhsSnowflakeStorageManager:
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
                "snowflake_table_name": os.getenv("SNOWFLAKE_TABLE_NAME"),
                "snowflake_table_name_contributors": os.getenv("SNOWFLAKE_TABLE_NAME_CONTRIBUTORS")
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

    def fetch_existing_repo_topics(self) -> List[Dict[str, any]]:
        """
        Fetches existing records from the 'repo_topics' table.

        Returns:
            List of dictionaries where each dictionary represents a record from the 'repo_topics' table.
        """
        conn = self.get_snowflake_connection()
        cursor = conn.cursor()

        cursor.execute(
            'SELECT "repo_name", "topic" FROM "repo_topics"')
        existing_records = cursor.fetchall()

        # Convert query results to a list of dictionaries
        existing_repo_topics: list = [
            {"repo_name": repo_name, "topic": topic}
            for repo_name, topic in existing_records
        ]

        cursor.close()
        conn.close()

        return existing_repo_topics

    def repo_topic_dict_to_dataframe(self, source_dict: Dict[str, List[str]]) -> pd.DataFrame:
        """Converts a dictionary to a pandas DataFrame."""
        # Flatten the dictionary into a list of dictionaries for easy DataFrame conversion
        data = [{"repo_name": repo_name, "repo_topic": topic}
                for repo_name, topics in source_dict.items()
                for topic in topics]
        df = pd.DataFrame(data)
        return df

    def store_repo_topics(self, dict_repo_topics: dict) -> int:
        """
        Inserts new records into the 'repo-topics' table from a DataFrame, excluding existing records.

        Parameters:
            df (pd.DataFrame): DataFrame containing the repo-topics data.
        Returns: 
            count of inserted rows
        """
        df: pd.DataFrame = self.repo_topic_dict_to_dataframe(dict_repo_topics)

        # Query to check existence of a row
        check_query = """
        SELECT COUNT(*) FROM "repo_topics"
        WHERE "repo_name" = %s AND "repo_topic" = %s;
        """

        # Insert query
        insert_query = """
        INSERT INTO "repo_topics" ("repo_name", "repo_topic")
        VALUES (%s, %s);
        """
        conn = self.get_snowflake_connection()
        cursor = conn.cursor()
        for _, row in df.iterrows():
            # Check if the row already exists
            cursor.execute(check_query, (row['repo_name'], row['repo_topic']))
            result = cursor.fetchone()
            if result[0] == 0:  # If the row does not exist
                # Insert the new row
                cursor.execute(
                    insert_query, (row['repo_name'], row['repo_topic']))
                print(f"Inserted: {row['repo_name']}, {row['repo_topic']}")

        # Commit transactions
        conn.commit()

        # Close connection
        cursor.close()
        conn.close()

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
                INSERT INTO "{target_table}" (
                    "repo",
                    "contributor_nodeid",
                    "contributor_name",
                    "contributor_username",
                    "curved_score",
                    "stats_beginning",
                    "stats_ending",
                    "contributor_first_commit_date",
                    "num_workdays",
                    "commits",
                    "prs",
                    "review_comments",
                    "changed_lines",
                    "avg_pr_duration",
                    "avg_code_movement_per_pr",
                    "commits_per_day",
                    "changed_lines_per_day",
                    "prs_per_day",
                    "review_comments_per_day",
                    "prs_diff_from_mean",
                    "prs_ntile",
                    "commits_ntile",
                    "lines_of_code_ntile",
                    "review_comments_ntile",
                    "avg_pr_duration_ntile",
                    "avg_ntile"
                )
                SELECT
                    staging."repo",
                    staging."contributor_nodeid",
                    staging."contributor_name",
                    staging."contributor_username",
                    staging."curved_score",
                    staging."stats_beginning",
                    staging."stats_ending",
                    staging."contributor_first_commit_date",
                    staging."num_workdays",
                    staging."commits",
                    staging."prs",
                    staging."review_comments",
                    staging."changed_lines",
                    staging."avg_pr_duration",
                    staging."avg_code_movement_per_pr",
                    staging."commits_per_day",
                    staging."changed_lines_per_day",
                    staging."prs_per_day",
                    staging."review_comments_per_day",
                    staging."prs_diff_from_mean",
                    staging."prs_ntile",
                    staging."commits_ntile",
                    staging."lines_of_code_ntile",
                    staging."review_comments_ntile",
                    staging."avg_pr_duration_ntile",
                    staging."avg_ntile"
                FROM
                    "{staging_table}" AS staging
                LEFT JOIN
                    "{target_table}" AS existing
                ON
                    staging."contributor_name" = existing."contributor_name"
                    AND staging."repo" = existing."repo"
                    AND staging."stats_beginning" = existing."stats_beginning"
                WHERE
                    existing."contributor_name" IS NULL;


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

    def fetch_existing_contributors(self) -> List[Dict[str, any]]:
        """
        Fetches existing records from the 'contributors' table.

        Returns:
            List of dictionaries where each dictionary represents a record from the 'contributors' table.
        """
        conn = self.get_snowflake_connection()
        cursor = conn.cursor()

        cursor.execute(
            'SELECT "contributor_nodeid", "contributor_name", "contributor_username" FROM "contributors"')
        existing_records = cursor.fetchall()

        # Convert query results to a list of dictionaries
        existing_contributors = [
            {"contributor_nodeid": nodeid, "contributor_name": name,
                "contributor_username": username}
            for nodeid, name, username in existing_records
        ]

        cursor.close()
        conn.close()

        return existing_contributors

    def fetch_existing_contributors(self) -> List[Dict[str, any]]:
        """
        Fetches existing records from the 'contributors' table.

        Returns:
            List of dictionaries where each dictionary represents a record from the 'contributors' table.
        """
        conn = self.get_snowflake_connection()
        cursor = conn.cursor()

        cursor.execute(
            'SELECT "contributor_nodeid", "contributor_name", "contributor_username" FROM "contributors"')
        existing_records = cursor.fetchall()

        # Convert query results to a list of dictionaries
        existing_contributors = [
            {"contributor_nodeid": nodeid, "contributor_name": name,
                "contributor_username": username}
            for nodeid, name, username in existing_records
        ]

        cursor.close()
        conn.close()

        return existing_contributors

    def insert_new_contributors(self, df: pd.DataFrame) -> int:
        """
        Inserts new records into the 'contributors' table from a DataFrame, excluding existing records.

        Parameters:
            df (pd.DataFrame): DataFrame containing the contributors data.
        Returns: 
            count of inserted rows
        """
        # Fetch existing contributors
        existing_contributors = self.fetch_existing_contributors()
        existing_nodeids = {contributor['contributor_nodeid']
                            for contributor in existing_contributors}

        # Establish connection to Snowflake
        conn = self.get_snowflake_connection()
        cursor = conn.cursor()
        count: int = 0

        for index, row in df.iterrows():
            # Check if the record exists
            if row['contributor_nodeid'] not in existing_nodeids:
                count += 1
                # Insert new record
                cursor.execute("""
                    INSERT INTO "contributors" ("contributor_nodeid", "contributor_name", "contributor_username")
                    VALUES (%s, %s, %s)
                """, (row['contributor_nodeid'], row['contributor_name'], row['contributor_username']))

        # Commit the transaction
        conn.commit()

        # Cleanup
        cursor.close()
        conn.close()

        print(f"{count} contributors saved.")
        return (count)

    def upsert_contributors(self, df: pd.DataFrame) -> None:
        # Fetch existing contributors and convert to a DataFrame
        existing_contributors_list: [] = self.fetch_existing_contributors()
        if len(existing_contributors_list) == 0:
            self.store_df(df, self.get_db_env().get(
                "snowflake_table_name_contributors", ""))

        else:
            existing_contributors_df: pd.DataFrame = pd.DataFrame(
                existing_contributors_list)

            # Remove records we already have
            new_records_df = df[~df['contributor_nodeid'].isin(
                existing_contributors_df['contributor_nodeid'])]

            # Insert new contributors
            count: int = 0
            if not new_records_df.empty:
                count = self.insert_new_contributors(new_records_df)

        return
