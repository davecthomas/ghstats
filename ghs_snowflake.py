import os
import time
from datetime import date
from typing import List, Dict, Optional, Tuple, Set
import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas


class GhsSnowflakeStorageManager:
    def __init__(self):
        self.dict_db_env = None
        self.conn: Optional[snowflake.connector.SnowflakeConnection] = None
        self.get_db_env()
        self.backoff_delays = [1, 2, 4, 8, 16]  # Delays in seconds

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

    def get_snowflake_connection(self) -> snowflake.connector.SnowflakeConnection:
        """Establishes a connection to Snowflake with hardcoded backoff delay."""
        if self.conn is None or self.conn.is_closed():
            dict_db_env = self.get_db_env()
            for attempt, delay in enumerate(self.backoff_delays, 1):
                try:
                    self.conn = snowflake.connector.connect(
                        user=dict_db_env["snowflake_user"],
                        password=dict_db_env["snowflake_password"],
                        account=dict_db_env["snowflake_account"],
                        warehouse=dict_db_env["snowflake_warehouse"],
                        database=dict_db_env["snowflake_db"],
                        schema=dict_db_env["snowflake_schema"],
                        timeout=30  # Set a timeout for connection
                    )
                    break
                except snowflake.connector.errors.OperationalError as e:
                    print(
                        f"Connection attempt {attempt} failed: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
            if self.conn is None or self.conn.is_closed():
                raise Exception(
                    "Could not connect to Snowflake after multiple attempts.")
        return self.conn

    # Other methods remain the same...

    def fetch_existing_repo_topics(self) -> List[Dict[str, any]]:
        """
        Fetches existing records from the 'repo_topics' table.
        Returns: List of dictionaries where each dictionary represents a record from the 'repo_topics' table.
        """
        conn = self.get_snowflake_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('SELECT "repo_name", "topic" FROM "repo_topics"')
            existing_records = cursor.fetchall()
            existing_repo_topics = [
                {"repo_name": repo_name, "topic": topic} for repo_name, topic in existing_records]
        except Exception as e:
            print(f"Error fetching existing repo topics: {e}")
            existing_repo_topics = []
        finally:
            cursor.close()
            self.close_connection()
        return existing_repo_topics

    def repo_topic_dict_to_dataframe(self, source_dict: Dict[str, List[str]]) -> pd.DataFrame:
        """Converts a dictionary to a pandas DataFrame."""
        data = [{"repo_name": repo_name, "repo_topic": topic}
                for repo_name, topics in source_dict.items() for topic in topics]
        return pd.DataFrame(data)

    def store_repo_topics(self, dict_repo_topics: dict) -> int:
        """
        Inserts new records into the 'repo-topics' table from a DataFrame, excluding existing records.
        Parameters: df (pd.DataFrame): DataFrame containing the repo-topics data.
        Returns: count of inserted rows
        """
        df: pd.DataFrame = self.repo_topic_dict_to_dataframe(dict_repo_topics)
        check_query = """
        SELECT COUNT(*) FROM "repo_topics"
        WHERE "repo_name" = %s AND "repo_topic" = %s;
        """
        insert_query = """
        INSERT INTO "repo_topics" ("repo_name", "repo_topic")
        VALUES (%s, %s);
        """
        conn = self.get_snowflake_connection()
        cursor = conn.cursor()
        rows_merged: int = 0
        for _, row in df.iterrows():
            try:
                cursor.execute(
                    check_query, (row['repo_name'], row['repo_topic']))
                result = cursor.fetchone()
                if result[0] == 0:
                    cursor.execute(
                        insert_query, (row['repo_name'], row['repo_topic']))
                    rows_merged += 1
            except Exception as e:
                print(
                    f"Error inserting repo topic {row['repo_name'], row['repo_topic']}: {e}")
        conn.commit()
        cursor.close()
        self.close_connection()
        print(f"Inserted {rows_merged} repo topics.")
        return rows_merged

    def run_select_query(self, query: str) -> pd.DataFrame:
        """Executes a SELECT query and returns the results as a pandas DataFrame."""
        conn = self.get_snowflake_connection()
        try:
            cursor = conn.cursor().execute(query)
            df = pd.DataFrame(cursor.fetchall(), columns=[
                              col[0] for col in cursor.description])
        except Exception as e:
            print(f"Error executing select query: {e}")
            df = pd.DataFrame()
        finally:
            self.close_connection()
        return df

    def save_df_to_csv(self, df: pd.DataFrame, filename: str):
        """Saves the DataFrame to a CSV file."""
        df.to_csv(filename, index=False)

    def save_summary_stats_csv(self, df: pd.DataFrame, filename: str):
        """Saves summary statistics of the DataFrame to a CSV file."""
        summary = df.describe()
        summary.to_csv(f"summary_{filename}")

    def store_df(self, df: pd.DataFrame, table_name: str):
        """Stores the DataFrame in a Snowflake table."""
        conn = self.get_snowflake_connection()
        try:
            success, nchunks, nrows, _ = write_pandas(conn, df, table_name)
            print(
                f"Data stored in Snowflake table {table_name}: {nrows} rows in {nchunks} chunks.")
        except Exception as e:
            print(f"Error storing DataFrame in Snowflake: {e}")
        finally:
            self.close_connection()

    def delete_all_rows(self, staging_table: str):
        """Deletes all rows from the specified staging table."""
        conn = self.get_snowflake_connection()
        try:
            conn.cursor().execute(f'DELETE FROM "{staging_table}";')
        except Exception as e:
            print(
                f"Error deleting rows from staging table {staging_table}: {e}")
        finally:
            self.close_connection()

    def upsert_contributor_stats_dataframe(self, df: pd.DataFrame, target_table: str, staging_table: str) -> int:
        """
        Upserts data from a DataFrame into the target table using a staging table.
        :param df: DataFrame to upsert.
        :param target_table: Name of the target table for upsert.
        :param staging_table: Name of the staging table for initial DataFrame upload.
        :return: Number of rows merged into the target table or None on error.
        """
        if df.empty:
            print("DataFrame is empty. Skipping upsert.")
            return 0

        try:
            self.delete_all_rows(staging_table)

            self.conn = self.get_snowflake_connection()

            df.reset_index(drop=True, inplace=True)
            success, nchunks, nrows, _ = write_pandas(
                self.conn, df, staging_table)
            print(
                f"DataFrame uploaded successfully: {nrows} rows in {nchunks} chunks.")
            merge_sql = f"""
                INSERT INTO "{target_table}" (
                    "repo", "contributor_nodeid", "contributor_name", "contributor_username", 
                    "curved_score", "stats_beginning", "stats_ending", 
                    "contributor_first_commit_date", "num_workdays", "commits", 
                    "prs", "review_comments", "changed_lines", "avg_pr_duration", 
                    "avg_code_movement_per_pr", "commits_per_day", "changed_lines_per_day", 
                    "prs_per_day", "review_comments_per_day", "prs_diff_from_mean", 
                    "prs_ntile", "commits_ntile", "lines_of_code_ntile", 
                    "review_comments_ntile", "avg_pr_duration_ntile", "avg_ntile", 
                    "median_pr_review_duration"
                )
                SELECT
                    staging."repo", staging."contributor_nodeid", staging."contributor_name", 
                    staging."contributor_username", staging."curved_score", staging."stats_beginning", 
                    staging."stats_ending", staging."contributor_first_commit_date", 
                    staging."num_workdays", staging."commits", staging."prs", staging."review_comments", 
                    staging."changed_lines", staging."avg_pr_duration", staging."avg_code_movement_per_pr", 
                    staging."commits_per_day", staging."changed_lines_per_day", staging."prs_per_day", 
                    staging."review_comments_per_day", staging."prs_diff_from_mean", staging."prs_ntile", 
                    staging."commits_ntile", staging."lines_of_code_ntile", staging."review_comments_ntile", 
                    staging."avg_pr_duration_ntile", staging."avg_ntile", staging."median_pr_review_duration"
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
            cursor = self.conn.cursor()
            cursor.execute(merge_sql)
            self.conn.commit()
            rows_merged: int = cursor.rowcount
            print(
                f"Stored {rows_merged} into {target_table} of {len(df)} potential rows.")
        except snowflake.connector.Error as e:
            print(f"An error occurred storing contributor stats: {e}")
            rows_merged = 0
        finally:
            self.close_connection()
        return rows_merged

    def store_list_dict(self, list_dict_test: List[Dict[str, any]], table_name: str):
        """Inserts a list of dicts into the specified table."""
        df = pd.DataFrame(list_dict_test)
        self.store_df(df, table_name)

    def fetch_existing_contributors(self) -> List[Dict[str, any]]:
        """Fetches existing records from the 'contributors' table."""
        conn = self.get_snowflake_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                'SELECT "contributor_nodeid", "contributor_name", "contributor_username" FROM "contributors"')
            existing_records = cursor.fetchall()
            existing_contributors = [{"contributor_nodeid": nodeid, "contributor_name": name,
                                      "contributor_username": username} for nodeid, name, username in existing_records]
        except Exception as e:
            print(f"Error fetching existing contributors: {e}")
            existing_contributors = []
        finally:
            cursor.close()
            self.close_connection()
        return existing_contributors

    def insert_new_contributors(self, df: pd.DataFrame) -> int:
        """Inserts new records into the 'contributors' table from a DataFrame, excluding existing records."""
        existing_contributors = self.fetch_existing_contributors()
        existing_nodeids = {contributor['contributor_nodeid']
                            for contributor in existing_contributors}
        conn = self.get_snowflake_connection()
        cursor = conn.cursor()
        count = 0
        try:
            for _, row in df.iterrows():
                if row['contributor_nodeid'] not in existing_nodeids:
                    cursor.execute("""
                        INSERT INTO "contributors" ("contributor_nodeid", "contributor_name", "contributor_username")
                        VALUES (%s, %s, %s)
                    """, (row['contributor_nodeid'], row['contributor_name'], row['contributor_username']))
                    count += 1
            conn.commit()
        except Exception as e:
            print(f"Error inserting new contributors: {e}")
        finally:
            cursor.close()
            self.close_connection()
        print(f"{count} contributors saved.")
        return count

    def upsert_contributors(self, df: pd.DataFrame) -> int:
        """Upserts contributors into the 'contributors' table."""
        existing_contributors_list = self.fetch_existing_contributors()
        if not existing_contributors_list:
            self.store_df(df, self.get_db_env().get(
                "snowflake_table_name_contributors", ""))
            return len(df)
        else:
            existing_contributors_df = pd.DataFrame(existing_contributors_list)
            new_records_df = df[~df['contributor_nodeid'].isin(
                existing_contributors_df['contributor_nodeid'])]
            if not new_records_df.empty:
                return self.insert_new_contributors(new_records_df)
        return 0

    def get_existing_repo_topics(self) -> Set[Tuple[str, str]]:
        """Retrieves all existing repo topics from Snowflake."""
        query = """SELECT "repo_name", "repo_topic" FROM "repo_topics";"""
        existing_topics = set()
        conn = self.get_snowflake_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            for row in cursor.fetchall():
                existing_topics.add((row[0], row[1]))
        except Exception as e:
            print(f"Error fetching existing repo topics: {e}")
        finally:
            cursor.close()
            self.close_connection()
        return existing_topics

    def insert_new_repo_topics(self, new_topics_df: pd.DataFrame) -> int:
        """Inserts new repo topics into Snowflake, avoiding duplicates."""
        existing_topics = self.get_existing_repo_topics()
        new_records = [(row['repo_name'], row['repo_topic']) for _, row in new_topics_df.iterrows(
        ) if (row['repo_name'], row['repo_topic']) not in existing_topics]
        if new_records:
            df_insert = pd.DataFrame(new_records, columns=[
                                     'repo_name', 'repo_topic'])
            conn = self.get_snowflake_connection()
            try:
                write_pandas(conn, df_insert, 'repo_topics')
                return len(df_insert)
            except Exception as e:
                print(f"Error inserting new repo topics: {e}")
            finally:
                self.close_connection()
        return 0

    def store_repo_stats(self, list_dict_repo_stats: List) -> int:
        """Inserts new repo stats into Snowflake, avoiding duplicates."""
        conn = self.get_snowflake_connection()
        cursor = conn.cursor()
        check_sql = """
            SELECT COUNT(*) FROM "repo_stats"
            WHERE "repo_name" = %s AND "stats_beginning" = %s
        """
        insert_sql = """
            INSERT INTO "repo_stats" ("repo_name", "stats_beginning", 
            "stats_ending", "num_workdays", "num_contributors",
            "avg_pr_duration", "median_pr_duration", "num_prs", "num_commits")
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        inserted_records = 0
        for record in list_dict_repo_stats:
            try:
                cursor.execute(
                    check_sql, (record['repo_name'], record['stats_beginning']))
                result = cursor.fetchone()
                if result[0] == 0:
                    cursor.execute(insert_sql, (
                        record['repo_name'], record['stats_beginning'], record['stats_ending'],
                        record['num_workdays'], record['num_contributors'],
                        record['avg_pr_duration'], record['median_pr_duration'],
                        record['num_prs'], record['num_commits']
                    ))
                    conn.commit()
                    inserted_records += 1
            except Exception as e:
                print(
                    f"Error inserting repo stats for {record['repo_name']}: {e}")
        self.close_connection()
        print(
            f"Successfully inserted {inserted_records} new records into repo_stats out of {len(list_dict_repo_stats)} attempted.")
        return inserted_records

    def insert_pr_review_comments(self, df_review_comments: pd.DataFrame) -> int:
        """Inserts or updates PR review comments into the 'pr_review_comments' table in Snowflake."""
        if df_review_comments.empty:
            return 0
        staging_table_name = "pr_review_comments_staging"
        target_table_name = "pr_review_comments"
        conn = self.get_snowflake_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(f'DELETE FROM "{staging_table_name}";')
            conn.commit()
            write_pandas(conn, df_review_comments, staging_table_name)
            merge_sql = f"""
                MERGE INTO "{target_table_name}" AS target
                USING "{staging_table_name}" AS staging
                ON target."comment_id" = staging."comment_id"
                WHEN MATCHED THEN
                    UPDATE SET
                    target."repo_name" = staging."repo_name",
                    target."pr_number" = staging."pr_number",
                    target."user_login" = staging."user_login",
                    target."body" = staging."body",
                    target."created_at" = staging."created_at"
                WHEN NOT MATCHED THEN
                    INSERT ("comment_id", "repo_name", "pr_number", "user_login", "body", "created_at")
                    VALUES (staging."comment_id", staging."repo_name", staging."pr_number", staging."user_login", staging."body", staging."created_at");
            """
            cursor.execute(merge_sql)
            nrows = cursor.rowcount
            conn.commit()
            print(
                f"Successfully upserted {nrows} PR review comments into '{target_table_name}'.")
            return nrows
        except Exception as e:
            print(
                f"Failed to upsert PR review comments into Snowflake. Error: {e}")
            return 0
        finally:
            cursor.close()
            self.close_connection()

    def fetch_pr_comments_body(self, repo_names: List[str], date_since: date = None, date_until: date = None, limit: int = -1) -> List[Dict[int, str]]:
        """Fetches the comment_id and body of PR comments for specified repositories within an optional date range from Snowflake."""
        if not repo_names:
            return []
        conn = self.get_snowflake_connection()
        cursor = conn.cursor()
        placeholders = ', '.join(['%s' for _ in repo_names])
        query = f"""
            SELECT "comment_id", "body" FROM "pr_review_comments"
            WHERE "repo_name" IN ({placeholders})
        """
        query_conditions = repo_names.copy()
        if date_since:
            query += ' AND "created_at" >= %s'
            query_conditions.append(date_since.strftime('%Y-%m-%d'))
        if date_until:
            query += ' AND "created_at" <= %s'
            query_conditions.append(date_until.strftime('%Y-%m-%d'))
        if limit != -1:
            query += f' LIMIT {limit}'
        try:
            cursor.execute(query, query_conditions)
            records = cursor.fetchall()
            pr_comments = [{record[0]: record[1]} for record in records]
        finally:
            cursor.close()
            self.close_connection()
        return pr_comments
