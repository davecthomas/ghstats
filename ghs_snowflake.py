from typing import List, Dict, Set, Tuple
import os
from datetime import date
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

    def store_dataframe(df: pd.DataFrame, table_name: str):
        pass

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
        rows_merged: int = 0
        for _, row in df.iterrows():
            # Check if the row already exists
            cursor.execute(check_query, (row['repo_name'], row['repo_topic']))
            result = cursor.fetchone()
            if result[0] == 0:  # If the row does not exist
                # Insert the new row
                cursor.execute(
                    insert_query, (row['repo_name'], row['repo_topic']))
                rows_merged += 1
                # print(f"Inserted: {row['repo_name']}, {row['repo_topic']}")

        # Commit transactions
        conn.commit()

        # Close connection
        cursor.close()
        conn.close()
        print(f"Inserted {rows_merged} repo topics.")
        return rows_merged

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

    def upsert_contributor_stats_dataframe(self, df: pd.DataFrame, target_table: str, staging_table: str) -> int:
        """
        Upserts data from a DataFrame into the target table using a staging table.
        This is to avoid us having duplicate records when we re-run similar timeframes for the same
        repos. 

        :param df: DataFrame to upsert.
        :param target_table: Name of the target table for upsert.
        :param staging_table: Name of the staging table for initial DataFrame upload.
        :return: Number of rows merged into the target table or None on error.
        """
        if df.empty:
            print("DataFrame is empty. Skipping upsert.")
            return None
        conn = self.get_snowflake_connection()
        # Prereq: this assume staging table is clear!
        self.delete_all_rows(staging_table)

        # Step 1: Upload DataFrame to a staging table
        # Avoid a warning due to a non-standard index in the dataframe
        df.reset_index(drop=True, inplace=True)
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
                    "median_pr_review_duration",
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
                    staging."median_pr_review_duration",
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
        print(
            f"\tStored {rows_merged} into {target_table} of {len(df)} potential rows.")
        conn.close()
        return rows_merged

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
        self.close_connection()

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
        self.close_connection()

        print(f"{count} contributors saved.")
        return (count)

    def upsert_contributors(self, df: pd.DataFrame) -> None:
        # Fetch existing contributors and convert to a DataFrame
        count: int = 0
        existing_contributors_list: List = self.fetch_existing_contributors()
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
            if not new_records_df.empty:
                count = self.insert_new_contributors(new_records_df)

        return count

    def get_existing_repo_topics(self) -> Set[Tuple[str, str]]:
        """
        Retrieves all existing repo topics from Snowflake.

        Returns:
            A set of tuples, each containing (repo_name, repo_topic).
        """
        query = """SELECT "repo_name", "repo_topic" FROM "repo_topics";"""
        existing_topics: Set[Tuple[str, str]] = set()
        conn = self.get_snowflake_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            for row in cursor.fetchall():
                # Adding a tuple of repo_name and repo_topic
                existing_topics.add((row[0], row[1]))
        finally:
            cursor.close()
            self.close_connection()
        return existing_topics

    def insert_new_repo_topics(self, new_topics_df: pd.DataFrame) -> int:
        """
        Inserts new repo topics into Snowflake, avoiding duplicates.

        Args:
            new_topics_df (pd.DataFrame): DataFrame containing new repo topics with columns 'repo_name' and 'repo_topic'.
        Returns: count of inserted records
        """
        existing_topics: Set[Tuple[str, str]] = self.get_existing_repo_topics(
        )  # Retrieve existing topics
        new_records: List[Tuple[str, str]] = []
        inserted: int = 0

        for index, row in new_topics_df.iterrows():
            if (row['repo_name'], row['repo_topic']) not in existing_topics:
                new_records.append((row['repo_name'], row['repo_topic']))

        # Convert new records to DataFrame for insertion
        if new_records:
            df_insert: pd.DataFrame = pd.DataFrame(
                new_records, columns=['repo_name', 'repo_topic'])
            conn = self.get_snowflake_connection()
            try:
                write_pandas(conn, df_insert, 'repo_topics')
                inserted = len(df_insert)
                # print(f"Inserted {len(df_insert)} new repo topics.")
            finally:
                self.close_connection()
        return inserted

    def store_repo_stats(self, list_dict_repo_stats: List) -> int:
        """
        Inserts new repo stats into Snowflake, avoiding duplicates.

        Args:
            list_dict_repo_stats (List): List of dictionaries containing new repo stats.
        Returns: count of inserted records
        """
        inserted_records = 0
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

        for record in list_dict_repo_stats:
            try:
                # Check if the record already exists
                cursor.execute(
                    check_sql, (record['repo_name'], record['stats_beginning']))
                result = cursor.fetchone()
            except Exception as e:
                print(
                    f"Error checking existence for {record['repo_name']}: {e}")
                continue  # Skip this record and move to the next

            if result[0] == 0:  # Record does not exist, proceed to insert
                try:
                    cursor.execute(insert_sql, (
                        record['repo_name'], record['stats_beginning'],
                        record['stats_ending'], record['num_workdays'],
                        record['num_contributors'], record['avg_pr_duration'],
                        record['median_pr_duration'], record['num_prs'], record['num_commits']))
                    conn.commit()
                    inserted_records += 1
                except Exception as e:
                    print(
                        f"Error inserting record for {record['repo_name']}: {e}")

        self.close_connection()
        print(
            f"Successfully inserted {inserted_records} new records into repo_stats out of {len(list_dict_repo_stats)} attempted.")
        return inserted_records

    def insert_pr_review_comments(self, df_review_comments: pd.DataFrame) -> int:
        """
        Inserts or updates PR review comments into the 'pr_review_comments' table in Snowflake.
        Uses a staging table and merge operation to prevent duplicates. Clears the staging table first.
        This function correctly handles lowercase column names by quoting them in SQL queries.

        Args:
            df_review_comments (pd.DataFrame): DataFrame containing PR review comments.

        Returns:
            int: Number of rows successfully upserted.
        """
        if df_review_comments.empty:
            # print("DataFrame is empty. No operation performed.")
            return 0

        staging_table_name = "pr_review_comments_staging"
        target_table_name = "pr_review_comments"

        # Ensure the Snowflake connection is established
        conn = self.get_snowflake_connection()

        try:
            cursor = conn.cursor()

            # Step 0: Clear the staging table first
            delete_sql = f'DELETE FROM "{staging_table_name}";'
            cursor.execute(delete_sql)
            conn.commit()
            # print(f"Cleared the staging table '{staging_table_name}'.")

            # Step 1: Insert DataFrame into the staging table
            # Note: write_pandas automatically handles the DataFrame to Snowflake insertion,
            # including the case sensitivity of column names.
            write_pandas(conn, df_review_comments, staging_table_name)

            # Step 2: Merge from staging table to target table based on 'comment_id'
            # Make sure to quote the lowercase column names in your SQL merge statement
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

    def fetch_pr_comments_body(self, repo_names: List[str], date_since: date = None, date_until: date = None, limit: int = -1) -> List[str]:
        """
        Fetches the body of PR comments for specified repositories within an optional date range from Snowflake.
        Can limit the number of results returned.

        Args:
            repo_names (List[str]): Names of the repositories.
            date_since (date, optional): Start date for filtering comments. Defaults to None.
            date_until (date, optional): End date for filtering comments. Defaults to None.
            limit (int, optional): Maximum number of PR comment bodies to fetch. Defaults to 250. Use -1 for no limit.

        Returns:
            List[str]: A list containing the body of each PR comment.
        """
        if not repo_names:
            return []

        conn = self.get_snowflake_connection()
        cursor = conn.cursor()

        placeholders = ', '.join(['%s' for _ in repo_names])

        query = f"""
            SELECT "body" FROM "pr_review_comments"
            WHERE "repo_name" IN ({placeholders})
        """

        query_conditions = repo_names

        if date_since:
            query += " AND \"created_at\" >= %s"
            query_conditions.append(date_since.strftime('%Y-%m-%d'))
        if date_until:
            query += " AND \"created_at\" <= %s"
            query_conditions.append(date_until.strftime('%Y-%m-%d'))

        # Adding limit clause to the SQL query if limit is not -1
        if limit != -1:
            query += f" LIMIT {limit}"

        try:
            cursor.execute(query, query_conditions)
            records = cursor.fetchall()
            pr_comments_body = [record[0] for record in records]

        finally:
            cursor.close()
            self.close_connection()

        return pr_comments_body
