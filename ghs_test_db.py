from ghs_snowflake import ContributorStatsStorageManager
import pandas as pd

storage_manager = ContributorStatsStorageManager()
conn = storage_manager.get_snowflake_connection()
df: pd.DataFrame = storage_manager.run_select_query(
    "select * from CONTRIBUTOR_STATS")
storage_manager.close_connection()
