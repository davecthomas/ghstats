from ghs_snowflake import ContributorStatsStorageManager
import pandas as pd


def generate_data(num_rows: int) -> list:
    list_to_generate: list = []
    for row in range(1, num_rows):
        dummy_row_data = {
            'repo': 'example_repo',
            'contributor_name': f'Name {row}',
            'contributor_username': f'user{row}',
            'curved_score': 88.5,
            'stats_beginning': '2023-01-01',
            'stats_ending': '2023-01-31',
            'contributor_first_commit_date': '2022-05-15',
            'num_workdays': 20,
            'commits': 42,
            'prs': f"{row}",
            'review_comments': 5,
            'changed_lines': 1500,
            'avg_pr_duration': 2.5,
            'avg_code_movement_per_pr': 300.0,
            'commits_per_day': 2.1,
            'changed_lines_per_day': 75.0,
            'prs_per_day': 0.5,
            'review_comments_per_day': 0.25,
            'prs_diff_from_mean': 1.5,
            'prs_ntile': 1,
            'commits_ntile': 2,
            'lines_of_code_ntile': 3,
            'review_comments_ntile': 1,
            'avg_pr_duration_ntile': 4,
            'avg_ntile': 2
        }
        list_to_generate.append(dummy_row_data)
    return list_to_generate


storage_manager = ContributorStatsStorageManager()
conn = storage_manager.get_snowflake_connection()
# df: pd.DataFrame = storage_manager.run_select_query(
#     "select * from CONTRIBUTOR_STATS")
# storage_manager.store_list_dict(generate_data(
#     5), storage_manager.get_db_env().get("snowflake_table_name_staging", ""))
storage_manager.upsert_dataframe(pd.DataFrame(generate_data(8)), storage_manager.get_db_env().get(
    "snowflake_table_name", ""), storage_manager.get_db_env().get("snowflake_table_name_staging", ""))
storage_manager.close_connection()
