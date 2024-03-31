from __future__ import annotations
from typing import List
import pandas as pd
from datetime import date

from ghstats import GhsGithub


def generate_data(num_rows: int) -> list:
    list_to_generate: list = []
    for row in range(1, num_rows):
        dummy_row_data = {
            'repo': 'example_repo',
            'contributor_nodeid': f'ABCDEF123456789{row}',
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


def generate_test_data_contributors(num_rows: int) -> pd.DataFrame:
    """
    Generates a dummy dataset of contributors as a pandas DataFrame.

    Parameters:
    - num_rows (int): The number of dummy contributors to generate.

    Returns:
    - pd.DataFrame: A DataFrame containing dummy contributor data.
    """
    # Generate dummy data
    contributor_nodeids = [f"node_{i}" for i in range(num_rows)]
    contributor_names = [f"Contributor Name {i}" for i in range(num_rows)]
    contributor_usernames = [f"user_{i}" for i in range(num_rows)]

    # Create a DataFrame
    df = pd.DataFrame({
        "contributor_nodeid": contributor_nodeids,
        "contributor_name": contributor_names,
        "contributor_username": contributor_usernames,
    })

    return df


ghs: GhsGithub = GhsGithub()
ghs.prep_repo_topics()
list_repos: List[str] = ghs.dict_env.get("repo_names")
ghs.fetch_and_store_pr_review_comments(list_repos)
