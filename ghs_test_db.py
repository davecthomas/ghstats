from __future__ import annotations
from itertools import islice
from typing import Dict, List
import pandas as pd
from datetime import date
from datetime import datetime

from ghs_sentiment import GhsSentimentAnalyzer
from ghstats import GhsGithub


# def generate_data(num_rows: int) -> list:
#     list_to_generate: list = []
#     for row in range(1, num_rows):
#         dummy_row_data = {
#             'repo': 'example_repo',
#             'contributor_nodeid': f'ABCDEF123456789{row}',
#             'contributor_name': f'Name {row}',
#             'contributor_username': f'user{row}',
#             'curved_score': 88.5,
#             'stats_beginning': '2023-01-01',
#             'stats_ending': '2023-01-31',
#             'contributor_first_commit_date': '2022-05-15',
#             'num_workdays': 20,
#             'commits': 42,
#             'prs': f"{row}",
#             'review_comments': 5,
#             'changed_lines': 1500,
#             'avg_pr_duration': 2.5,
#             'avg_code_movement_per_pr': 300.0,
#             'commits_per_day': 2.1,
#             'changed_lines_per_day': 75.0,
#             'prs_per_day': 0.5,
#             'review_comments_per_day': 0.25,
#             'prs_diff_from_mean': 1.5,
#             'prs_ntile': 1,
#             'commits_ntile': 2,
#             'lines_of_code_ntile': 3,
#             'review_comments_ntile': 1,
#             'avg_pr_duration_ntile': 4,
#             'avg_ntile': 2
#         }
#         list_to_generate.append(dummy_row_data)
#     return list_to_generate


# def generate_test_data_contributors(num_rows: int) -> pd.DataFrame:
#     """
#     Generates a dummy dataset of contributors as a pandas DataFrame.

#     Parameters:
#     - num_rows (int): The number of dummy contributors to generate.

#     Returns:
#     - pd.DataFrame: A DataFrame containing dummy contributor data.
#     """
#     # Generate dummy data
#     contributor_nodeids = [f"node_{i}" for i in range(num_rows)]
#     contributor_names = [f"Contributor Name {i}" for i in range(num_rows)]
#     contributor_usernames = [f"user_{i}" for i in range(num_rows)]

#     # Create a DataFrame
#     df = pd.DataFrame({
#         "contributor_nodeid": contributor_nodeids,
#         "contributor_name": contributor_names,
#         "contributor_username": contributor_usernames,
#     })

#     return df


ghs: GhsGithub = GhsGithub()
list_repos: List[str] = ghs.dict_env.get("repo_names")
sentiment: GhsSentimentAnalyzer = GhsSentimentAnalyzer()
list_dict_comments: List[Dict[int, str]] = ghs.storage_manager.fetch_pr_comments_body(
    list_repos)
# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(list_dict_comments)
# Get the current date and format it as a string
date_str = datetime.now().strftime('%Y-%m-%d')

# Prepend the date string to the filename
output_csv_path_with_date = f"{date_str}_pr_comments.csv"
# Save the DataFrame to a CSV file
df.to_csv(output_csv_path_with_date, index=False)
# truncated_comments = sentiment.truncate_comments(list_dict_comments)
# # Calculate the length of the longest comment
# longest_comment_length = len(max(list_comments, key=len))
# print(
#     f"Longest comment length: {longest_comment_length} characters")
# list_token_counts: list[int] = sentiment.count_tokens(list_comments)
# longest_token_length: int = max(list_token_counts)
# print(
#     f"Token counts for the first 50 code review comments:\n{list_token_counts[:50]}. Longest token count: {longest_token_length} tokens.")
# sentiments: List[dict] = sentiment.evaluate_sentiments(truncated_comments)
# sentiment.output_to_training_csv(list_comments, sentiments)
# print(f"Sentiment analysis results for the first 50 code review comments:")
# i: int = 0
# for entry in islice(sentiments, 50):
#     print(f"{entry} {list_comments[i]}")
#     i += 1
# scores: List[float] = sentiment.summarize_sentiments(sentiments)
# scores_rounded = [round(num, 2) for num in scores]
# print(
#     f'Summary sentiment scores for code review comments in {list_repos}:\nPositive: {scores_rounded[0]} | Negative: {scores_rounded[1]} | Total: {scores_rounded[2]} | Normalized: {scores_rounded[3]}')
