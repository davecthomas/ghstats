import os
import requests
import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Define API endpoints
base_url = "https://api.github.com"
repos_endpoint = "/repos/{owner}/{repo}"
commits_endpoint = "/repos/{owner}/{repo}/commits"
pulls_endpoint = "/repos/{owner}/{repo}/pulls"

# Define date range for statistics calculation
start_date = datetime.date(2022, 1, 1)
end_date = datetime.date(2022, 12, 31)

# Define function to calculate number of workdays between two dates
def workdays(start_date, end_date):
    days = (end_date - start_date).days + 1
    return sum(1 for day in range(days) if datetime.date.weekday(start_date + datetime.timedelta(day)) < 5)

# Define function to calculate statistics per contributor per repo
def calculate_statistics(users, repos):
    # Initialize dictionary to store statistics
    statistics = {}

    # Loop through each user and repo combination
    for user in users:
        for repo in repos:
            # Make API requests to retrieve data
            repo_url = base_url + repos_endpoint.format(owner=user, repo=repo)
            repo_data = requests.get(repo_url, headers={"Authorization": f"token {os.environ.get('GITHUB_API_TOKEN')}"}).json()
            commits_url = base_url + commits_endpoint.format(owner=user, repo=repo)
            commits_data = requests.get(commits_url, headers={"Authorization": f"token {os.environ.get('GITHUB_API_TOKEN')}"}).json()
            pulls_url = base_url + pulls_endpoint.format(owner=user, repo=repo)
            pulls_data = requests.get(pulls_url, headers={"Authorization": f"token {os.environ.get('GITHUB_API_TOKEN')}"}).json()

            # Initialize dictionary to store statistics for current user and repo combination
            user_stats = {
                "commits_per_workday": 0,
                "prs_per_workday": 0,
                "review_comments_per_workday": 0,
                "lines_of_code_per_workday": 0
            }

            # Loop through each commit and calculate statistics
            for commit in commits_data:
                commit_date = datetime.datetime.strptime(commit["commit"]["author"]["date"], "%Y-%m-%dT%H:%M:%SZ").date()
                if start_date <= commit_date <= end_date:
                    commit_workdays = workdays(start_date, commit_date)
                    user_stats["commits_per_workday"] += 1 / commit_workdays
                    user_stats["lines_of_code_per_workday"] += commit["stats"]["additions"] / commit_workdays

            # Loop through each pull request and calculate statistics
            for pull in pulls_data:
                pull_date = datetime.datetime.strptime(pull["created_at"], "%Y-%m-%dT%H:%M:%SZ").date()
                if start_date <= pull_date <= end_date:
                    pull_workdays = workdays(start_date, pull_date)
                    user_stats["prs_per_workday"] += 1 / pull_workdays

                    # Make API request to retrieve review comments for pull request
                    review_comments_url = pull["comments_url"]
                    review_comments_data = requests.get(review_comments
