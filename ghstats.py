import json
from datetime import datetime
from datetime import timedelta, date
import os
from typing import List, Tuple
import pandas as pd
import numpy as np
import time
import requests

GITHUB_API_BASE_URL = "https://api.github.com"
DEFAULT_MONTHS_LOOKBACK = int(os.getenv("DEFAULT_MONTHS_LOOKBACK", 0))
API_TOKEN = os.getenv("GITHUB_API_TOKEN")


def get_github_collaborator_name(username):
    """
    Returns the name of a collaborator on a GitHub repository, given their username.

    Parameters:
    owner (str): The username of the owner of the repository.
    repo (str): The name of the repository.
    username (str): The username of the collaborator whose name to retrieve.
    access_token (str): A personal access token with appropriate permissions to access the collaborator's information.

    Returns:
    str: The name of the collaborator, or None if the collaborator is not found or there is an error retrieving their information.
    """
    url = f"https://api.github.com/users/{username}"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"Bearer {API_TOKEN}"
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        collaborator_info = json.loads(response.text)
        collaborator_name = collaborator_info['name']
        return collaborator_name
    else:
        return None


def get_commenters_stats(repo_owner, repo_name, months_lookback):
    # calculate start and end dates
    today = datetime.now().date()
    start_date = today - timedelta(days=months_lookback*30)
    end_date = today + timedelta(days=1)

    # get all pull requests for the repo within the lookback period
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls?state=closed&since={start_date}&until={end_date}"
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    response = requests.get(url, headers=headers)

    pull_requests = response.json()

    # get all comments for each pull request and add up the number of comments per commenter
    commenters = {}
    for pr in pull_requests:
        url = pr['comments_url']
        response = requests.get(url, headers=headers)
        comments = response.json()
        for comment in comments:
            commenter_name = comment['user']['login']
            if commenter_name not in commenters:
                commenters[commenter_name] = 1
            else:
                commenters[commenter_name] += 1

    # convert commenters dictionary to list of dictionaries and sort by number of comments
    commenters_list = [{'commenter_name': k, 'num_comments': v}
                       for k, v in commenters.items()]
    commenters_list = sorted(
        commenters_list, key=lambda x: x['num_comments'], reverse=True)

    return commenters_list


def get_first_commit_date(repo_owner, repo_name, contributor_username):
    """
    Retrieves the date of the first commit made by a contributor to a repository.

    Args:
        repo_owner (str): The username or organization name that owns the repository.
        repo_name (str): The name of the repository.
        contributor_username (str): The username of the contributor.

    Returns:
        str: The date of the first commit made by the contributor in the format "YYYY-MM-DD".
    """
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"Bearer {API_TOKEN}"
    }
    response = requests.get(f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits",
                            params={"author": contributor_username}, headers=headers)
    response.raise_for_status()

    commits = response.json()

    if not commits:
        print(
            f"\tNo commits found for contributor {contributor_username} in repository {repo_owner}/{repo_name}")
        return None

    first_commit_date = commits[-1]["commit"]["author"]["date"][:10]

    return datetime.strptime(first_commit_date, '%Y-%m-%d')


def get_prs_for_contributor(repo_owner: str, repo_name: str, contributor: str):
    access_token = API_TOKEN
    default_lookback = int(os.getenv('DEFAULT_MONTHS_LOOKBACK', 3))
    since_date = (datetime.now(
    ) - timedelta(weeks=default_lookback*4)).strftime('%Y-%m-%dT%H:%M:%SZ')

    url = f'https://api.github.com/search/issues?q=type:pr+repo:{repo_owner}/{repo_name}+author:{contributor}+created:>{since_date}&per_page=1000'
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"Bearer {API_TOKEN}"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        if response.status_code == 422:  # Try again
            print(
                f"\tGot a 422 response on search/issues?q=type:pr+repo:{repo_owner}/{repo_name}+author:{contributor}+created:>{since_date}, so retrying after 5 sec...")
            time.sleep(5)
            response = requests.get(url, headers=headers)
    # response.raise_for_status()
    if response.status_code == 200:
        return response.json()['total_count']


def get_workdays(start_date, end_date):
    weekdays = 0
    delta = timedelta(days=1)

    while start_date <= end_date:
        if start_date.weekday() < 5:
            weekdays += 1
        start_date += delta

    return weekdays


def add_quintile_stats(df):
    # df is the dataframe of contributor stats. Calc quintiles, add columns, return new df
    df['prs_quintile'] = pd.qcut(
        df['prs_per_day'], 5, labels=False, duplicates='drop')
    df['commits_quintile'] = pd.qcut(
        df['commits_per_day'], 5, labels=False, duplicates='drop')
    df['lines_of_code_quintile'] = pd.qcut(
        df['changed_lines_per_day'], 5, labels=False, duplicates='drop',)
    df['review_comments_quintile'] = pd.qcut(
        df['review_comments_per_day'], 5, labels=False, duplicates='drop',)
    cols_to_average = ['prs_quintile', 'commits_quintile',
                       'review_comments_quintile', 'lines_of_code_quintile']
    df['avg_quintile'] = df[cols_to_average].mean(axis=1)
    return df


def get_contributors_stats(repo_owner: str, repo_names: List[str], months_lookback: int) -> List[Tuple[str, str, str, float, float, float, float]]:
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

    today = datetime.now().date()
    start_date = pd.Timestamp.today() - pd.DateOffset(months=DEFAULT_MONTHS_LOOKBACK)
    end_date = pd.Timestamp.today()
    contributors = []

    max_num_workdays = get_workdays(start_date, end_date)
    for repo_name in repo_names:
        print(f"\n{repo_owner}/{repo_name}")
        list_dict_commenter_stats: list = get_commenters_stats(
            repo_owner, repo_name, months_lookback)
        # Returns the total number of commits authored by the contributor. In addition, the response includes a Weekly Hash (weeks array) with the following information:
        # w - Start of the week, given as a Unix timestamp.
        # a - Number of additions
        # d - Number of deletions
        # c - Number of commits
        url = f"{GITHUB_API_BASE_URL}/repos/{repo_owner}/{repo_name}/stats/contributors"
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            if response.status_code == 202:
                retry_url = response.headers.get('Location')
                if retry_url:
                    # Wait for 5 seconds before checking the status
                    time.sleep(5)
                    response = requests.get(retry_url, headers=headers)
                    # Handle the status response
                else:
                    # Wait for 5 seconds before retrying the request
                    time.sleep(5)
                    response = requests.get(url, headers=headers)
        # Handle the response
        if response.status_code == 200:
            for contributor in response.json():
                contributor_username = contributor.get(
                    "author", {}).get("login", "")
                contributor_name = get_github_collaborator_name(
                    contributor_username)
                if contributor_name is None:
                    contributor_name = contributor_username
                print(f"\t{contributor_name} ({contributor_username})")
                first_commit_date = get_first_commit_date(
                    repo_owner, repo_name, contributor_username)
                if first_commit_date is not None and first_commit_date > start_date:
                    num_workdays = get_workdays(first_commit_date, end_date)
                else:
                    num_workdays = max_num_workdays

                contributor_stats = {"repo": repo_name, "contributor_name": contributor_name,
                                     "contributor_username":  contributor_username,
                                     "stats_beginning": start_date,
                                     "stats_ending": today,
                                     "contributor_first_commit_date": first_commit_date,
                                     "num_workdays": num_workdays, "commits": 0, "prs": 0,
                                     "review_comments": 0, "changed_lines": 0}
                for weekly_stat in contributor["weeks"]:
                    weekly_date = datetime.utcfromtimestamp(
                        weekly_stat["w"])

                    if weekly_date >= start_date:
                        contributor_stats["commits"] += weekly_stat["c"]
                        contributor_stats["changed_lines"] += weekly_stat.get(
                            "d", 0) + weekly_stat.get("a", 0)

                prs = get_prs_for_contributor(
                    repo_owner, repo_name, contributor_username)
                if prs is not None:
                    contributor_stats["prs"] += prs
                for dict_commenter_stats in list_dict_commenter_stats:
                    if dict_commenter_stats["commenter_name"] == contributor_username:
                        contributor_stats["review_comments"] += dict_commenter_stats["num_comments"]

                # Only save stats if there are stats to save
                if contributor_stats["commits"] == 0 and contributor_stats["prs"] == 0 and contributor_stats["review_comments"] == 0 and contributor_stats["changed_lines"] == 0:
                    continue

                # Normalize per workday in lookback period
                else:

                    if contributor_stats["commits"] > 0:
                        contributor_stats["commits_per_day"] = round(
                            contributor_stats["commits"]/num_workdays, 3)
                    if contributor_stats["changed_lines"] > 0:
                        contributor_stats["changed_lines_per_day"] = round(
                            contributor_stats["changed_lines"]/num_workdays, 3)
                    if contributor_stats["prs"] > 0:
                        contributor_stats["prs_per_day"] = round(
                            contributor_stats["prs"]/num_workdays, 3)
                    if contributor_stats["review_comments"] > 0:
                        contributor_stats["review_comments_per_day"] = round(
                            contributor_stats["review_comments"]/num_workdays, 3)

                contributors.append(contributor_stats)

    return contributors


def save_contributors_to_csv(contributors, filename):
    df = pd.DataFrame(contributors)
    df = add_quintile_stats(df)
    df.to_csv(filename, index=False)
    return df


if __name__ == "__main__":
    repo_owner = os.getenv("REPO_OWNER")
    repo_names = os.getenv("REPO_NAMES").split(",")
    months_lookback = int(os.getenv("DEFAULT_MONTHS_LOOKBACK", 3))

    date_string = date.today().strftime('%Y-%m-%d')
    contributors_stats = get_contributors_stats(
        repo_owner, repo_names, months_lookback)

    df = save_contributors_to_csv(
        contributors_stats, f'{date_string}-{months_lookback}-{repo_owner}_contributor_stats.csv')
    summary = df.describe()

    # write the summary statistics to a new CSV file
    summary.to_csv(
        f'{date_string}-{months_lookback}-{repo_owner}_summary_stats.csv')
