from __future__ import annotations

import json
from datetime import datetime, timezone
from datetime import timedelta, date
import os
import sys
from typing import List, Tuple
import pandas as pd
import numpy as np
import time
import requests
from scipy.stats import norm
import itertools
from dateutil.relativedelta import relativedelta


GITHUB_API_BASE_URL = "https://api.github.com"
DEFAULT_MONTHS_LOOKBACK = int(os.getenv("DEFAULT_MONTHS_LOOKBACK", 0))
API_TOKEN = os.getenv("GITHUB_API_TOKEN")
MAX_ITEMS_PER_PAGE = int(os.getenv("MAX_ITEMS_PER_PAGE", 1000))
MIN_WORKDAYS_AS_CONTRIBUTOR = int(os.getenv("MIN_WORKDAYS_AS_CONTRIBUTOR", 30))
# WIP: changing HTTP 202 response retry logic
exponential_backoff_retry_delays: list[int] = [1, 2, 4, 8, 16]


def get_date_months_ago(months_ago) -> datetime:
    current_date = datetime.now()
    date_months_ago = current_date - relativedelta(months=months_ago)
    return date_months_ago

# Hit the url and back off retries if we get 202 or 422


def github_request_exponential_backoff(url):
    exponential_backoff_retry_delays_list: list[int] = [1, 2, 4, 8, 16]
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"Bearer {API_TOKEN}"
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        if response.status_code == 422 or response.status_code == 202:  # Try again
            for retry_attempt_delay in exponential_backoff_retry_delays_list:
                retry_url = response.headers.get('Location')
                # Wait for n seconds before checking the status
                time.sleep(retry_attempt_delay)
                retry_response_url: str = retry_url if retry_url else url
                print(
                    f"Retrying request for {retry_response_url} after {retry_attempt_delay} sec due to {response.status_code} response")
                response = requests.get(
                    retry_response_url, headers=headers)

                # Check if the retry response is 200
                if response.status_code == 200:
                    break  # Exit the loop on successful response

    if response.status_code == 200:
        return response.json()
    else:
        return None


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


def get_commenters_stats(repo_owner, repo_name, since_date: datetime):
    # Exclude bot users who don't get measured
    user_exclude_env: str = os.getenv('USER_EXCLUDE', None)
    user_exclude_list = user_exclude_env.split(',') if user_exclude_env else []
    commenters_list = []
    # calculate start and end dates
    today: datetime = datetime.now().date()
    start_date: datetime = since_date
    end_date: datetime = today + timedelta(days=1)

    # get all pull requests for the repo within the lookback period
    base_pr_url: str = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls"
    url = f"{base_pr_url}?state=closed&since={start_date}&until={end_date}&per_page={MAX_ITEMS_PER_PAGE}"
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    pull_requests = []
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Optional: Check for HTTP errors
        pull_requests = response.json()
    except requests.Timeout:
        print(
            f"\tTimeout error: The request for {url} timed out")

    except requests.exceptions.RequestException as e:
        print(f"\tRequest error: {e} for {url}")

    # get all comments for each pull request and add up the number of comments per commenter
    commenters = {}
    for pr in pull_requests:
        url = pr['comments_url']
        comments = []
        try:
            # Get comments
            response = requests.get(url, headers=headers)
            comments = response.json()

            # Get reviews with changes requested
            reviews_url = f"{base_pr_url}/{pr['number']}/reviews"
            review_response = requests.get(reviews_url, headers=headers)
            reviews = review_response.json()

            for comment in comments:
                if "user" not in comment:
                    continue
                commenter_name = comment['user']['login']
                # skip those users we aren't tracking (typically bots)
                if commenter_name in user_exclude_list:
                    continue
                if commenter_name not in commenters:
                    commenters[commenter_name] = 1
                else:
                    commenters[commenter_name] += 1

            # Count CHANGES_REQUESTED reviews by the specified user
            for review in reviews:
                reviewer_name: str = review['user']['login']
                # skip those users we aren't tracking (typically bots)
                if reviewer_name in user_exclude_list:
                    continue
                if review['state'] == 'CHANGES_REQUESTED':
                    if reviewer_name not in commenters:
                        commenters[reviewer_name] = 1
                    else:
                        commenters[reviewer_name] += 1

        except requests.Timeout:
            print(f"\tTimeout error: The request for {url} timed out")
            continue  # to the next PR
        except requests.exceptions.RequestException as e:
            print(f"\tRequest error: {e} for {url}")
            continue  # to the next PR

    if len(commenters) > 0:
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
                            params={"author": contributor_username, "per_page": MAX_ITEMS_PER_PAGE}, headers=headers)
    response.raise_for_status()

    commits = response.json()

    if not commits:
        print(
            f"\tNo commits found for contributor {contributor_username} in repository {repo_owner}/{repo_name}")
        return None

    first_commit_date = commits[-1]["commit"]["author"]["date"][:10]

    return datetime.strptime(first_commit_date, '%Y-%m-%d')

# Return a count of PRs for a contributor or None

# This is missing PRs - check it for AP


def get_prs_for_contributor(repo_owner: str, repo_name: str, contributor: str):
    access_token = API_TOKEN
    default_lookback = int(os.getenv('DEFAULT_MONTHS_LOOKBACK', 3))
    since_date = (datetime.now(
    ) - timedelta(weeks=default_lookback*4)).strftime('%Y-%m-%dT%H:%M:%SZ')

    url = f'https://api.github.com/search/issues?q=type:pr+repo:{repo_owner}/{repo_name}+author:{contributor}+created:>{since_date}&per_page={MAX_ITEMS_PER_PAGE}'
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"Bearer {API_TOKEN}"
    }

    # TO DO - test this new exponential backoff and make it a method, since they're all the same
    response = github_request_exponential_backoff(url)

    if response is not None and 'total_count' in response:
        return response['total_count']
    else:
        return None


def get_workdays(start_date, end_date):
    weekdays = 0
    delta = timedelta(days=1)

    while start_date <= end_date:
        if start_date.weekday() < 5:
            weekdays += 1
        start_date += delta

    return weekdays


def convert_to_letter_grade(score):
    grades = ['F', 'D', 'C', 'B', 'A']
    modifiers = ['-', '', '+']

    grade_idx = int(score) if score == 4 else int(score // 1)
    modifier_idx = 0

    if (score - grade_idx) >= 0.666:
        modifier_idx = 2
    elif (score - grade_idx) >= 0.333:
        modifier_idx = 1
    elif score == 4:
        modifier_idx = 1

    letter_grade = grades[grade_idx] + modifiers[modifier_idx]

    return letter_grade


def add_ntile_stats(df):
    ntile = 10  # Decile
    # df is the dataframe of contributor stats. Calc ntiles, add columns, return new df
    df['prs_ntile'] = pd.qcut(
        df['prs_per_day'], ntile, labels=False, duplicates='drop')
    df['commits_ntile'] = pd.qcut(
        df['commits_per_day'], ntile, labels=False, duplicates='drop')
    df['lines_of_code_ntile'] = pd.qcut(
        df['changed_lines_per_day'], ntile, labels=False, duplicates='drop',)
    df['review_comments_ntile'] = pd.qcut(
        df['review_comments_per_day'], ntile, labels=False, duplicates='drop',)
    cols_to_average = ['prs_ntile', 'commits_ntile',
                       'review_comments_ntile', 'lines_of_code_ntile']
    df['avg_ntile'] = df[cols_to_average].mean(axis=1)
    # df['grade'] = df['avg_ntile'].apply(convert_to_letter_grade)
    return df


def curve_scores(df, scores_column_name, curved_score_column_name):
    # Calculate the mean and standard deviation of the scores
    mean = df[scores_column_name].mean()
    std_dev = df[scores_column_name].std()

    # Calculate the Z-scores for each score
    z_scores = (df[scores_column_name] - mean) / std_dev

    # Create a normal distribution with mean 0 and standard deviation 1
    norm_dist = norm(0, 1)

    # Calculate the cumulative distribution function (CDF) for each Z-score
    cdf = norm_dist.cdf(z_scores)

    # Map the CDF values to a 0-100 range
    curved_scores = (cdf * 100).round().astype(int)

    # Update the DataFrame with the curved scores, near left side since this is important data
    df.insert(3, curved_score_column_name, curved_scores)

    return df


def get_contributors_stats(repo_owner: str, repo_names: List[str], since_date: datetime) -> List[Tuple[str, str, str, float, float, float, float]]:
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

    today: datetime = datetime.now().date()
    start_date: datetime = since_date
    end_date: datetime = pd.Timestamp.today()  # Seems redundant with today above
    contributors = []
    # {username: name} So we only call get_github_collaborator_name if we don't already have it
    dict_user_names = {}

    max_num_workdays = get_workdays(start_date, end_date)
    for repo_name in repo_names:
        print(f"\n{repo_owner}/{repo_name}")
        list_dict_commenter_stats: list = get_commenters_stats(
            repo_owner, repo_name, since_date)
        # Returns the total number of commits authored by the contributor. In addition, the response includes a Weekly Hash (weeks array) with the following information:
        # w - Start of the week, given as a Unix timestamp.
        # a - Number of additions
        # d - Number of deletions
        # c - Number of commits
        url = f"{GITHUB_API_BASE_URL}/repos/{repo_owner}/{repo_name}/stats/contributors?per_page={MAX_ITEMS_PER_PAGE}"

        # Get a list of contributors to this repo
        response = github_request_exponential_backoff(url)
        if response is not None and isinstance(response, list):
            for contributor in response:
                first_commit_date: date = None
                contributor_name: str = None
                contributor_username: str = None
                contributor_stats: dict = None
                contributor_already_added: bool = False
                days_since_first_commit: int = 0

                contributor_username = contributor.get(
                    "author", {}).get("login", "")
                # See if we have cached it to avoid unnecessary duplicate calls to GH
                if contributor_username not in dict_user_names:
                    contributor_name = get_github_collaborator_name(
                        contributor_username)
                    dict_user_names[contributor_username] = contributor_name
                else:
                    contributor_name = dict_user_names[contributor_username]

                if contributor_name is None:
                    contributor_name = contributor_username
                print(f"\t{contributor_name} ({contributor_username})")
                first_commit_date = get_first_commit_date(
                    repo_owner, repo_name, contributor_username)
                num_workdays = max_num_workdays
                # If their first commit is < roughly 6 weeks ago (configurable),
                # don't measure contributions.
                if first_commit_date is not None:
                    days_since_first_commit = get_workdays(
                        first_commit_date, datetime.now())
                if days_since_first_commit < MIN_WORKDAYS_AS_CONTRIBUTOR:
                    continue
                # Conditionally override num workdays if they started after start date
                if first_commit_date is not None and first_commit_date > start_date:
                    num_workdays = get_workdays(first_commit_date, end_date)

                # TO DO: if the contributor username is already in the array, use the one with the earliest date
                # and add in the other stats to the existing stats
                for contributor_stats_dict in contributors:
                    if contributor_stats_dict["contributor_name"] == contributor_name:
                        contributor_stats = contributor_stats_dict
                        contributor_already_added = True
                        print(
                            f"\t\tMerging stats for {contributor_name} ({contributor_username}) with previously found contributor {contributor_stats_dict['contributor_name']} ({contributor_stats_dict['contributor_username']})")
                        contributor_stats["repo"] = f"{contributor_stats['repo']},{repo_name}"
                        if first_commit_date < contributor_stats["contributor_first_commit_date"]:
                            contributor_stats["contributor_first_commit_date"] = first_commit_date
                        break

                # if this is the first time we've seen this contributor, init a dict
                if contributor_stats is None:
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
                prs_count = get_prs_for_contributor(
                    repo_owner, repo_name, contributor_username)
                if prs_count is not None:
                    contributor_stats["prs"] += prs_count
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

                if not contributor_already_added:
                    contributors.append(contributor_stats)

        # This means we have nothing returned from our attempt to get contributors for a repo
        else:
            print(
                f'\tNo contributors found for {repo_name} since {since_date}.')
            continue

    return contributors

# Shorten filename


def truncate_filename(repos):
    max_length = 230
    if len(repos) > max_length:
        repos = repos[:max_length]
        # remove any illegal characters
        # filename = ''.join(c for c in filename if c.isalnum() or c in ['_', '.', ' '])
    return repos


def save_contributors_to_csv(contributors, filename):
    df = None
    if len(contributors) > 0:
        # Clean up
        df = pd.DataFrame(contributors)
        df["commits_per_day"] = df['commits_per_day'].fillna(0)
        df["changed_lines_per_day"] = df['changed_lines_per_day'].fillna(0)
        df["prs_per_day"] = df['prs_per_day'].fillna(0)
        df["review_comments_per_day"] = df['review_comments_per_day'].fillna(0)
        df["prs_per_day"] = df['prs_per_day'].fillna(0)
        # Remove any rows where there are no commits and no PRs.
        # I'm seeing Github return PR comments from people who were not involved in the lookback
        # period. I haven't diagnosed this. This is a hacky way to get rid of them.
        # Obvy, if PRs and commits are zero, so are changed_lines.
        df = df[~((df['commits'] == 0) & (df['prs'] == 0))]
        df = add_ntile_stats(df)
        df = curve_scores(df, "avg_ntile", "curved_score")
        df.to_csv(filename, index=False)
    else:
        print(f"\t No contributors for {filename}")
    return df


def get_repos_by_topic(repo_owner, topic, topic_exclude, since_date_str):
    # url = f'https://api.github.com/search/repositories?q=topic:{topic}+org:{repo_owner}+pushed:>={since_date_str}&per_page={MAX_ITEMS_PER_PAGE}'
    url = (
        f'https://api.github.com/search/repositories?q=topic:{topic}'
        f'+org:{repo_owner}'
        f'+pushed:>={since_date_str}'
        f'{"+-topic:" + topic_exclude if topic_exclude is not None else ""}'
        f'&per_page={MAX_ITEMS_PER_PAGE}'
    )

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"Bearer {API_TOKEN}"
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        if response.status_code == 422:  # Try again
            for retry_attempt_delay in exponential_backoff_retry_delays:
                retry_url = response.headers.get('Location')
                # Wait for n seconds before checking the status
                time.sleep(retry_attempt_delay)
                retry_response_url: str = retry_url if retry_url else url
                print(
                    f"Retrying request for {retry_response_url} after {retry_attempt_delay} sec due to {response.status_code} response")
                response = requests.get(
                    retry_response_url, headers=headers)

                # Check if the retry response is 200
                if response.status_code == 200:
                    break  # Exit the loop on successful response

    item_list: {} = {}
    item_list_returned: [] = []
    if response.status_code == 200:
        item_list = response.json().get('items', [])
        for item in item_list:
            item_list_returned.append(item["name"])
        return item_list_returned


if __name__ == "__main__":
    # Get the env settings
    repo_owner = os.getenv("REPO_OWNER")
    repo_names_env = os.getenv("REPO_NAMES")
    repo_names = repo_names_env.split(",") if repo_names_env else []
    months_lookback = int(os.getenv("DEFAULT_MONTHS_LOOKBACK", 3))
    topic_env = os.getenv("TOPIC")
    topic_name = topic_env if topic_env else None
    topic_exclude_env = os.getenv("TOPIC_EXCLUDE")
    topic_exclude_name = topic_exclude_env if topic_exclude_env else None
    if months_lookback < 1:
        months_lookback = 3

    default_lookback = int(os.getenv('DEFAULT_MONTHS_LOOKBACK', 3))
    since_date: datetime = get_date_months_ago(default_lookback)
    since_date_str: str = since_date.strftime('%Y-%m-%d')

    # If there were no repo_names in .env, we can pull the repos based on the topic
    if len(repo_names) == 0 and topic_name is not None:
        repo_names = get_repos_by_topic(
            repo_owner, topic_name, topic_exclude_name, since_date_str)

    if len(repo_names) == 0 and topic_name is None:
        print(f'Either TOPIC or REPO_NAMES must be provided in .env')
        sys.exit()

    # Tell the user what they're getting
    print(f'Stats for {repo_owner} repos:')
    print(f', '.join(repo_names))
    print(f':')

    # This does all the work
    contributors_stats = get_contributors_stats(
        repo_owner, repo_names, since_date)

    date_string = date.today().strftime('%Y-%m-%d')
    filename = truncate_filename(
        f'{date_string}-{months_lookback}-{repo_owner}_{repo_names}')
    df = save_contributors_to_csv(
        contributors_stats, f'contribs_{filename}.csv')
    # summary = df.describe()

    # write the summary statistics to a new CSV file
    # summary.to_csv(f'summary_{filename}.csv')
