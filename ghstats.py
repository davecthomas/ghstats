import os
from datetime import date, datetime, timedelta
from typing import List, Tuple
import pandas as pd
import numpy as np
import time
import requests

GITHUB_API_BASE_URL = "https://api.github.com"
DEFAULT_MONTHS_LOOKBACK = int(os.getenv("DEFAULT_MONTHS_LOOKBACK", 0))
API_TOKEN = os.getenv("GITHUB_API_TOKEN")

import requests
import datetime

import requests
import json

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
    today = datetime.datetime.now().date()
    start_date = today - datetime.timedelta(days=months_lookback*30)
    end_date = today + datetime.timedelta(days=1)

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
    commenters_list = [{'commenter_name': k, 'num_comments': v} for k, v in commenters.items()]
    commenters_list = sorted(commenters_list, key=lambda x: x['num_comments'], reverse=True)

    return commenters_list


def get_prs_for_contributor(repo_owner: str, repo_name: str, contributor: str):
    access_token = API_TOKEN
    default_lookback = int(os.getenv('DEFAULT_MONTHS_LOOKBACK', 3))
    since_date = (datetime.datetime.now() - timedelta(weeks=default_lookback*4)).strftime('%Y-%m-%dT%H:%M:%SZ')

    url = f'https://api.github.com/search/issues?q=type:pr+repo:{repo_owner}/{repo_name}+author:{contributor}+created:>{since_date}&per_page=1000'
    headers = {'Authorization': f'token {access_token}'}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        if response.status_code == 422: # Try again
            time.sleep(2000)
            response = requests.get(url, headers=headers)
    # response.raise_for_status()
    if response.status_code == 200:
        return response.json()['total_count']

def get_workdays(start_date, end_date):
    dates = pd.date_range(start_date, end_date, freq='B')
    dates = dates.values.astype('datetime64[D]')  # cast to datetime64[D]
    mask = np.is_busday(dates)
    num_workdays = np.count_nonzero(mask)
    return num_workdays

def get_contributors_stats(repo_owner: str, repo_names: List[str], months_lookback: int) -> List[Tuple[str, str, str, float, float, float, float]]:
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

    today = datetime.datetime.now().date()
    start_date = pd.Timestamp.today() - pd.DateOffset(months=DEFAULT_MONTHS_LOOKBACK)
    end_date = pd.Timestamp.today()
    num_workdays = get_workdays(start_date, end_date)
    contributors = []

    for repo_name in repo_names:
        list_dict_commenter_stats: list = get_commenters_stats(repo_owner, repo_name, months_lookback)
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
                contributor_username = contributor.get("author", {}).get("login", "")
                contributor_name = get_github_collaborator_name(contributor_username)
                contributor_stats = {"repo": repo_name, "contributor_name": contributor_name, "contributor_username": contributor_username, "start_date": start_date, "end_date": today, "num_workdays": num_workdays, "commits": 0, "prs": 0, "review_comments": 0, "changed_lines": 0}
                for weekly_stat in contributor["weeks"]:
                    weekly_date = datetime.datetime.utcfromtimestamp(weekly_stat["w"])

                    if weekly_date >= start_date:
                        contributor_stats["start_date"] = weekly_date
                        contributor_stats["commits"] += weekly_stat["c"]
                        contributor_stats["changed_lines"] += weekly_stat.get("d", 0) + weekly_stat.get("a", 0)

                contributor_stats["prs"] = get_prs_for_contributor(repo_owner,repo_name, contributor_username)
                for dict_commenter_stats in list_dict_commenter_stats:
                    if dict_commenter_stats["commenter_name"] == contributor_username:
                        contributor_stats["review_comments"] += dict_commenter_stats["num_comments"]

                # Only save stats if there are stats to save
                if contributor_stats["commits"] == 0 and contributor_stats["prs"] == 0 and contributor_stats["review_comments"] == 0 and contributor_stats["changed_lines"] == 0:
                    continue

                # Normalize per workday in lookback period
                else:
                    contributor_stats["commits"] /= num_workdays
                    contributor_stats["changed_lines"] /= num_workdays
                    contributor_stats["prs"] /= num_workdays
                    contributor_stats["review_comments"] /= num_workdays
            
                contributors.append(contributor_stats)

    return contributors


def save_contributors_to_csv(contributors, filename):
    df = pd.DataFrame(contributors)
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    repo_owner = os.getenv("REPO_OWNER")
    repo_names = os.getenv("REPO_NAMES").split(",")
    months_lookback = int(os.getenv("DEFAULT_MONTHS_LOOKBACK", 3))

    contributors_stats = get_contributors_stats(repo_owner, repo_names, months_lookback)
    save_contributors_to_csv(contributors_stats, f'{repo_owner}_contributor_stats.csv')

    for contributor_stat in contributors_stats:
        print(contributor_stat)
