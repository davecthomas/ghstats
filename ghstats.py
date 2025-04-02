from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import json
from datetime import datetime
from datetime import date
import os
import sys
import pandas as pd
import numpy as np
import time
from dotenv import load_dotenv
import requests
from requests.exceptions import Timeout, RequestException, HTTPError
from urllib3.exceptions import ProtocolError
from scipy.stats import norm
from dateutil.relativedelta import relativedelta
from requests.models import Response
from collections import defaultdict

from ghs_snowflake import GhsSnowflakeStorageManager
from ghs_utils import *

# Globals
GITHUB_API_BASE_URL = "https://api.github.com"
DEFAULT_MONTHS_LOOKBACK = int(os.getenv("DEFAULT_MONTHS_LOOKBACK", 0))
API_TOKEN = os.getenv("GITHUB_API_TOKEN")
MAX_ITEMS_PER_PAGE = 100  # Limited by Github API
MIN_WORKDAYS_AS_CONTRIBUTOR = int(os.getenv("MIN_WORKDAYS_AS_CONTRIBUTOR", 30))
STATS_TABLE_NAME = "CONTRIBUTOR_STATS"
MAX_REPO_NAMES_TO_PRINT = 20  # Limit the number of repo names we print out in the log
SECONDS_PER_DAY = 86400  # For calculating PR durations

# Our "cache" of contributors basic attributes
# {username: {"display_name": Display name, "node_id": node_id}}
# Each dictionary entry is just a cache of what is returned from get_github_user_attributes
gdict_user_attributes: Dict[str, Dict[str, str]] = {}


class GhsGithub:
    def __init__(self):
        self.dict_env: dict = None
        self.storage_manager = GhsSnowflakeStorageManager()
        self.dict_env = self.get_env()
        self.since_date: date = get_first_day_months_ago(
            self.dict_env.get("months_lookback", DEFAULT_MONTHS_LOOKBACK)
        )
        self.until_date: date = get_end_of_last_complete_month()

    def __del__(self):
        """Destructor to ensure the Snowflake connection is closed."""
        try:
            self.storage_manager.close_connection()
        except Exception as e:
            print(f"Error closing Snowflake connection: {e}")

    def get_env(self) -> dict:
        load_dotenv()  # Load environment variables from .env file
        """
        Get env values
        Returns None if anything is missing, else returns a dict of settings
        """
        dict_env: dict = {
            "repo_owner": None,
            "repo_names": [],
            "repo_names_exclude": [],
            "months_lookback": DEFAULT_MONTHS_LOOKBACK,
            "months_count": 1,
            "topic_name": None,
            "topic_exclude_name": None,
            "dict_all_repo_topics": {},
        }
        # Get the env settings
        dict_env["repo_owner"] = os.getenv("REPO_OWNER", None)
        print(f"repo_owner: {dict_env['repo_owner']}")
        # Optional env var
        repo_names_env = os.getenv("REPO_NAMES")
        dict_env["repo_names"] = repo_names_env.split(",") if repo_names_env else []
        # Optional env var
        repo_names_exclude_env = os.getenv("REPO_NAMES_EXCLUDE")
        dict_env["repo_names_exclude"] = (
            repo_names_exclude_env.split(",") if repo_names_exclude_env else []
        )
        # For each month we look back, we need to collect a set of stats so we can get a time series
        dict_env["months_lookback"] = int(os.getenv("DEFAULT_MONTHS_LOOKBACK", 3))
        if dict_env["months_lookback"] < 1:
            dict_env["months_lookback"] = 3
        dict_env["months_count"] = int(os.getenv("MONTHS_COUNT", 1))
        # Optional env var, which can be "all" to get all repo topics (thus all repos with topics) in the org
        topic_env = os.getenv("TOPIC")
        dict_env["topic_name"] = topic_env if topic_env else None
        # Optional env var
        topic_exclude_env = os.getenv("TOPIC_EXCLUDE")
        dict_env["topic_exclude_name"] = (
            topic_exclude_env if topic_exclude_env else None
        )
        if dict_env["repo_owner"] is None:
            print("No repo_owner in setttings.")
            return None

        # do a deep copy of the dictionary
        self.dict_env = dict_env.copy()

        # These next two get initialized in prep_repo_topics
        if "all" in dict_env["repo_names"] or "all" == dict_env["topic_name"]:
            self.prep_repo_topics()

        return self.dict_env

    def check_API_rate_limit(self, response: Response) -> bool:
        """
        Check if we overran our rate limit. Take a short nap if so.
        Return True if we overran

        """
        if response.status_code == 403 and "X-Ratelimit-Remaining" in response.headers:
            if int(response.headers["X-Ratelimit-Remaining"]) == 0:
                print(
                    f"\t403 forbidden response header shows X-Ratelimit-Remaining at {response.headers['X-Ratelimit-Remaining']} requests."
                )
                sleep_until_ratelimit_reset_time(
                    int(response.headers["X-RateLimit-Reset"])
                )
        return (
            response.status_code == 403 and "X-Ratelimit-Remaining" in response.headers
        )

    def github_request_exponential_backoff(
        self, url: str, params: Dict[str, Any] = {}
    ) -> List[Dict]:
        """
        Returns a list of pages (or just one page) where each page is the full json response
        object. The caller must know to process these pages as the outer list of the result.
        Retry backoff in 422, 202, or 403 (rate limit exceeded) responses
        """
        exponential_backoff_retry_delays_list: list[int] = [1, 2, 4, 8, 16, 32, 64]
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"Bearer {API_TOKEN}",
        }

        retry: bool = False
        retry_count: int = 0
        response: Response = Response()
        retry_url: str = None
        pages_list: List[Dict] = []
        page = 1
        if "per_page" not in params:
            params["per_page"] = MAX_ITEMS_PER_PAGE

        while True:
            params["page"] = page

            try:
                response = requests.get(url, headers=headers, params=params)
            except Timeout:
                print(
                    f"Request to {url} with params {params} timed out on attempt {retry_count + 1}. Retrying in {exponential_backoff_retry_delays_list[retry_count]} seconds."
                )
                retry = True
                retry_count += 1
                continue
            except ProtocolError as e:
                print(
                    f"Protocol error on attempt {retry_count + 1}: {e}. Retrying in {exponential_backoff_retry_delays_list[retry_count]} seconds."
                )
                retry = True
                retry_count += 1
                continue
            except ConnectionError as ce:
                print(
                    f"Connection error on attempt {retry_count + 1}: {ce}. Retrying in {exponential_backoff_retry_delays_list[retry_count]} seconds."
                )
                retry = True
                retry_count += 1
                continue
            except HTTPError as he:
                print(
                    f"HTTP error on attempt {retry_count + 1}: {he}. Retrying in {exponential_backoff_retry_delays_list[retry_count]} seconds."
                )
                retry = True
                retry_count += 1
                continue
            except RequestException as e:
                print(
                    f"Request exception on attempt {retry_count + 1}: {e}. Retrying in {exponential_backoff_retry_delays_list[retry_count]} seconds."
                )
                retry = True
                retry_count += 1
                continue

            if retry or (response is not None and response.status_code != 200):
                if (
                    response.status_code == 422
                    and response.reason == "Unprocessable Entity"
                ):
                    dict_error: Dict[str, any] = json.loads(response.text)
                    print(
                        f"Skipping: {response.status_code} {response.reason} for url {url}\n\t{dict_error['message']}\n\t{dict_error['errors'][0]['message']}"
                    )

                elif (
                    retry or response.status_code == 202 or response.status_code == 403
                ):  # Try again
                    for retry_attempt_delay in exponential_backoff_retry_delays_list:
                        if "Location" in response.headers:
                            retry_url = response.headers.get("Location")
                        # The only time we override the exponential backoff if we are asked by Github to wait
                        if "Retry-After" in response.headers:
                            retry_attempt_delay = response.headers.get("Retry-After")
                        # Wait for n seconds before checking the status
                        time.sleep(retry_attempt_delay)
                        retry_response_url: str = retry_url if retry_url else url
                        print(
                            f"Retrying request for {retry_response_url} after {retry_attempt_delay} sec due to {response.status_code} response"
                        )
                        # A 403 may require us to take a nap
                        self.check_API_rate_limit(response)

                        try:
                            response = requests.get(retry_response_url, headers=headers)
                        except Timeout:
                            print(
                                f"Request to {url} with params {params} timed out on attempt {retry_count + 1}. Retrying in {retry_attempt_delay} seconds."
                            )
                            retry = True
                            retry_count += 1
                            continue
                        except ProtocolError as e:
                            print(
                                f"Protocol error on attempt {retry_count + 1}: {e}. Retrying in {retry_attempt_delay} seconds."
                            )
                            retry = True
                            retry_count += 1
                            continue
                        except ConnectionError as ce:
                            print(
                                f"Connection error on attempt {retry_count + 1}: {ce}. Retrying in {retry_attempt_delay} seconds."
                            )
                            retry = True
                            retry_count += 1
                            continue
                        except HTTPError as he:
                            print(
                                f"HTTP error on attempt {retry_count + 1}: {he}. Retrying in {retry_attempt_delay} seconds."
                            )
                            retry = True
                            retry_count += 1
                            continue
                        except RequestException as e:
                            print(
                                f"Request exception on attempt {retry_count + 1}: {e}. Retrying in {retry_attempt_delay} seconds."
                            )
                            retry = True
                            retry_count += 1
                            continue
                        except Exception as e:
                            print(
                                f"Unexpected exception on attempt {retry_count + 1}: {e}. Retrying in {retry_attempt_delay} seconds."
                            )
                            retry = True
                            retry_count += 1
                            continue

                        # Check if the retry response is 200
                        if response.status_code == 200:
                            break  # Exit the loop on successful response
                        else:
                            print(
                                f"\tRetried request and still got bad response status code: {response.status_code}"
                            )

            if response.status_code == 200:
                page_json = response.json()
                if not page_json or (isinstance(page_json, list) and not page_json):
                    break  # Exit if the page is empty
                pages_list.append(response.json())
            else:
                self.check_API_rate_limit(response)
                print(
                    f"Retries exhausted. Giving up. Status code: {response.status_code}"
                )
                break

            if "next" not in response.links:
                break  # Check for a 'next' link to determine if we should continue
            else:
                # print(
                # f"Page {page} complete. Moving to page {page+1}")
                url = response.links.get("next", "").get("url", "")

            page += 1

        return pages_list

    def get_github_user_attributes(self, username) -> dict:
        """
        get basic user attributes

        Parameters:
        owner (str): The username of the owner of the repository.

        Returns: a dict of
        {contributor_username: str, contributor_name: str, contributor_nodeid: str}
        """
        user_attributes: dict = {}
        global gdict_user_attributes
        if username in gdict_user_attributes:
            return gdict_user_attributes[username]

        url = f"https://api.github.com/users/{username}"

        response = self.github_request_exponential_backoff(url)

        if response is not None and isinstance(response, List) and len(response) > 0:
            collaborator_info = response[0]
            display_name: str = collaborator_info.get("name", username)
            if display_name is None:
                display_name = username
            user_attributes = {
                "contributor_username": username,
                "contributor_name": display_name,
                "contributor_nodeid": collaborator_info.get("node_id", None),
            }
            gdict_user_attributes[username] = user_attributes
            return user_attributes
        else:
            return None

    def get_pr_stats(
        self, repo_owner: str, repo_name: str, since_date: date, until_date: date
    ) -> Dict[str, any]:
        """
        Returns all commenter stats, durations for all PRs in a repo during the date interval,
        and earliest date of each PR in the time interval. This is used later to calculate the duration of each PR.
        { "commenters_stats": List[Dict[str, any]],
        "contributor_pr_durations": List[Dict[str, any]],
        "prs_durations_dict": List[Dict[str, any]]}
        where prs_durations_dict is a dictionary of {pr_number: duration in days from first commit to closure}

        This might be confusing. We're returning contributors_pr_durations_list as well as prs_durations_dict
        Each of these is used for a different, but related, purpose.
        The first is used to calculate the average duration of a PR for each contributor.
        The second is used to calculate the duration of each PR. One is focued on the contributor, the other on the repo as a whole.
        Since a different person might be submitting the first commit in a PR, we don't want to conflate the two.

        TO DO: we aren't currently calculating or storing the duration of each PR. TBD after testing.
        1. Store this as an average for each repo in the database for each time period
        """
        pr_stats_dict: Dict[str, any] = {
            "commenters_stats": {},
            "contributor_pr_durations": {},
            "prs_durations_dict": {},
        }

        # Exclude bot users who don't get measured
        user_exclude_env: str = os.getenv("USER_EXCLUDE", None)
        user_exclude_list = user_exclude_env.split(",") if user_exclude_env else []
        commenters_list = []

        list_review_status_updates_we_track: List[str] = [
            "APPROVED",
            "CHANGES_REQUESTED",
            "COMMENTED",
        ]

        # Base URL for the Search API focusing on issues (which includes PRs)
        base_search_url: str = "https://api.github.com/search/issues"

        # Formulating the query part
        query: str = (
            f"repo:{repo_owner}/{repo_name}+is:pr+is:closed+created:{since_date.strftime('%Y-%m-%d')}..{until_date.strftime('%Y-%m-%d')}"
        )

        # Complete URL
        url: str = f"{base_search_url}?q={query}"

        # used when we get reviews for each PR, in the PR loop
        pr_reviews_base_url: str = (
            f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls"
        )

        pull_requests: list = []
        pull_requests_pages: list = []

        response = self.github_request_exponential_backoff(url)
        if response is not None and isinstance(response, List) and len(response) > 0:
            pull_requests_pages = response

        # get all comments for each pull request and add up the number of comments per commenter
        # Also, get the duration of each PR
        commenters: Dict[str, any] = {}
        contributors: Dict[str, any] = {}
        prs_durations_dict: Dict[str, datetime] = {}
        for pull_requests in pull_requests_pages:
            pull_requests_items: list = pull_requests.get("items", [])
            for pr in pull_requests_items:
                datetime_pr_closed: datetime = None
                # First, get duration
                duration_countable: bool = False
                try:
                    contributor_name: str = pr["user"]["login"]
                except KeyError:
                    continue
                except TypeError:
                    continue
                merged_at: str = pr.get("merged_at")
                closed_at: str = pr.get("closed_at")
                if merged_at:
                    duration = get_duration_in_days(pr["created_at"], merged_at)
                    duration_countable = True
                    datetime_pr_closed = datetime.strptime(
                        merged_at, "%Y-%m-%dT%H:%M:%SZ"
                    )
                elif closed_at:
                    duration = get_duration_in_days(pr["created_at"], closed_at)
                    duration_countable = True
                    datetime_pr_closed = datetime.strptime(
                        closed_at, "%Y-%m-%dT%H:%M:%SZ"
                    )
                if duration_countable:
                    # Save a tuple of total duration, number of PRs (so we can average later)
                    if contributor_name not in contributors:
                        contributors[contributor_name] = (duration, 1)
                    else:
                        contributors[contributor_name] = (
                            contributors[contributor_name][0] + duration,
                            contributors[contributor_name][1] + 1,
                        )

                """
                Get reviews with changes requested using the pr_reviews_base_url + the pr number 
                We want to count how many useful state responses each reviewer provides, 
                since this is an indication of repo productivity. 
                """
                reviews = []
                reviews_pages: List = []

                pr_number = pr.get("number", None)
                reviews_url = pr_number and f"{pr_reviews_base_url}/{pr_number}/reviews"
                # pr.get("review_comments_url", None)
                reviews_response = (
                    reviews_url and self.github_request_exponential_backoff(reviews_url)
                )
                if (
                    reviews_response is not None
                    and isinstance(reviews_response, List)
                    and len(reviews_response) > 0
                ):
                    reviews_pages = reviews_response

                for reviews in reviews_pages:
                    # Count state changes where the reviewer should get credit for their contribution
                    # TO DO: consider only providing credit for reviews with a "body" length < x,
                    # Since "LGTM" reviews are likely not contributing enough value to be counted.
                    for review in reviews:
                        # Sometimes there is a None user, such as when a PR is closed on a user
                        # no longer with the organization. Wrap in try:except, swallow, continue...
                        try:
                            reviewer_name: str = review["user"]["login"]
                        except KeyError:
                            continue
                        except TypeError:
                            continue
                        # skip those users we aren't tracking (typically GHA helper bots)
                        if reviewer_name in user_exclude_list:
                            continue
                        review_state: str = review.get("state", None)
                        if review_state in list_review_status_updates_we_track:
                            if reviewer_name not in commenters:
                                commenters[reviewer_name] = 1
                            else:
                                commenters[reviewer_name] += 1
                                reviews_url = (
                                    pr_number
                                    and f"{pr_reviews_base_url}/{pr_number}/reviews"
                                )

                # Get commits and get first date of commit per PR
                commits_url = pr_number and f"{pr_reviews_base_url}/{pr_number}/commits"
                commits_response = (
                    commits_url and self.github_request_exponential_backoff(commits_url)
                )
                if (
                    commits_response is not None
                    and isinstance(commits_response, List)
                    and len(commits_response) > 0
                ):
                    commits_pages = commits_response

                    # Find the earliest commit across any pages of commits returned
                    first_commit_datetime = datetime.now()  # default to now
                    for commits in commits_pages:
                        try:
                            commit_datetime: datetime = datetime.strptime(
                                commits[0]["commit"]["author"]["date"],
                                "%Y-%m-%dT%H:%M:%SZ",
                            )
                            if commit_datetime < first_commit_datetime:
                                first_commit_datetime = commit_datetime
                            # datetime_pr_closed
                        except KeyError:
                            continue
                        except TypeError:
                            continue

                    # Calculate the overall duration of the PR based on the bookends of first commit and closure
                    prs_durations_dict[pr_number] = (
                        datetime_pr_closed - first_commit_datetime
                    ).total_seconds() / SECONDS_PER_DAY
                # (if we don't have any commits in this PR, we skip storing any duration)

        # Convert the dictionary of tuples to a list of dictionaries
        contributors_pr_durations_list: List[Dict[str, any]] = []
        if len(contributors) > 0:
            contributors_pr_durations_list = [
                {"contributor_name": name, "total_duration": duration, "num_prs": prs}
                for name, (duration, prs) in contributors.items()
            ]
        if len(commenters) > 0:
            # convert commenters dictionary to list of dictionaries and sort by number of comments
            commenters_list = [
                {"commenter_name": k, "num_comments": v} for k, v in commenters.items()
            ]
            commenters_list = sorted(
                commenters_list, key=lambda x: x["num_comments"], reverse=True
            )
        pr_stats_dict["commenters_stats"] = commenters_list
        pr_stats_dict["contributor_pr_durations"] = contributors_pr_durations_list
        pr_stats_dict["prs_durations_dict"] = prs_durations_dict
        return pr_stats_dict

    def get_first_commit_date(
        self, repo_owner, repo_name, contributor_username
    ) -> date:
        """
        Retrieves the date of the first commit made by a contributor to a repository.

        Args:
            repo_owner (str): The username or organization name that owns the repository.
            repo_name (str): The name of the repository.
            contributor_username (str): The username of the contributor.

        Returns:
            str: The date of the first commit made by the contributor in the format "YYYY-MM-DD".
        """

        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits?author={contributor_username}"

        response = self.github_request_exponential_backoff(url)
        first_commit_date: date = date.today()

        if response is not None and isinstance(response, List) and len(response) > 0:
            commits_pages: List = response

            for commits in commits_pages:
                # Check for empty results
                if commits is None or (isinstance(commits, list) and len(commits)) == 0:
                    print(
                        f"\tNo commits found for contributor {contributor_username} in repository {repo_owner}/{repo_name}"
                    )
                    return None

                # Grab the last item in the list (the first time they committed)
                first_commit: Dict[str, any] = commits[-1]["commit"]
                try:
                    commit_date_str = first_commit["author"]["date"][:10]
                    commit_date: date = datetime.strptime(
                        commit_date_str, "%Y-%m-%d"
                    ).date()
                    if commit_date < first_commit_date:
                        first_commit_date = commit_date
                except KeyError:
                    print(
                        f"\tMalformed commit data found for contributor {contributor_username} in repository {repo_owner}/{repo_name}"
                    )
                    return None

            return first_commit_date

    def add_ntile_stats(self, df):
        """
        Take all the stats that will roll into a curved score and bucket by ntile

        """
        ntile = 10  # Decile
        # df is the dataframe of contributor stats. Calc ntiles, add columns, return new df
        df["prs_ntile"] = pd.qcut(
            df["prs_per_day"], ntile, labels=False, duplicates="drop"
        )
        df["commits_ntile"] = pd.qcut(
            df["commits_per_day"], ntile, labels=False, duplicates="drop"
        )
        df["lines_of_code_ntile"] = pd.qcut(
            df["changed_lines_per_day"],
            ntile,
            labels=False,
            duplicates="drop",
        )
        df["review_comments_ntile"] = pd.qcut(
            df["review_comments_per_day"],
            ntile,
            labels=False,
            duplicates="drop",
        )
        df["avg_pr_duration_ntile"] = pd.qcut(
            df["avg_pr_duration"],
            ntile,
            labels=False,
            duplicates="drop",
        )
        # We don't include lines of code since it's easy to argue that code quantity doesn't
        cols_to_average: List[str] = [
            "prs_ntile",
            "commits_ntile",
            "review_comments_ntile",
            "avg_pr_duration_ntile",
        ]
        df["avg_ntile"] = df[cols_to_average].mean(axis=1)
        # df['grade'] = df['avg_ntile'].apply(convert_to_letter_grade)
        return df

    def curve_scores(
        self, df: pd.DataFrame, scores_column_name: str, curved_score_column_name: str
    ) -> pd.DataFrame:
        # Handle cases where standard deviation is zero or there are NaN values
        mean = df[scores_column_name].mean()
        std_dev = df[scores_column_name].std()

        if std_dev > 0:
            # Calculate Z-scores
            z_scores = (df[scores_column_name] - mean) / std_dev
            # Calculate CDF values using the Z-scores
            norm_dist = norm(0, 1)
            cdf = norm_dist.cdf(z_scores)
            # Safely convert CDF values to a 0-100 scale and round to integer
            curved_scores = np.nan_to_num(cdf * 100).round().astype(int)
        else:
            # Assign a default score or handle the case as appropriate
            curved_scores = np.zeros(len(df))

        # Update the DataFrame
        df.insert(3, curved_score_column_name, curved_scores)

        return df

    def get_contributors_stats(
        self, env_dict: dict, repo_name: str, since_date: date, until_date: date
    ) -> Dict[str, Any]:
        """
        For each repo in the list of repos,
        Get the PR stats
        -   One list of dict of PR review stats
        -   One list of dict of PR durations and counts
        -   One dict of PR durations for each PR

        Get commits and count up changed lines of code per contributor

        Aggregate all this into a single dictionary of stats for the each contributor, so their stats
        accumulate (or average, depending on the stat), across all repos in the list.
        For example, we average the PR open duration, but we accumulate PRs and commits

        Returns: A dictionary of  {contributor_stats: List[Dict[str, any]], repo_stats: Dict[str, any]}
        1. "contributor_stats" - a list of dictionaries of stats for each contributor
        2. "repo_stats" - a dictionary of stats for the repo as a whole
        """
        dict_return: Dict[str, any] = {
            "contributor_stats": [],
            "repo_stats": Dict[str, any],
        }

        # This is the list of dictionaries of our stats, which this function populates
        list_dict_contributor_stats: List[Dict[str, Any]] = []

        max_num_workdays = get_workdays(since_date, until_date)
        # This is the dictionary of our stats for the repo, which this function populates
        dict_repo_stats: Dict[str, Any] = {
            "repo_name": repo_name,
            "stats_beginning": since_date,
            "stats_ending": until_date,
            "num_workdays": max_num_workdays,
            "num_contributors": 0,
            "avg_pr_duration": 0.0,
            "median_pr_duration": 0.0,
            "num_prs": 0,
            "num_commits": 0,
        }

        # Average PR durations for a contributor. One avg per repo. A list across all repos.
        # contributor_name: [list of averages]
        dict_avg_durations: Dict[str, List[float]] = {}

        # Get the PR reviewer activity and duration of PRs for each contributor to this repo
        # pr_stats_tuple: Tuple[List[Dict[str, any]], List[Dict[str, any]]] = self.get_pr_stats(
        #     env_dict["repo_owner"], repo_name, since_date, until_date)
        dict_pr_stats: Dict = self.get_pr_stats(
            env_dict["repo_owner"], repo_name, since_date, until_date
        )

        # unpack the tuple into 2 list of dict - which we'll process in the user loop below
        list_dict_commenter_stats: List[Dict[str, any]] = dict_pr_stats.get(
            "commenters_stats", []
        )
        list_dict_pr_durations: List[Dict[str, any]] = dict_pr_stats.get(
            "contributor_pr_durations", []
        )
        prs_durations_dict: Dict[str, datetime] = dict_pr_stats.get(
            "prs_durations_dict", {}
        )
        # No activity in this repo, just return the empty dict
        if (
            len(list_dict_commenter_stats) == 0
            and len(list_dict_pr_durations) == 0
            and len(prs_durations_dict) == 0
        ):
            return dict_return

        # Get average PR duration for this entire repo during the time period
        prs_durations_list: List[datetime] = list(prs_durations_dict.values())
        dict_repo_stats["avg_pr_duration"] = (
            np.mean(prs_durations_list) if len(prs_durations_dict) > 0 else 0
        )
        dict_repo_stats["median_pr_duration"] = (
            np.median(prs_durations_list) if len(prs_durations_dict) > 0 else 0
        )

        # Returns the total number of commits authored by all contributors.
        # In addition, the response includes a Weekly Hash (weeks array) with the following info:
        # w - Start of the week, given as a Unix timestamp.
        # a - Number of additions
        # d - Number of deletions
        # c - Number of commits
        # IMPORTANT - for repos with > 10K commits, this REST endpoint now returns zero for additions and deletions
        # See https://github.blog/changelog/2023-11-29-upcoming-changes-to-repository-insights/
        # If you really need this data on large repos, consider using this git command instead:
        # git log --pretty="format:%m%ad----%ae%n%-(trailers:only,unfold)" --date=raw --shortstat --no-renames --no-merges
        url = f"{GITHUB_API_BASE_URL}/repos/{env_dict['repo_owner']}/{repo_name}/stats/contributors"

        # Get a list of contributors to this repo
        response = self.github_request_exponential_backoff(url)
        if response is not None and isinstance(response, list):
            contributors_pages: List = response
            for contributors_page in contributors_pages:
                for contributor in contributors_page:
                    if contributor is None:
                        continue

                    first_commit_date: date = None
                    contributor_name: str = None
                    contributor_username: str = None
                    contributor_stats: dict = None
                    contributor_already_added: bool = False
                    days_since_first_commit: int = 0

                    contributor_author = contributor.get("author", {})
                    contributor_username = contributor_author.get("login", "")
                    if contributor_username == "":
                        continue

                    user_attributes: dict = self.get_github_user_attributes(
                        contributor_username
                    )

                    contributor_name = user_attributes.get(
                        "contributor_name", contributor_username
                    )

                    print(
                        f'\t{repo_name}: {contributor_name} ({contributor_username}) from {since_date.strftime("%Y-%m-%d")} to {until_date.strftime("%Y-%m-%d")}'
                    )
                    first_commit_date = self.get_first_commit_date(
                        env_dict["repo_owner"], repo_name, contributor_username
                    )
                    num_workdays = max_num_workdays

                    # If their first commit is < roughly 6 weeks ago (configurable),
                    # don't measure contributions.
                    if first_commit_date is not None:
                        days_since_first_commit = get_workdays(
                            first_commit_date, date.today()
                        )
                    if days_since_first_commit < MIN_WORKDAYS_AS_CONTRIBUTOR:
                        continue

                    # Conditionally override num workdays if they started in during the evaluated interval
                    if (
                        first_commit_date is not None
                        and first_commit_date > since_date
                        and first_commit_date < until_date
                    ):
                        num_workdays = get_workdays(first_commit_date, until_date)

                    # If the contributor just started, don't include their stats
                    if num_workdays == 0:
                        continue

                    # If the contributor username is already in the array, use the one with the earliest date
                    # and add in the other stats to the existing stats
                    for contributor_stats_dict in list_dict_contributor_stats:
                        if (
                            contributor_stats_dict["contributor_name"]
                            == contributor_name
                        ):
                            contributor_stats = contributor_stats_dict
                            contributor_already_added = True
                            print(
                                f"\t\tMerging stats for {contributor_name} ({contributor_username}) with previously found contributor {contributor_stats_dict['contributor_name']} ({contributor_stats_dict['contributor_username']})"
                            )
                            contributor_stats["repo"] = (
                                f"{contributor_stats['repo']},{repo_name}"
                            )
                            if (
                                first_commit_date
                                < contributor_stats["contributor_first_commit_date"]
                            ):
                                contributor_stats["contributor_first_commit_date"] = (
                                    first_commit_date
                                )
                            break

                    # if this is the first time we've seen this contributor, init a dict
                    if contributor_stats is None:
                        contributor_stats = {
                            "repo": repo_name,
                            "contributor_name": contributor_name,
                            "contributor_nodeid": user_attributes.get(
                                "contributor_nodeid", ""
                            ),
                            "contributor_username": contributor_username,
                            "stats_beginning": since_date,
                            "stats_ending": until_date,
                            "contributor_first_commit_date": first_commit_date,
                            "num_workdays": num_workdays,
                            "commits": 0,
                            "prs": 0,
                            "review_comments": 0,
                            "changed_lines": 0,
                            "avg_pr_duration": 0.0,
                            "median_pr_review_duration": 0.0,
                            "avg_code_movement_per_pr": 0,
                        }

                    # Get a list of contributions per week. If they are in the time window, accumulate.
                    contributor_weekly_contribs_list: list = contributor.get(
                        "weeks", []
                    )
                    for weekly_stat in contributor_weekly_contribs_list:
                        weekly_stat_utc: int = weekly_stat.get("w", None)
                        weekly_date: date = (
                            weekly_stat_utc
                            and datetime.utcfromtimestamp(weekly_stat_utc).date()
                        )

                        if (
                            weekly_date
                            and weekly_date >= since_date
                            and weekly_date < until_date
                        ):
                            contributor_stats["commits"] += weekly_stat.get("c", 0)
                            contributor_stats["changed_lines"] += weekly_stat.get(
                                "d", 0
                            ) + weekly_stat.get("a", 0)

                    for dict_commenter_stats in list_dict_commenter_stats:
                        if (
                            dict_commenter_stats["commenter_name"]
                            == contributor_username
                        ):
                            contributor_stats[
                                "review_comments"
                            ] += dict_commenter_stats["num_comments"]

                    # Add PR durations from get_pr_stats: this dictionary has 3 entries:
                    # contributor_name, total_duration, and num_prs. From this we can get an avg duration
                    for dict_pr_durations in list_dict_pr_durations:
                        if (
                            dict_pr_durations["contributor_name"]
                            == contributor_username
                        ):
                            # Add PRs from get_pr_stats
                            contributor_stats["prs"] += dict_pr_durations["num_prs"]
                            avg_duration: float = (
                                dict_pr_durations["total_duration"]
                                / dict_pr_durations["num_prs"]
                                if dict_pr_durations["num_prs"] != 0
                                else 0.0
                            )
                            # add this avg duration to a dict with list of durations for this user, to be averaged later.
                            if contributor_username not in dict_avg_durations:
                                dict_avg_durations[contributor_name] = [avg_duration]
                            else:
                                dict_avg_durations[contributor_username].append(
                                    avg_duration
                                )

                    # Only save stats if there are stats to save
                    if (
                        contributor_stats["commits"] == 0
                        and contributor_stats["prs"] == 0
                        and contributor_stats["review_comments"] == 0
                        and contributor_stats["changed_lines"] == 0
                    ):
                        continue

                    # Since a given contributor will only show up in one page of the results
                    # for a given repo, there is no need to aggregate a single contributor's
                    # results across pages
                    #
                    # Normalize per workday in lookback period
                    else:
                        if contributor_stats["commits"] > 0:
                            contributor_stats["commits_per_day"] = round(
                                contributor_stats["commits"] / num_workdays, 3
                            )
                        if contributor_stats["changed_lines"] > 0:
                            contributor_stats["changed_lines_per_day"] = round(
                                contributor_stats["changed_lines"] / num_workdays, 3
                            )
                            if contributor_stats["prs"] > 0:
                                contributor_stats["avg_code_movement_per_pr"] = round(
                                    contributor_stats["changed_lines"]
                                    / contributor_stats["prs"],
                                    3,
                                )
                        if contributor_stats["prs"] > 0:
                            contributor_stats["prs_per_day"] = round(
                                contributor_stats["prs"] / num_workdays, 3
                            )
                        if contributor_stats["review_comments"] > 0:
                            contributor_stats["review_comments_per_day"] = round(
                                contributor_stats["review_comments"] / num_workdays, 3
                            )

                    if not contributor_already_added:
                        list_dict_contributor_stats.append(contributor_stats)

        # This means we have nothing returned from our attempt to get contributors for a repo
        else:
            print(
                f"\tNo contributors found for {repo_name} between {since_date} and {until_date}."
            )
            return None

        # Across all contributions for all repos, calculate avg duration of PRs.
        # Generate a new dictionary with contributor_name to average duration
        dict_average_pr_review_durations: Dict[str, float] = {
            contributor: np.mean(durations) if durations else 0
            for contributor, durations in dict_avg_durations.items()
        }
        dict_median_pr_review_durations: Dict[str, float] = {
            contributor: np.median(durations) if durations else 0
            for contributor, durations in dict_avg_durations.items()
        }
        # Merge this into contributors list of dict
        for contributor in list_dict_contributor_stats:
            contributor_name = contributor["contributor_name"]
            # Check if the user_name exists in the dict_median_pr_review_durations dictionary
            if contributor_name in dict_median_pr_review_durations:
                # Add a new key-value pair for the average duration
                contributor["median_pr_review_duration"] = (
                    dict_median_pr_review_durations[contributor_name]
                )
            if contributor_name in dict_average_pr_review_durations:
                # Add a new key-value pair for the average duration
                contributor["avg_pr_duration"] = dict_average_pr_review_durations[
                    contributor_name
                ]
        # Add the number of contributors to the repo stats
        dict_repo_stats["num_contributors"] = len(list_dict_contributor_stats)
        # Prepare the return dictionary
        dict_return["contributor_stats"] = list_dict_contributor_stats
        dict_return["repo_stats"] = dict_repo_stats
        return dict_return

    def get_repos_by_single_topic(
        self, topic: str, since_date_str: str, until_date_str: str
    ) -> list:
        """
        Get all the repos within the org that share the same topic, pushed to in the date range
        """
        url: str = (
            f"https://api.github.com/search/repositories?q=topic:{topic}"
            f'+org:{self.dict_env["repo_owner"]}'
            f"+pushed:>={since_date_str}+pushed:<{until_date_str}"
        )

        response = self.github_request_exponential_backoff(url)

        item_list: Dict = {}
        repo_list_returned: list = []
        if response is not None and isinstance(response, List) and len(response) > 0:
            pages_list: List = response
            for page in pages_list:
                item_list = page.get("items", [])
                for item in item_list:
                    repo_list_returned.append(item["name"])
            return repo_list_returned

    def get_repos_by_topic(
        self, topic: str, topic_exclude: str, since_date_str: str, until_date_str: str
    ) -> list:
        """Get all the repos within the org that share the same topic, pushed to in the date range,
        also allowing for the special case of "all" to get all repos in the org."""
        repo_list_returned: list = []
        item_list_single_topic: list = []
        # Special case - "all" means all repos in the org. Get the dict of all repo-topics and iterate thru it
        if topic and topic.lower() == "all":  # Case insensitive match
            for topic in self.dict_env["dict_all_repo_topics"]:
                item_list_single_topic = self.get_repos_by_single_topic(
                    topic, since_date_str, until_date_str
                )
                repo_list_returned.extend(item_list_single_topic)
            # Override "all" with the actual list of repos
            self.dict_env["repo_names"] = repo_list_returned
        else:
            repo_list_returned = self.get_repos_by_single_topic(
                topic, since_date_str, until_date_str
            )
        return repo_list_returned

    def get_organization_repo_names(self, org_name: str) -> List[str]:
        """
        Fetches names of all repositories owned by the specified organization.

        Args:
            org_name (str): The name of the organization.

        Returns:
            List[str]: A list of repository names owned by the organization.
        """
        # GitHub API endpoint for fetching organization repositories.
        url = f"https://api.github.com/orgs/{org_name}/repos"

        # Initial parameters for pagination
        params = {
            "per_page": 100,  # Adjust per_page to the maximum allowed to minimize the number of requests
            "type": "all",  # Fetch all repos including forks, sources, and private repos if the token has permissions
        }

        # Use the exponential backoff helper function to handle API rate limits and possible errors.
        repos_pages = self.github_request_exponential_backoff(url, params)

        # Extract just the names of the repositories from each page of results
        repo_names = [repo["name"] for page in repos_pages for repo in page]

        return repo_names

    def get_repo_topics(self, repo_owner: str, repo_names: list) -> dict:
        """
        Returns a dict of {repo_name: [list of topics], ...}
        if repo_names contains "all", it will get all the topics for all the repos in the org
        """
        dict_repo_topics: dict = {}
        url_base: str = f"https://api.github.com/repos/{repo_owner}"

        # Override the repo_names list with all the repos in the org
        if "all" in repo_names:
            repo_names = self.get_organization_repo_names(repo_owner)
            self.dict_env["repo_names"] = repo_names
            print(
                f"Getting all topics for all {len(repo_names)} repos in {repo_owner}."
            )

        for repo in repo_names:
            url = f"{url_base}/{repo}/topics"
            response = self.github_request_exponential_backoff(url)
            if (
                response is not None
                and isinstance(response, List)
                and len(response) > 0
            ):
                pages_list: List = response
                for page in pages_list:
                    if "names" in page:
                        dict_repo_topics[repo] = page.get("names", [])

        return dict_repo_topics

    def prepare_for_storage(self, list_dict_contributor_stats: list) -> pd.DataFrame:
        df: pd.DataFrame = pd.DataFrame()
        if len(list_dict_contributor_stats) > 0:
            # Clean up
            df = pd.DataFrame(list_dict_contributor_stats)

            df["commits_per_day"] = df["commits_per_day"].fillna(0)
            df["changed_lines_per_day"] = df["changed_lines_per_day"].fillna(0)
            df["prs_per_day"] = df["prs_per_day"].fillna(0)
            if "review_comments_per_day" not in df:
                df["review_comments_per_day"] = 0
            df["review_comments_per_day"] = df["review_comments_per_day"].fillna(0)
            df["avg_pr_duration"] = df["avg_pr_duration"].fillna(0)
            df["median_pr_review_duration"] = df["median_pr_review_duration"].fillna(0)
            df["avg_code_movement_per_pr"] = df["avg_code_movement_per_pr"].fillna(0)

            # Remove any rows where there are no commits and no PRs.
            # I'm seeing Github return PR comments from people who were not involved in the lookback
            # period. I haven't diagnosed this. This is a hacky way to get rid of them.
            # Obvy, if PRs and commits are zero, so are changed_lines.
            df = df[~((df["commits"] == 0) & (df["prs"] == 0))]

            # Calculate the difference from the mean for each row in the 'pr' column
            df["prs_diff_from_mean"] = (df["prs"] - df["prs"].mean()).round(3)

            # Decile time!
            df = self.add_ntile_stats(df)

            # Create a curved score from 1-100 based on the average decile
            df = self.curve_scores(df, "avg_ntile", "curved_score")

            # Sort the DataFrame by 'curved_score' in descending order
            df = df.sort_values(by="curved_score", ascending=False)
        else:
            print(f"\t No contributors found")
        return df

    def store_contributors(self) -> None:
        df = pd.DataFrame(list(gdict_user_attributes.values()))
        self.storage_manager.upsert_contributors(df)
        return

    def store_contributor_stats(self, df: pd.DataFrame) -> None:
        self.store_contributors()
        # storage_manager.save_df_to_csv(df, filename)
        # storage_manager.save_summary_stats_csv(df, filename)
        """
        This upsert does the following:
        Clears out the staging table
        Dumps the dataframe to the staging table
        Merges the staging table contents to the main table (prevents dupes)
        """
        count: int = self.storage_manager.upsert_contributor_stats_dataframe(
            df,
            self.storage_manager.get_db_env().get("snowflake_table_name", ""),
            self.storage_manager.get_db_env().get("snowflake_table_name_staging", ""),
        )
        return

    def store_repo_stats(self, list_dict_repo_stats: List[Dict[str, Any]]) -> int:
        count: int = self.storage_manager.store_repo_stats(list_dict_repo_stats)
        return count

    def store_repo_topics(self, dict_repo_topics: dict) -> int:
        count: int = self.storage_manager.store_repo_topics(dict_repo_topics)
        return count

    def merge_repo_stats(
        self,
        list_dict_repo_stats: List[Dict[str, Any]],
        list_dict_contributors_stats: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Merge the contributor_stats into the repo_stats
        This is where we accumulate the PRs and commits for each repo from our individual contributor stats
        and add them to the repo stats list of dicts
        Once this is done, we have a list of dictionaries that is ready for storage in DB
        Returns: list of dicts of repo stats, ready for storage (which is unnecessary since it's modified in place)
        The reason for this requires a bit of python knowledge about mutable vs immutable objects.
        Immutable objects such as strings, numbers, and tuples are immutable.
        This means that once they are created, they cannot be changed.
        Mutable objects such as lists, sets, and dictionaries are mutable.
        This means that they can be changed after they are created.
        Since we're changing a list, it is mutable and therefore we don't need to return it explicitly.
        """
        # Sum PRs and commits for each repo using defaultdict
        dict_repo_stats_aggregated = defaultdict(
            lambda: {"num_prs": 0, "num_commits": 0}
        )
        for stat in list_dict_contributors_stats:
            dict_repo_stats_aggregated[stat["repo"]]["num_prs"] += stat["prs"]
            dict_repo_stats_aggregated[stat["repo"]]["num_commits"] += stat["commits"]

        # Add num_prs and num_commits to each repo stats
        for repo_stat in list_dict_repo_stats:
            if type(repo_stat) == "Dict":
                repo_name = repo_stat.get("repo_name", "")
                if repo_name in dict_repo_stats_aggregated:
                    repo_stat.update(dict_repo_stats_aggregated[repo_name])
        # this return is unnecessary, but it's here for clarity, since the list is modified in place
        return list_dict_repo_stats

    def prep_repo_topics(self) -> None:
        """this initializes the process of getting repo topics for all repos in the org
        the format of this dictionary is {repo_name: [list of topics], ...}
        It then creates a list of all the repo names from the dictionary of repo topics
        """
        if len(self.dict_env["dict_all_repo_topics"]) > 0:
            return
        dict_repo_topics: dict = self.get_repo_topics(
            self.dict_env["repo_owner"], self.dict_env["repo_names"]
        )
        if len(dict_repo_topics) > 0:
            self.dict_env["dict_all_repo_topics"] = dict_repo_topics.copy()
            self.store_repo_topics(dict_repo_topics)
            self.dict_env["repo_names"] = list(dict_repo_topics.keys())

    def get_repo_data_over_months(self) -> None:
        """this initializes the process of getting repo data over the months in the lookback period
        it first starts by getting the .env settings, establishing the member dict_env, for access to settings
        """
        if self.dict_env is None:
            print(f"Missing env vars - README")
            return

        since_date_str: str = self.since_date.strftime("%Y-%m-%d")

        # Get the topics for all the repos. Need this now because we accept "all" as a topic, below
        # We stash the dict of repo_topics in the settings for use later
        # This also creates a list of all the repo names from the dictionary of repo topics
        self.prep_repo_topics()

        # If there were no repo_names in .env, we can pull the repos based on the topic (which can be "all")
        # if len(self.dict_env["repo_names"]) == 0 and self.dict_env["topic_name"] is not None:
        #     until_date = get_end_of_last_complete_month()
        #     until_date_str: str = until_date.strftime('%Y-%m-%d')
        #     self.dict_env["repo_names"] = self.get_repos_by_topic(
        #         self.dict_env["topic_name"], self.dict_env["topic_exclude_name"], since_date_str, until_date_str)

        # Filter out repos in exclude list
        self.dict_env["repo_names"] = [
            item
            for item in self.dict_env["repo_names"]
            if item not in self.dict_env["repo_names_exclude"]
        ]

        if (
            len(self.dict_env["repo_names"]) == 0
            and self.dict_env["topic_name"] is None
        ):
            print(f"Either TOPIC or REPO_NAMES must be provided in .env")
            sys.exit()

        # Tell the user what they're getting
        print(
            f'Stats for {len(self.dict_env["repo_names"])} {self.dict_env["repo_owner"]} repos since {since_date_str}:'
        )
        # Don't print out too many
        if len(self.dict_env["repo_names"]) < MAX_REPO_NAMES_TO_PRINT:
            print(f", ".join(self.dict_env["repo_names"]))
        else:
            print(f"Too many repos to list (more than {MAX_REPO_NAMES_TO_PRINT}).")
        # END - TO DO move into the loop section below

        # Loop for each month in the lookback period and pull stats
        # The first until_date is now. Step back 1 month each iteration
        # Initialize contributor_stats as an empty dictionary. Then add to it each iteration
        list_dict_contributors_stats: List[Dict[str, Any]] = []

        count_repos: int = 0
        for repo in self.dict_env["repo_names"]:
            # print progress - count offset and number of repos in dictionary
            count_repos += 1
            print(f"\nRepo #{count_repos} of {len(self.dict_env['repo_names'])}...\n")
            # Initialize repo_stats as an empty dictionary. Then add to it each iteration.
            list_dict_repo_stats: List[Dict[str, Any]] = []
            # Initialize until_date to the last day of the last complete month (which is only today if today is the last day)
            current_until_date: date = None
            if self.dict_env["months_count"] == 0:
                current_until_date = get_end_of_last_complete_month()
            else:
                current_until_date = get_last_day_months_ago(
                    self.dict_env["months_lookback"]
                )
            # Initialize the number of months to look back
            months_count: int = 0
            # this allows us to selectively pull a range of months from the past and not all the months to the present
            # Needed when we find bad data and need to patch!
            # example: if it's August and you want to just get data for June (not July), set months_lookback=2 and months_count=1
            if self.dict_env["months_count"] > 0:
                months_count = self.dict_env["months_count"]
            else:
                months_count = self.dict_env["months_lookback"]
            for month_delta in range(1, months_count + 1):
                # Calculate the start (since_date) of the month period (should be the first day of the month)
                # current_until_date is currently the last day of the last month, so we need to step back a month, then ahead one day
                since_date = get_first_day_of_month(current_until_date)

                print(
                    f'\nGetting stats for {self.dict_env["repo_owner"]}/{repo} from {since_date.strftime("%Y-%m-%d")} to {current_until_date.strftime("%Y-%m-%d")}'
                )
                current_period_contributors_stats: List[Dict[str, Any]] = []
                dict_stats: Dict[str, Any] = self.get_contributors_stats(
                    self.dict_env, repo, since_date, current_until_date
                )
                current_period_contributors_stats = dict_stats.get(
                    "contributor_stats", []
                )
                current_period_repo_stats = dict_stats.get("repo_stats", {})
                # Extend our list: store this iteration's list of dict of stats in our main list
                if len(current_period_contributors_stats) > 0:
                    list_dict_contributors_stats.extend(
                        current_period_contributors_stats
                    )
                    list_dict_repo_stats.append(current_period_repo_stats)
                # Update until_date for the next iteration to step back another month
                current_until_date = since_date

            # Store this repo data for the whole sequence of months
            if len(list_dict_contributors_stats) > 0:
                df: pd.DataFrame = self.prepare_for_storage(
                    list_dict_contributors_stats
                )
                if df is not None and not df.empty > 0:
                    # print(
                    #     f"\tRetrieved stats for {len(df)} contributors. Merging them.")
                    self.store_contributor_stats(df)
            if list_dict_repo_stats and len(list_dict_repo_stats) > 0:
                list_dict_repo_stats = self.merge_repo_stats(
                    list_dict_repo_stats, list_dict_contributors_stats
                )
                print(
                    f"\tRetrieved stats for {len(list_dict_repo_stats)} repos. Storing them."
                )
                self.store_repo_stats(list_dict_repo_stats)

    def retrieve_and_store_org_repos(self) -> None:
        """
        Retrieves all repositories for the organization set in the environment variable REPO_OWNER
        and stores their names and topics in the Snowflake 'repo_topics' table.
        """
        repo_owner = self.dict_env.get("repo_owner")
        if not repo_owner:
            print(
                "Repository owner (organization) not specified in environment variables."
            )
            return

        # Base GitHub API URL for fetching organization repositories
        repos_url = f"{GITHUB_API_BASE_URL}/orgs/{repo_owner}/repos?per_page={MAX_ITEMS_PER_PAGE}"
        repos_response = self.github_request_exponential_backoff(repos_url)

        # Process each repository to fetch its topics
        repo_topics_data = []
        for repo_page in repos_response:
            for repo in repo_page:
                repo_name = repo.get("name")
                topics_url = (
                    f"{GITHUB_API_BASE_URL}/repos/{repo_owner}/{repo_name}/topics"
                )
                topics_response = self.github_request_exponential_backoff(
                    topics_url,
                    params={"accept": "application/vnd.github.mercy-preview+json"},
                )

                if topics_response:
                    # Assuming the first page has all topics
                    topics_list = topics_response[0].get("names", [])
                    for topic in topics_list:
                        repo_topics_data.append(
                            {"repo_name": repo_name, "repo_topic": topic}
                        )

        # Convert to DataFrame for easy storage
        df_repo_topics = pd.DataFrame(repo_topics_data)

        # Assuming write_pandas is a method you have for writing DFs to Snowflake
        self.storage_manager.insert_new_repo_topics(df_repo_topics)

        print(f"Stored {len(df_repo_topics)} repository topics in Snowflake.")

    def fetch_and_store_pr_review_comments(
        self, repo_names: List[str], since_date: date = None, until_date: date = None
    ):
        """
        Fetches PR review comments for given repositories within a specified date range and stores them in Snowflake.
        If since_date and until_date are None, all comments are fetched without date filtering.

        Args:
            repo_names (List[str]): A list of repository names.
            since_date (date, optional): The start date for filtering comments (inclusive).
            until_date (date, optional): The end date for filtering comments (inclusive).
        """
        repo_owner: str = self.dict_env.get("repo_owner")

        for repo_name in repo_names:
            print(f"Fetching PR review comments for {repo_name}...")
            comments_url = (
                f"{GITHUB_API_BASE_URL}/repos/{repo_owner}/{repo_name}/pulls/comments"
            )

            review_comments_pages = self.github_request_exponential_backoff(
                comments_url
            )

            review_comments_data: List[Dict[str, any]] = []
            for page in review_comments_pages:
                for comment in page:
                    # Convert comment's created_at to date for comparison
                    comment_date = datetime.strptime(
                        comment["created_at"], "%Y-%m-%dT%H:%M:%SZ"
                    ).date()

                    # If dates are provided, filter comments by date range
                    # Sometimes user is None, suggesting perhaps the github user was deleted since the PR comment was made
                    if (not since_date or comment_date >= since_date) and (
                        not until_date or comment_date <= until_date
                    ):
                        review_comments_data.append(
                            {
                                "comment_id": comment["id"],
                                "repo_name": repo_name,
                                "pr_number": comment["pull_request_url"].split("/")[-1],
                                "user_login": (
                                    comment["user"]["login"]
                                    if comment["user"] is not None
                                    else ""
                                ),
                                "body": comment["body"],
                                "created_at": comment["created_at"],
                            }
                        )

            # Convert to DataFrame
            df_review_comments = pd.DataFrame(review_comments_data)

            rows: int = self.storage_manager.insert_pr_review_comments(
                df_review_comments
            )
            print(f"Stored {rows} review comments for {repo_name} in Snowflake.")
