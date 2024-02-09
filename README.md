# ghstats

Grab statistics from GitHub for a set of repos for a given organization and save them to csv and Snowflake, if configured.

## Overview

This summary aims to provide comprehensive insights into the contributions and activities of individual contributors within a GitHub repository, allowing for detailed analysis and comparison across multiple dimensions of productivity and engagement.

# Install

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 ghstats.py
```

# Settings - the .env file should have

### Your API token

GITHUB_API_TOKEN=

### Your Org name

REPO_OWNER=my_orgname

### Comma-separated list of repos

REPO_NAMES=repo1,repo2,repo3

### How many months of data you want to pull

DEFAULT_MONTHS_LOOKBACK=3

### If the contributor is new to the repo, what's the minimum time before you want to include them in results?

MIN_WORKDAYS_AS_CONTRIBUTOR=30

### Topic - if your Github org uses topics to group repos by team

TOPIC=your-topic-name

### Users to exclude from measuring. Often there are bots that comment on PRs. Don't measure stats on these.

USER_EXCLUDE=username

# CSV File Output Format

The CSV output file summarizes various productivity and engagement metrics for contributors within a GitHub repository.

## Column Descriptions

| Column Name                     | Description                                                                                                                                          |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `repo`                          | The name of the GitHub repository.                                                                                                                   |
| `contributor_name`              | The full name of the contributor as listed in their GitHub profile.                                                                                  |
| `contributor_username`          | The GitHub username of the contributor.                                                                                                              |
| `curved_score`                  | A calculated score that may factor in various metrics to assess the contributor's performance, adjusted or "curved" based on specific criteria.      |
| `stats_beginning`               | The start date for the period over which the statistics were calculated.                                                                             |
| `stats_ending`                  | The end date for the period over which the statistics were calculated.                                                                               |
| `contributor_first_commit_date` | The date of the first commit made by the contributor within the repository.                                                                          |
| `num_workdays`                  | The number of days the contributor was actively contributing to the repository, based on commit history.                                             |
| `commits`                       | The total number of commits made by the contributor within the specified period.                                                                     |
| `prs`                           | The total number of pull requests submitted by the contributor.                                                                                      |
| `review_comments`               | The total number of review comments made by the contributor on pull requests.                                                                        |
| `changed_lines`                 | The total number of lines of code added or removed by the contributor's commits.                                                                     |
| `avg_pr_duration`               | The average duration (in days) that the contributor's pull requests remained open before being merged or closed.                                     |
| `avg_code_movement`             | An average measure of code changes per pull request, potentially considering both additions and deletions.                                           |
| `commits_per_day`               | The average number of commits made by the contributor per active workday.                                                                            |
| `changed_lines_per_day`         | The average number of lines changed by the contributor per active workday.                                                                           |
| `avg_code_movement_per_pr`      | The average amount of code changes (additions and deletions) per pull request submitted by the contributor.                                          |
| `prs_per_day`                   | The average number of pull requests submitted by the contributor per active workday.                                                                 |
| `review_comments_per_day`       | The average number of review comments made by the contributor per active workday.                                                                    |
| `prs_ntile`                     | A percentile ranking of the contributor based on the number of pull requests submitted, compared to other contributors in the repository.            |
| `commits_ntile`                 | A percentile ranking of the contributor based on the number of commits made, compared to other contributors in the repository.                       |
| `lines_of_code_ntile`           | A percentile ranking of the contributor based on the number of lines of code changed, compared to other contributors in the repository.              |
| `review_comments_ntile`         | A percentile ranking of the contributor based on the number of review comments made, compared to other contributors in the repository.               |
| `avg_ntile`                     | An average of the percentile rankings across the different metrics, providing a general performance indicator of the contributor relative to others. |
