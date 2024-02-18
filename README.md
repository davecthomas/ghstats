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

# Contributor Stats Metrics

## Utility and Considerations

These metrics offer a comprehensive view of individual and team productivity in software development. However, they should not be used in isolation. High volume of commits or PRs does not always equate to high quality or impactful work. It's important to balance quantitative data with qualitative assessments of contribution quality, the complexity of tasks, and the collaborative aspects of software development. Additionally, the context of each project and the specific roles of contributors should be considered when evaluating these metrics.

It is crucial that software leaders understand the history and context of the individuals, teams, and repos being measured. For example

- a star contributor who joins a team working on a repo of poor quality with an unmaintainable technical debt load will appear less productive than a mediocre developer on a newer codebase of higher quality. This is an important reason to avoid comparing metrics across repos of vastly different contraints.

## Numeric Metrics, which are stored in timeseries, averaged across each month

- `curved_score` - A normalized score representing the overall contribution level of a contributor, scaled to a curve. This metric can help identify top performers and areas where contributors may need support. It is based on the average decile ranking across each of the key numeric metrics deciles. These key metrics include `prs_per_day`, `review_comments_per_day`, `commits_per_day`, and `changed_lines_per_day`.

- `num_workdays` - The number of workdays during the stats collection period (in the month). This metric provides context for other per-day metrics (the denominator), helping to normalize data across different time frames.

- `commits` - The total number of commits a contributor has made. This metric is a direct indicator of contribution volume but should be considered alongside the quality and impact of changes.

- `prs` - The total number of pull requests (PRs) a contributor has made. This metric indicates engagement and contribution to project development. High numbers suggest active participation, but the metric should be balanced with PR quality.

- `review_comments` - The total number of review comments made by a contributor. This metric indicates a contributor's involvement in code review processes, crucial for maintaining code quality and collaborative improvement.

- `changed_lines` - The total number of lines of code added or removed. While a high number may indicate significant contributions, it's essential to consider the context and impact of the changes. This is especially crucial in repos where libraries of other contributors' work is committed.

- `avg_pr_duration` - The average duration that a contributor's PRs remain open. Shorter durations can indicate efficient workflow processes, but quality should not be compromised for speed.

- `avg_code_movement_per_pr` - The average number of lines of code changed per PR. This metric helps understand the scope and impact of contributions, with high values potentially indicating significant features or changes. Higher code movement may indicate larger "WIP" (work in process) size, which is a potential constraint in operational flow which has been quantitatively correlated to longer overall cycle time.

- `commits_per_day` - The average number of commits per workday. This metric offers insight into a contributor's daily productivity, though the impact and quality of commits are also important to consider.

- `changed_lines_per_day` - The average number of lines of code changed per workday. This metric reflects daily coding activity, providing a measure of how much code a contributor is impacting regularly.

- `prs_per_day` - The average number of PRs opened per workday. This metric helps gauge a contributor's active engagement in project development on a daily basis.

- `review_comments_per_day` - The average number of review comments made per workday. High values indicate active participation in code review processes, contributing to code quality and team collaboration.

- `prs_diff_from_mean` - The difference between a contributor's PRs and the mean PR count. This metric can highlight contributors who are significantly more or less active than average. This is a useful relative score if the decile scores aren't considered useful for whatever reason.

- `prs_ntile`, `commits_ntile`, `lines_of_code_ntile`, `review_comments_ntile`, `avg_pr_duration_ntile`- These n-tile metrics rank contributors on a scale (e.g., deciles) within specific categories (PRs, commits, lines of code, etc.). They are useful for identifying outliers and understanding distribution of productivity across the team.

- `avg_ntile` is an average of the above deciles. It is directly correlated to the `curved_score`.

# Database Column Descriptions (see dbschema.sql for the latest)

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
