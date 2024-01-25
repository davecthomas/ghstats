# ghstats

Grab statistics from GitHub

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
