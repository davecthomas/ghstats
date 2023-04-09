# ghstats
Grab statistics from GitHub

# Install
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

# Settings
.env file should have
GITHUB_API_TOKEN=
REPO_OWNER=my_orgname
REPO_NAMES=repo1,repo2,repo3
DEFAULT_MONTHS_LOOKBACK=3
