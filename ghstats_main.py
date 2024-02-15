from ghstats import GhsGithub

# TO DO - get all repo_topics (get all repos, then get their topics and store them)
if __name__ == "__main__":
    ghs: GhsGithub = GhsGithub()
    ghs.get_repo_data_over_months()
