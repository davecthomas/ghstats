from ghstats import GhsGithub

if __name__ == "__main__":
    ghs: GhsGithub = GhsGithub()
    ghs.get_repo_data_over_months()
