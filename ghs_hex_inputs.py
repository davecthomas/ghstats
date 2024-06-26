"""
NOTE: This code WILL NOT work outside of a Hex environment. It is designed to be used within the Hex ecosystem.
You must have a Hex account and be logged in to run this code, along with a valid datasource set up to use the data.
See the dbschema.sql for setting up a Snowflake datasource in Hex here: https://hex.tech/docs/snowflake
"""
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta

def get_first_day_of_month_one_year_ago() -> datetime.date:
    # Get today's date
    today = datetime.datetime.now().date()
    
    # Subtract one year from today's date
    one_year_ago = today - relativedelta(years=1)
    
    # Adjust to the first day of that month
    first_day_of_month_one_year_ago = one_year_ago.replace(day=1)
    
    return first_day_of_month_one_year_ago

def present_input_selections(metric_selector, repo_aggregated: bool, repo_topic_aggregated: bool, start_date: datetime.date, multiselect_contributors, multiselect_repo_names, multiselect_repo_topics)->dict:
    default_metric_selector: list = ["commits_per_day", "prs_per_day", "changed_lines_per_day", "review_comments_per_day", "avg_pr_duration", "avg_code_movement_per_pr"]
    dict_inputs: dict = {
        "metric_selector": metric_selector,
        "multiselect_contributors": multiselect_contributors,
        "repo_aggregated": repo_aggregated,
        "repo_topic_aggregated": repo_topic_aggregated,
        "multiselect_repo_names": multiselect_repo_names,
        "multiselect_repo_topics": multiselect_repo_topics,
        "start_date": start_date
    }
    # Default metric_selector
    if len(metric_selector) == 0: 
        dict_inputs["metric_selector"] = default_metric_selector

    # Calculate the default start_date if not provided
    if start_date is None:
        start_date = get_first_day_of_month_one_year_ago()
    dict_inputs["start_date"] = start_date

    # Override logic for aggregation mode and nullify contributors if repo_topic_aggregated is True
    if repo_topic_aggregated and (dict_inputs["repo_aggregated"] is True or len(dict_inputs["multiselect_contributors"]) > 0):
        dict_inputs["repo_aggregated"] = False  # Ensure repo aggregation is not simultaneously selected
        dict_inputs["multiselect_contributors"] = []  # Nullify contributors selection
        if len(dict_inputs["multiselect_repo_topics"]) == 0: 
            print("When repo topic aggregation is requested, at least one repo topic must be selected")
            return None
        else:
            print("Repo topic aggregation is selected, overriding repo aggregation and nullifying contributors selection.")

    # Display the explanation and values to be used
    print(f"Creating Hex plot output using the following metrics: {dict_inputs['metric_selector']}")
    print(f"Date range of analysis period: {start_date} to the present")

    if dict_inputs["repo_topic_aggregated"]:
        print(f"Aggregating results across all contributors to all repos sharing the same repo topic {dict_inputs['multiselect_repo_topics']}.")
    elif dict_inputs["repo_aggregated"]:
        print(f"Aggregating results across all contributors within these repositories: {dict_inputs['multiselect_repo_names']}..")
    else:
        if dict_inputs["multiselect_contributors"]:
            print(f"Contributors: {dict_inputs['multiselect_contributors']}")

    return dict_inputs

# Call the function to display the values to be used
gdict_inputs: dict = present_input_selections(metric_selector, repo_aggregated, repo_topic_aggregated, start_date, multiselect_contributors, multiselect_repo_names, multiselect_repo_topics)
start_date_date = gdict_inputs["start_date"]