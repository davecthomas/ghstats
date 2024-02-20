"""
NOTE: This code WILL NOT work outside of a Hex environment. It is designed to be used within the Hex ecosystem.
You must have a Hex account and be logged in to run this code, along with a valid datasource set up to use the data.
See the dbschema.sql for setting up a Snowflake datasource in Hex here: https://hex.tech/docs/snowflake
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hextoolkit
import matplotlib.dates as mdates
import datetime
from dateutil.relativedelta import relativedelta

# Establish connection using HexToolkit
hex_snowflake_conn = hextoolkit.get_data_connection('Contributor Stats')
session = hex_snowflake_conn.get_snowpark_session()


def create_and_process_dataframe(dict_inputs: dict, selected_metric: str) -> dict:
    """ 
    Returns a dict of {"df": the plottable dataset , "contributors_list": a list of contributors, if filtered in selectors}
    """
    dict_results: dict = {"df": None, "contributors_list": None}
    # Extracting necessary parameters from dict_inputs
    repo_aggregated_view = dict_inputs.get("repo_aggregated", False)
    repo_topic_aggregated_view = dict_inputs.get(
        "repo_topic_aggregated", False)
    username_filter = dict_inputs.get("multiselect_contributors", [])
    repo_filter = dict_inputs.get("multiselect_repo_names", [])
    topic_filter = dict_inputs.get("multiselect_repo_topics", [])
    start_date_date = dict_inputs.get(
        "start_date", datetime.date.today() - relativedelta(years=1))

    start_date_str = start_date_date.strftime('%Y-%m-%d')

    # Adjust the query based on filters applied
    contributors: str = ""
    query_conditions = []
    if 'multiselect_contributors' in dict_inputs and dict_inputs['multiselect_contributors']:
        contributors = "', '".join(dict_inputs['multiselect_contributors'])
        query_conditions.append(
            f'cs."contributor_name" IN (\'{contributors}\')')
    if 'multiselect_repo_names' in dict_inputs and dict_inputs['multiselect_repo_names']:
        repos = "', '".join(dict_inputs['multiselect_repo_names'])
        query_conditions.append(f'cs."repo" IN (\'{repos}\')')
    if 'multiselect_repo_topics' in dict_inputs and dict_inputs['multiselect_repo_topics']:
        topics = "', '".join(dict_inputs['multiselect_repo_topics'])
        query_conditions.append(f'rt."repo_topic" IN (\'{topics}\')')

    conditions_str = " AND ".join(
        query_conditions) if query_conditions else "1=1"

    query_template = f"""
    SELECT 
        cs."repo", 
        cs."stats_beginning", 
        cs."contributor_name", 
        cs."{selected_metric}" AS "selected_metric",
        rt."repo_topic",
        cs."changed_lines_per_day",
        cs."commits_per_day",
        cs."review_comments_per_day",
        cs."prs_per_day", 
        cs."avg_pr_duration", 
        cs."avg_code_movement_per_pr"
    FROM "contributor_stats" cs
    LEFT JOIN "repo_topics" rt ON cs."repo" = rt."repo_name"
    WHERE cs."stats_beginning" >= '{start_date_str}' AND ({conditions_str})
    ORDER BY cs."stats_beginning" ASC, cs."repo";
    """

    # Execute the query and load results into a pandas DataFrame
    df = session.sql(query_template).to_pandas()
    df['stats_beginning'] = pd.to_datetime(df['stats_beginning'])

    # Capture the contributor names before aggregation if username_filter is not empty
    if len(contributors) > 0:
        contributors_list = df[df['contributor_name'].isin(
            username_filter)]['contributor_name'].unique()
    else:
        contributors_list = []

    dict_results["contributors_list"] = contributors_list

    # Aggregation and calculation of repo_count for repo_topic_aggregated_view
    if repo_topic_aggregated_view:
        # Aggregate data
        agg_dict = {
            selected_metric: 'mean',  # Average of the selected metric
            'contributor_name': 'nunique',  # Count unique contributors
            'repo': 'nunique',  # Count unique repos
            # Sum of per day metrics
            'commits_per_day': 'sum',
            'prs_per_day': 'sum',
            'changed_lines_per_day': 'sum',
            'review_comments_per_day': 'sum',
            'avg_pr_duration': 'mean',
            'avg_code_movement_per_pr': 'mean'
        }

        df_agg = df.groupby(['repo_topic', 'stats_beginning']).agg(
            agg_dict).reset_index()

        # Calculate per contributor metrics
        for metric in ['commits_per_day', 'prs_per_day', 'changed_lines_per_day', 'review_comments_per_day']:
            per_contributor_metric = f"{metric}_per_contributor"
            df_agg[per_contributor_metric] = df_agg[metric] / \
                df_agg['contributor_name']

        df = df_agg.rename(
            columns={'contributor_name': 'contributor_count', 'repo': 'repo_count'})

    elif repo_aggregated_view:
        # Define aggregation dictionary
        agg_dict = {
            selected_metric: 'mean',  # Average of the selected metric
            'contributor_name': 'nunique',  # Count unique contributors
            # Sum of per day metrics
            'commits_per_day': 'sum',
            'prs_per_day': 'sum',
            'changed_lines_per_day': 'sum',
            'review_comments_per_day': 'sum',
            'avg_pr_duration': 'mean',
            'avg_code_movement_per_pr': 'mean'
        }

        # Perform aggregation
        df_agg = df.groupby(['repo', 'stats_beginning']
                            ).agg(agg_dict).reset_index()

        # Rename 'contributor_name' column to 'contributor_count' for clarity
        df_agg.rename(
            columns={'contributor_name': 'contributor_count'}, inplace=True)

        # Calculate "per contributor" metrics for the aggregated data
        for metric in ['commits_per_day', 'prs_per_day', "changed_lines_per_day", "review_comments_per_day"]:
            per_contributor_metric = f"{metric}_per_contributor"
            df_agg[per_contributor_metric] = df_agg[metric] / \
                df_agg['contributor_count']

        # Assign the aggregated DataFrame back to df
        df = df_agg

    # Handling duplicates and filtering based on the initial configuration
    if not repo_aggregated_view and not repo_topic_aggregated_view:
        df = df.drop_duplicates(
            subset=['contributor_name', 'repo', 'stats_beginning'], keep='first')

    dict_results["df"] = df

    return dict_results


def plot_data(dict_data: dict, dict_inputs: dict, selected_metric: str):
    df: pd.DataFrame = dict_data.get("df")
    contributors_list: list = dict_data.get("contributors_list", [])
    title: str = dict_data.get("title", "Productivity")

    start_date_str: str = dict_inputs["start_date"].strftime('%Y-%m-%d')
    repo_aggregated_view: bool = dict_inputs.get("repo_aggregated", False)
    repo_topic_aggregated_view: bool = dict_inputs.get(
        "repo_topic_aggregated", False)

    # Initialize the figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))

    if repo_topic_aggregated_view or repo_aggregated_view:
        # Determine if "per contributor" metrics should be plotted
        per_contributor_metric = f"{selected_metric}_per_contributor"
        if per_contributor_metric in df.columns:
            plot_metric = per_contributor_metric
            label_suffix = " per contributor"
        else:
            plot_metric = selected_metric
            label_suffix = ""

        if repo_topic_aggregated_view:
            # When data is aggregated by repo_topic
            for topic, group in df.groupby('repo_topic'):
                ax.plot(group['stats_beginning'], group[plot_metric], marker='', linewidth=2,
                        label=f"{topic}{label_suffix} - Contributors: {group['contributor_count'].iloc[0]}, Repos: {group['repo_count'].iloc[0]}")
            legend_title = 'Repo Topic'
        elif repo_aggregated_view:
            # When data is aggregated by repo
            for (repo, group) in df.groupby('repo'):
                ax.plot(group['stats_beginning'], group[plot_metric], marker='', linewidth=2,
                        label=f"{repo} (Contributors: {group['contributor_count'].iloc[0]})")

            legend_title = 'Repo'
    else:
        # When showing individual contributor data
        for name, group in df.groupby('contributor_name'):
            ax.plot(group['stats_beginning'], group[selected_metric],
                    marker='', linewidth=2, label=name)
        legend_title = 'Contributor Name'

    # Setting titles and labels
    ytitle: str = f'{selected_metric}{label_suffix} since {start_date_str}'

    ax.set_title(title, color='black')  # Ensure title is visible
    ax.set_xlabel('Date', color='black')
    ax.set_ylabel(ytitle)

    # Date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())

    # Adjust tick parameters to ensure they are visible
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')

    # Setting legend with explicit visibility
    ax.legend(title=f'{legend_title}', bbox_to_anchor=(
        1.05, 1), loc='upper left', edgecolor='black')

    # Explicitly set the figure and axes background color after all plotting commands
    # Set the outer background color of the plot
    fig.patch.set_facecolor('white')
    # Set the inner background color of the plot area
    ax.set_facecolor('white')

    # Rotate date labels to avoid overlap
    plt.xticks(rotation=45)  # Adjust the rotation angle if necessary

    # Ensure layout is adjusted to not cut off elements
    plt.tight_layout()

    # Display the plot
    # print(f"Plot of {title}")
    plt.show()
    return


# We store a list of dictionaries, including all our DFs so we can look at the table later if we want.
df_list: list = []
if start_date_date is not None:
    # list entry format is dict = {"title": str_title, "df": pd.DataFrame, "csv": str_filename}
    for metric in gdict_inputs.get("metric_selector"):
        # Create and process the DataFrame for the current metric
        dict_results: dict = {}
        dict_results = create_and_process_dataframe(gdict_inputs, metric)

        df = dict_results.get("df", pd.DataFrame())
        contributors_list: list = dict_results.get("contributors_list", [])

        # Check if DataFrame is not empty
        if not df.empty:
            # Build parts of the message conditionally using list comprehensions
            parts: list = [
                f"({metric}) since {start_date_date.strftime('%Y-%m-%d')}\n in GitHub repos " +
                ", ".join(
                    multiselect_repo_names) if multiselect_repo_names else "",
                f"({metric}) since {start_date_date.strftime('%Y-%m-%d')}\n in GitHub repo topic " +
                ", ".join(
                    multiselect_repo_topics) if multiselect_repo_topics else "",
                f"\nfiltered by contributors " +
                ", ".join(contributors_list) if len(
                    contributors_list) > 0 else ""
            ]

            message = f"Productivity " + " ".join(filter(None, parts))
            dict_results["title"] = message
            # csv_file: str = f"{start_date_date.strftime('%Y-%m-%d')}-{message}.csv"
            df_list.append(dict_results.copy())
            # df.to_csv(csv_file, index=False)
            print(message)
            # Plot the data for the current metric
            plot_data(dict_results, gdict_inputs, metric)
        else:
            print(
                f"No data to plot for {metric} based on the selected filters.")
else:
    print("Skipped running due to lack of inputs")
