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


def create_and_process_dataframe(dict_inputs: dict, selected_metric: str) -> pd.DataFrame:

    repo_aggregated_view: bool = dict_inputs.get("repo_aggregated", False)
    repo_topic_aggregated_view: bool = dict_inputs.get(
        "repo_topic_aggregated", False)
    username_filter: list = dict_inputs["multiselect_contributors"]
    repo_filter: list = dict_inputs.get("multiselect_repo_names", [])
    topic_filter: list = dict_inputs.get("multiselect_repo_topics", [])
    start_date_date: datetime.date = dict_inputs.get(
        "start_date", get_first_day_of_month_one_year_ago())
    # for modifying SQL
    start_date_str = start_date_date.strftime('%Y-%m-%d')

    query_template = f"""
    SELECT cs."repo", cs."stats_beginning", cs."contributor_name", cs."{selected_metric}", rt."repo_topic"
    FROM "contributor_stats" cs
    LEFT JOIN "repo_topics" rt ON cs."repo" = rt."repo_name"
    WHERE cs."stats_beginning" >= '{start_date_str}'
    ORDER BY cs."stats_beginning" ASC, cs."repo";
    """

    # Execute the query and load results into a pandas DataFrame
    df: pd.DataFrame = session.sql(query_template).to_pandas()
    # Ensure 'stats_beginning' is a datetime format
    df['stats_beginning'] = pd.to_datetime(df['stats_beginning'])

    # Repo topic aggregate: also count the number of contributors and repos
    if repo_topic_aggregated_view:
        df = df.groupby(['repo_topic', 'stats_beginning']).agg(
            avg_selected_metric=(selected_metric, 'mean'),
            contributor_count=('contributor_name', 'nunique'),
            repo_count=('repo', 'nunique')).reset_index()

    elif repo_aggregated_view:
        # Group by 'repo' and 'stats_beginning', then aggregate
        df = df.groupby(['repo', 'stats_beginning']).agg(
            avg_selected_metric=(selected_metric, 'mean'),
            contributor_count=('contributor_name', 'nunique')).reset_index()
        df = df.drop_duplicates(
            subset=['repo', 'stats_beginning'], keep='first')

    # Remove duplicates if no aggregation is applied
    if not repo_aggregated_view and not repo_topic_aggregated_view:
        df = df.drop_duplicates(
            subset=['contributor_name', 'repo', 'stats_beginning'], keep='first')

    # Filter DataFrame
    if 'contributor_name' in df.columns and username_filter and (repo_aggregated_view is False and repo_topic_aggregated_view is False):
        df = df[df['contributor_name'].isin(username_filter)]
    if 'repo' in df.columns and repo_filter:
        df = df[df['repo'].isin(repo_filter)]
    if 'repo_topic' in df.columns and topic_filter:
        df = df[df['repo_topic'].isin(topic_filter)]

    return df


def plot_data(df, dict_inputs: dict, selected_metric: str):
    start_date_str: str = dict_inputs["start_date"].strftime('%Y-%m-%d')
    repo_aggregated_view: bool = dict_inputs.get("repo_aggregated", False)
    repo_topic_aggregated_view: bool = dict_inputs.get(
        "repo_topic_aggregated", False)

    # Initialize the figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))

    if repo_topic_aggregated_view:
        # When data is aggregated by repo_topic
        for topic, group in df.groupby('repo_topic'):
            ax.plot(group['stats_beginning'], group['avg_selected_metric'], marker='', linewidth=2,
                    label=f"{topic} - Contributors: {group['contributor_count'].iloc[0]}, Repos: {group['repo_count'].iloc[0]}")
        legend_title = 'Repo Topic'
    elif repo_aggregated_view:
        # When data is aggregated by repo
        for (repo, group) in df.groupby('repo'):
            ax.plot(group['stats_beginning'], group['avg_selected_metric'], marker='',
                    linewidth=2, label=f"{repo} (Contributors: {group['contributor_count'].iloc[0]})")

        legend_title = 'Repo'
    else:
        # When showing individual contributor data
        for name, group in df.groupby('contributor_name'):
            ax.plot(group['stats_beginning'], group[selected_metric],
                    marker='', linewidth=2, label=name)
        legend_title = 'Contributor Name'

    # # Plotting data
    # for name, group in df.groupby('contributor_name'):
    #     ax.plot(group['stats_beginning'], group[f'{selected_metric}'], marker='', linewidth=2, label=name)

    # Setting titles and labels
    title: str = f'{selected_metric} since {start_date_str}'

    ax.set_title(title, color='black')  # Ensure title is visible
    ax.set_xlabel('Date', color='black')
    ax.set_ylabel(title)

    # Date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())

    # Adjust tick parameters to ensure they are visible
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')

    # Setting legend with explicit visibility
    ax.legend(title='Contributor Name', bbox_to_anchor=(
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
    print(f"Plot of {title}")
    plt.show()


if start_date_date is not None:
    for metric in metric_selector:
        # Create and process the DataFrame for the current metric
        df = create_and_process_dataframe(gdict_inputs, metric)

        # Check if DataFrame is not empty
        if not df.empty:
            # Build parts of the message conditionally using list comprehensions
            parts: list = [
                "in GitHub repos " +
                ", ".join(
                    multiselect_repo_names) if multiselect_repo_names else "",
                "in GitHub repo topic " +
                ", ".join(
                    multiselect_repo_topics) if multiselect_repo_topics else ""
            ]

            # Filter out any empty strings and join the remaining parts with a space
            message = f"Productivity " + " ".join(filter(None, parts))
            print(message)
            # Plot the data for the current metric
            plot_data(df, gdict_inputs, metric)
        else:
            print(
                f"No data to plot for {metric} based on the selected filters.")
else:
    print("Skipped running due to lack of inputs")
