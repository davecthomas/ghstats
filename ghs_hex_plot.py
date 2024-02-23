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
import itertools

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

    # print(query_template)
    # Execute the query and load results into a pandas DataFrame
    df = session.sql(query_template).to_pandas()
    df['stats_beginning'] = pd.to_datetime(df['stats_beginning'])

    # Capture the contributor names before aggregation if username_filter is not empty
    if len(username_filter) > 0:
        # contributors_list = df[df['contributor_name'].isin(username_filter)]['contributor_name'].unique()
        contributors_list = df['contributor_name'].unique().tolist()
    else:
        contributors_list = []

    dict_results["contributors_list"] = contributors_list

    # Aggregation and calculation of repo_count for repo_topic_aggregated_view
    # repo_topic is selected but not aggregated: include all contributors to a repo_topic
    if topic_filter and not repo_topic_aggregated_view:
        agg_columns = {
            selected_metric: 'mean',
            'repo': 'nunique',  # Count unique repos
            'changed_lines_per_day': 'mean',
            'commits_per_day': 'mean',
            'review_comments_per_day': 'mean',
            'prs_per_day': 'mean',
            'avg_pr_duration': 'mean',
            'avg_code_movement_per_pr': 'mean'
        }
        df_agg: pd.DataFrame = df.groupby(
            ['repo_topic', 'stats_beginning', 'contributor_name']).agg(agg_columns).reset_index()
        """
        ['repo_topic', 'stats_beginning', 'prs_per_day', 'repo',
       'changed_lines_per_day', 'commits_per_day', 'review_comments_per_day',
       'avg_pr_duration', 'avg_code_movement_per_pr'],
       """
        df_agg = df_agg.rename(columns={'repo': 'repo_count'})

        # Calculate per contributor metrics
        contributor_count: int = len(
            df_agg['contributor_name'].unique().tolist())

        for metric in ['commits_per_day', 'prs_per_day', 'changed_lines_per_day', 'review_comments_per_day']:
            per_contributor_metric = f"{metric}_per_contributor"
            df_agg[per_contributor_metric] = df_agg[metric] / contributor_count

        df = df_agg

    # Basic repo topic aggregation when both a topic is selected and aggregation requested
    elif topic_filter and repo_topic_aggregated_view:
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
        df_agg = df_agg.rename(
            columns={'contributor_name': 'contributor_count', 'repo': 'repo_count'})

        # Calculate per contributor metrics
        for metric in ['commits_per_day', 'prs_per_day', 'changed_lines_per_day', 'review_comments_per_day']:
            per_contributor_metric = f"{metric}_per_contributor"
            df_agg[per_contributor_metric] = df_agg[metric] / \
                df_agg['contributor_count']

        df = df_agg

    # Repo aggregation selected
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
    if not repo_aggregated_view and not repo_topic_aggregated_view and len(topic_filter) == 0:
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
    repo_topic_list: list = dict_inputs.get("multiselect_repo_topics", [])
    label_suffix: str = ""

    # Initialize the figure and axes
    fig, ax = plt.subplots(figsize=(15, 6))

    # Define separate cycles for line styles, markers, and colors
    line_styles = itertools.cycle(['-', '--', '-.', ':'])
    markers = itertools.cycle(['o', '^', 's', '*', '+', 'x', 'D', '|', '_'])
    # Using tab20 for more distinct colors
    colors = itertools.cycle(plt.cm.tab20(np.linspace(0, 1, 20)))
    widths = itertools.cycle([2])

    # internal function to get the next style
    def get_next_style():
        return next(line_styles), next(markers), next(colors), next(widths)

    plot_lines: list = []
    line_labels: list = []

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
                style, marker, color, width = get_next_style()
                line, = ax.plot(group['stats_beginning'], group[plot_metric], linestyle=style, marker=marker, color=color, linewidth=width,
                                label=f"{topic}{label_suffix} - Contributors: {group['contributor_count'].iloc[0]}, Repos: {group['repo_count'].iloc[0]}")
                plot_lines.append(line)
                line_labels.append(topic)
            legend_title = 'Repo Topic'
        elif repo_aggregated_view:
            # When data is aggregated by repo
            for (repo, group) in df.groupby('repo'):
                style, marker, color, width = get_next_style()
                line, = ax.plot(group['stats_beginning'], group[plot_metric], linestyle=style, marker=marker,
                                color=color, linewidth=width, label=f"{repo} (Contributors: {group['contributor_count'].iloc[0]})")
                plot_lines.append(line)
                line_labels.append(repo)
            legend_title = 'Repo'
    # elif len(repo_topic_list) > 0 and repo_topic_aggregated_view is False:
    #     # if 'contributor_name' in df:
    #     print("TBD\n\n")
    #     pass
    else:
        # grouped_data = df.groupby('contributor_name')

        # # Calculate the number of subplots needed
        # num_plots = len(grouped_data) // 10 + (1 if len(grouped_data) % 10 > 0 else 0)

        # for i in range(num_plots):
        #     start_index = i * 10
        #     end_index = min((i + 1) * 10, len(grouped_data))
        #     for name, group in grouped_data[start_index:end_index]:
        #         style, marker, color, width = get_next_style()
        #         line, = ax.plot(group['stats_beginning'], group[selected_metric], linestyle=style, marker=marker, color=color, linewidth=width, label=f"{name}")
        #         plot_lines.append(line)
        #         line_labels.append(name)
        #     plt.show()
        # When showing individual contributor data
        for name, group in df.groupby('contributor_name'):
            style, marker, color, width = get_next_style()
            line, = ax.plot(group['stats_beginning'], group[selected_metric], linestyle=style,
                            marker=marker, color=color, linewidth=width, label=f"{name}")
            plot_lines.append(line)
            line_labels.append(name)
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
    # print(line_labels)
    plt.show()
    return


# We store a list of dictionaries, including all our DFs so we can look at the table later if we want.
list_dict_dfs: list = []
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
            MAX_NAMES_TO_LIST: int = 8
            parts: list = [
                f"since {start_date_date.strftime('%Y-%m-%d')} in GitHub repos " + ", ".join(multiselect_repo_names) if (
                    len(multiselect_repo_names) > 0 and len(multiselect_repo_names) < MAX_NAMES_TO_LIST) else "",
                f"since {start_date_date.strftime('%Y-%m-%d')} in GitHub repos " + ", ".join(
                    multiselect_repo_names[:MAX_NAMES_TO_LIST]) + "... (etc)" if len(multiselect_repo_names) >= MAX_NAMES_TO_LIST else "",
                f"since {start_date_date.strftime('%Y-%m-%d')} in GitHub repo topic " + ", ".join(
                    multiselect_repo_topics) if multiselect_repo_topics else "",
                f"since {start_date_date.strftime('%Y-%m-%d')} in GitHub repo topic " + ", ".join(
                    multiselect_repo_topics[:MAX_NAMES_TO_LIST]) + "... (etc)" if len(multiselect_repo_topics) >= MAX_NAMES_TO_LIST else "",
                f"\nfiltered by contributors " + ", ".join(contributors_list) if (len(
                    contributors_list) > 0 and len(contributors_list) < MAX_NAMES_TO_LIST) else "",
                f"\nfiltered by contributors " +
                ", ".join(contributors_list[:MAX_NAMES_TO_LIST]) + "... (etc)" if len(
                    contributors_list) >= MAX_NAMES_TO_LIST else ""
            ]

            message = f"Productivity of {metric} " + \
                " ".join(filter(None, parts))
            dict_results["title"] = message
            stats_message: str = f" ".join(filter(None, parts))
            dict_results["stats_title"] = stats_message
            # csv_file: str = f"{start_date_date.strftime('%Y-%m-%d')}-{message}.csv"
            list_dict_dfs.append(dict_results.copy())
            # df.to_csv(csv_file, index=False)

            print(message)
            # Plot the data for the current metric
            plot_data(dict_results, gdict_inputs, metric)
        else:
            print(
                f"No data to plot for {metric} based on the selected filters.")
else:
    print("Skipped running due to lack of inputs")
