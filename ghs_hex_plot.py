import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hextoolkit
import matplotlib.dates as mdates


# Assuming this variable comes from a Hex input widget
# Placeholder for dynamic input from Hex
trailing_months: int = trailing_average_months

# Convert months to days for the rolling window
trailing_days: int = trailing_months * 30  # Approximation

# Establish connection using HexToolkit
hex_snowflake_conn = hextoolkit.get_data_connection('Contributor Stats')
session = hex_snowflake_conn.get_snowpark_session()


def create_and_process_dataframe(selected_metric: str, trailing_months: int, username_filter: list, repo_filter: list, topic_filter: list) -> pd.DataFrame:
    # Example query template. Customize it based on your actual database schema.
    query_template = f"""
        SELECT 
            cs.*,
            rt."repo_topic"
        FROM 
            "contributor_stats" cs
        LEFT JOIN 
            "repo_topics" rt ON cs."repo" = rt."repo_name"
        ORDER BY 
            cs."stats_beginning" ASC, cs."contributor_name";
        """

    # Execute the query and load results into a pandas DataFrame
    df: pd.DataFrame = session.sql(query_template).to_pandas()

    # Ensure 'stats_beginning' is a datetime format
    df['stats_beginning'] = pd.to_datetime(df['stats_beginning'])

    # Calculate rolling average
    rolling_days: int = trailing_months * 30  # Convert months to days
    df[f'{selected_metric}_rolling_avg'] = df.groupby('contributor_name')[selected_metric].transform(
        lambda x: x.rolling(window=rolling_days, min_periods=1).mean())

    # Filter DataFrame
    if username_filter:
        df = df[df['contributor_name'].isin(username_filter)]
    if repo_filter:
        df = df[df['repo'].isin(repo_filter)]
    if topic_filter:
        # Adjust if your schema is different
        df = df[df['repo_topic'].isin(topic_filter)]

    return df


def plot_data(df, metric: str):
    # Initialize the figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plotting data
    for name, group in df.groupby('contributor_name'):
        ax.plot(group['stats_beginning'],
                group[f'{metric}_rolling_avg'], marker='', linewidth=2, label=name)

    # Setting titles and labels
    ax.set_title(metric, color='black')  # Ensure title is visible
    ax.set_xlabel('Date', color='black')
    ax.set_ylabel(
        f'{metric} {trailing_months}-Month Trailing Median Score', color='black')

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
    plt.show()


for metric in metric_selector:
    # Create and process the DataFrame for the current metric
    df = create_and_process_dataframe(
        metric, trailing_average_months, multiselect_contributors, multiselect_repo_names, multiselect_repo_topics)

    # Check if DataFrame is not empty
    if not df.empty:
        # Plot the data for the current metric
        plot_data(df, metric)
    else:
        print(f"No data to plot for {metric} based on the selected filters.")
