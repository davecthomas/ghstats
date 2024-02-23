"""
NOTE: This code WILL NOT work outside of a Hex environment. It is designed to be used within the Hex ecosystem.
You must have a Hex account and be logged in to run this code, along with a valid datasource set up to use the data.
See the dbschema.sql for setting up a Snowflake datasource in Hex here: https://hex.tech/docs/snowflake
"""
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

for dict_df in list_dict_dfs:
    print(dict_df["stats_title"])

    df = dict_df['df']
    title = dict_df['stats_title']

    # Filter out the non-numeric and specific columns you don't want to plot
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    exclude_cols = ['stats_beginning', 'repo_topic',
                    'repo_name']  # Add 'repo_name' if it exists
    plot_cols = [col for col in numeric_cols if col not in exclude_cols]

    # Generate box plots for each numeric column
    for col in plot_cols:
        # Create a figure for the histogram and box plot
        fig, axs = plt.subplots(1, 2, figsize=(17, 6), facecolor='white')

        # Histogram on the left
        axs[0].hist(df[col].dropna(), bins=30,
                    color='skyblue', edgecolor='black')
        axs[0].set_title(f'Histogram of {col} ' + title)
        axs[0].set_xlabel(col)
        axs[0].set_ylabel('Frequency')

        # plt.figure(figsize=(8, 4), facecolor='white')
        box = axs[1].boxplot(df[col].dropna(), patch_artist=True, showmeans=True,
                             meanprops={"marker": "D", "markerfacecolor": "green", "markeredgecolor": "black"})  # Drop NA values for plotting

        # Optional: Set properties for the boxes, whiskers, caps, and the median line
        plt.setp(box['boxes'], facecolor='gray', edgecolor='blue',
                 linewidth=2)  # Make box borders thicker
        plt.setp(box['whiskers'], linestyle='-', color='black',
                 linewidth=2)  # Make whiskers thicker
        plt.setp(box['caps'], color='black', linewidth=2)  # Make caps thicker
        plt.setp(box['medians'], color='orange',
                 linewidth=2.5)  # Make medians thicker

        axs[1].set_title(f"Box plot of {col}" + title)
        axs[1].set_ylabel(col)
        axs[1].set_xticks([1], [col])  # Set x-ticks to just the column name
        axs[1].grid(True)  # Optional: Add grid for better readability
        # Create custom legend handles and labels
        legend_elements = [
            mpatches.Patch(facecolor='gray', edgecolor='blue',
                           linewidth=2, label='Middle Quartiles'),
            mlines.Line2D([], [], color='green', marker='D', markeredgecolor='black',
                          linestyle='None', markersize=10, label='Mean (green diamond)'),
            mlines.Line2D([], [], color='orange', linestyle='-',
                          linewidth=2, label='Median (orange line)'),
            mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                          markersize=5, label='Outliers (black circles)'),
            mlines.Line2D([], [], color='black', linestyle='-',
                          linewidth=2, label='Whiskers (bounds on non-outliers)')
        ]

        # Create the legend
        # axs[1].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        axs[1].legend(handles=legend_elements, loc='upper left',
                      bbox_to_anchor=(1.05, 1), fontsize='small')
        plt.tight_layout()
        plt.show()
        print("\n")
