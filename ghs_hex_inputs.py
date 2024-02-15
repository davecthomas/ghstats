import pandas as pd


def print_instructions():
    instructions = """
    Instructions for Plotting Data:
    - The 'multiselect_contributors' input is required.
    - You must provide input for either 'multiselect_repo_names' or 'multiselect_repo_topics'.
    - If inputs are provided for both 'multiselect_repo_names' and 'multiselect_repo_topics', 
      only the values from 'multiselect_repo_names' will be used for plotting.
    - Ensure at least one of 'multiselect_repo_names' or 'multiselect_repo_topics' has values selected
      for the plot to include repository-specific data.
    """
    print(instructions)


def display_plot_values(multiselect_contributors, multiselect_repo_names, multiselect_repo_topics):
    # Check if 'multiselect_contributors' has input
    if not multiselect_contributors:
        print("Error: 'multiselect_contributors' is required.")
        return

    # Determine which repository input to use
    # True if 'multiselect_repo_names' has input
    use_repo_names = bool(multiselect_repo_names)

    # Display the explanation and values to be used
    print("Creating Hex plot output using the following values:")
    print(f"Contributors: {multiselect_contributors}")

    if use_repo_names:
        print(
            f"Repository Names: {multiselect_repo_names} (Selected repository names will be used for plotting.)")
    elif multiselect_repo_topics:
        print(
            f"Repository Topics: {multiselect_repo_topics} (Selected repository topics will be used for plotting, as no repository names were provided.)")
    else:
        print("No repository names or topics were selected. Plotting will proceed without repository-specific data.")


# Call the function to print instructions
print_instructions()

# Call the function to display the values to be used
display_plot_values(multiselect_contributors,
                    multiselect_repo_names, multiselect_repo_topics)
