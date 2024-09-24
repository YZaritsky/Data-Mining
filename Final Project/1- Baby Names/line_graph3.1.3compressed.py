import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

# Load baby name data from files
def load_name_data(folder_path):
    data = {}
    for year in range(1900, 2024):
        file_path = os.path.join(folder_path, f'yob{year}.txt')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, names=['name', 'gender', 'count'])
            data[year] = df
    return data

# Retrieve name counts for each year
def get_name_counts(name, data, start_year):
    counts = []
    for year in range(start_year, 2024):
        if year in data:
            count = data[year][data[year]['name'] == name]['count'].sum()
            counts.append(count)
        else:
            counts.append(0)
    return counts


# Plot individual show popularity in a grid of 2x5 on A4-sized paper
def plot_individual_graphs(all_shows, name_data, selected_shows=None, show_dotted=False):
    fig, axs = plt.subplots(5, 2, figsize=(8.3, 11.7))  # A4 size in vertical orientation (8.3x11.7 inches)
    # fig, axs = plt.subplots(3,4 ,figsize=(17,14))
    # fig, axs = plt.subplots(2, 5, figsize=(15, 6))  # A4 size in horizontal orientation looks bad though

    axs = axs.flatten()  # Flatten to easily loop over subplots
    # colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan']  # Fixed set of 6 colors
    colors = ['red', 'darkorange', 'green', 'royalblue', 'navy', 'purple']  # Fixed set of 6 colors

    for i, show in enumerate(all_shows):
        if selected_shows and show['tv_show_name'] not in selected_shows:
            continue

        if show['tv_show_name'] == "Friends":
            start_year = 1970
        else:
            start_year = 1990

        ax = axs[i]  # Get the current subplot

        # Plot individual characters with names at the end of the lines
        for j, character in enumerate(show['characters']):
            first_name = character.split()[0]
            counts = get_name_counts(first_name, name_data, start_year)
            counts = [max(c, 1) for c in counts]  # Avoid log scale errors

            release_year = show['release_year']
            pre_release_counts = counts[:release_year - start_year + 1]
            post_release_counts = counts[release_year - start_year:]

            if show_dotted:
                ax.plot(range(start_year, release_year + 1), pre_release_counts, linestyle=':', color=colors[j % len(colors)])
                ax.plot(range(release_year, 2024), post_release_counts, linestyle='-', color=colors[j % len(colors)])
                ax.scatter(release_year, counts[release_year - start_year], color=colors[j % len(colors)], marker='o', s=60, zorder=5)
            else:
                ax.plot(range(start_year, 2024), counts, color=colors[j % len(colors)])

            # Add the character's name at the end of their line
            ax.annotate(first_name, xy=(2023, counts[-1]), xytext=(5, 0), textcoords='offset points',
                        color=colors[j % len(colors)], fontsize=6.5, fontweight='bold')

        # Set the title and y-axis to logarithmic scale
        ax.set_title(f"{show['tv_show_name']} ({release_year})", fontsize=10)
        ax.set_yscale('log')
        ax.grid(True)

    # Remove any unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.suptitle("TV Show Character Name Popularity (1990-2023)", fontsize=14)

    fig.tight_layout()
    plt.savefig("individual_tv_shows_A4_grid.pdf")  # Save the plot as a PDF for A4 printing
    plt.show()

# Plot the overall average graph with the legend
def plot_average_popularity(all_shows, name_data, file_name, selected_shows=None, show_dotted=True):
    plt.figure(figsize=(8, 4))  # Standard size for a separate average plot
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'palevioletred']

    all_show_averages = []
    handles = []  # To store legend handles
    labels = []   # To store legend labels

    for i, show in enumerate(all_shows):
        if selected_shows and show['tv_show_name'] not in selected_shows:
            continue

        start_year = 1990
        show_counts = []

        for character in show['characters']:
            first_name = character.split()[0]
            counts = get_name_counts(first_name, name_data, start_year)
            show_counts.append(counts)

        show_average = np.mean(show_counts, axis=0) if show_counts else np.zeros(len(range(start_year, 2024)))
        all_show_averages.append(show_average)

        release_year = show['release_year']
        pre_release_average = show_average[:release_year - start_year + 1]
        post_release_average = show_average[release_year - start_year:]

        if show_dotted:
            plt.plot(range(start_year, release_year + 1), pre_release_average, linestyle=':', color=colors[i])
            plt.plot(range(release_year, 2024), post_release_average, linestyle='-', color=colors[i])
            plt.scatter(release_year, show_average[release_year - start_year], color=colors[i], marker='o', s=100, zorder=5)
        else:
            plt.plot(range(start_year, 2024), show_average, label=f"{show['tv_show_name']}: {show['release_year']}", color=colors[i])

        # Add one label for each show (only once)
        handles.append(plt.Line2D([0], [0], color=colors[i], linestyle='-'))
        labels.append(f"{show['tv_show_name']} ({show['release_year']})")

    # Calculate the overall average of all shows
    overall_average = np.mean(all_show_averages, axis=0)
    plt.plot(range(1990, 2024), overall_average, label="Overall Average (All 50 Names)", color='black', linestyle='-', linewidth=2.3, alpha = 0.7)

    # Add one handle for the overall average
    handles.append(plt.Line2D([0], [0], color='black', linestyle='-', linewidth=2.5))
    labels.append("Overall Average (All 50 Names)")

    # Set the custom legend
    plt.legend(handles=handles, labels=labels, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title("Average Character Name Popularity Per Show (1990-2023)")
    plt.xlabel('Year')
    plt.ylabel('Number of Babies')
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()

    # Add the file name to the bottom right corner
    plt.text(1, 0.02, f"File: {file_name}", horizontalalignment='right', verticalalignment='bottom',
             transform=plt.gca().transAxes, fontsize=6, color='black')

    plt.show()

# Main plot function to choose between individual, average, specific names, or specific shows
def main_plot(all_shows, name_data, file_name, plot_averages=False, plot_individuals=False, selected_shows=None,
              selected_names=None, show_dotted=False):
    # Plot individual graphs for each show or specific shows
    if plot_individuals:
        plot_individual_graphs(all_shows, name_data, selected_shows, show_dotted)

    # If plot_averages is True, plot the average popularity graph
    if plot_averages:
        plot_average_popularity(all_shows, name_data, file_name, selected_shows)

# Main execution: Load name data and plot for each show
folder_path = r'C:\Users\ybars\OneDrive - huji.ac.il\HUJI Documents Shana BET\MATAR Bet\Data Mining 47717\babynames text files'  # Replace with your actual folder path
name_data = load_name_data(folder_path)

def load_shows_from_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


# Load the shows data from the provided file, which includes release years
shows_file_path = r'C:\Users\ybars\PycharmProjects\datamining\DM_Names_new_approach\output\tv_shows_new_release.json'
all_shows = load_shows_from_file(shows_file_path)

# Specify the file name you want to display on the graph
file_name = "line_graph_A4_vertical.pdf"

# Run the plot with the option to plot specific shows or names

# Run the plot with the option to plot specific shows or names
main_plot(all_shows, name_data, file_name, plot_averages=False, plot_individuals=True, show_dotted=True)  # For individual plots
main_plot(all_shows, name_data, file_name, plot_averages=True, plot_individuals=False)  # For average graph
