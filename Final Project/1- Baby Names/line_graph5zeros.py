import os
import pandas as pd
import matplotlib.pyplot as plt
import json


# Load baby name data
def load_name_data(folder_path):
    data = {}
    for year in range(1970, 2024):  # Data available from 1970 to 2023
        file_path = os.path.join(folder_path, f'yob{year}.txt')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, names=['name', 'gender', 'count'])
            data[year] = df
    return data


# Retrieve counts for each name over 7 years following the debut year
def get_name_counts(name, data, debut_year):
    counts = []
    for year in range(debut_year, debut_year + 8):
        if year in data:
            count = data[year][data[year]['name'] == name]['count'].sum()
            counts.append(count)
        else:
            counts.append(0)
    return counts


def plot_jump_decline_over_7_years(all_shows, name_data, file_name):
    plt.figure(figsize=(8, 4))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'magenta', 'yellow', 'brown', 'pink', 'black']  # Removed cyan
    filtered_characters = []  # To store the characters who had 0 counts at debut
    color_index = 0  # To cycle through colors

    for show in all_shows:
        debut_year = show['release_year']
        for character in show['characters']:
            first_name = character.split()[0]
            debut_count = get_name_counts(first_name, name_data, debut_year)[0]
            if debut_count == 0:  # Only include characters with 0 counts at debut
                counts = get_name_counts(first_name, name_data, debut_year)
                filtered_characters.append(first_name)

                # X-axis: 1st year to 7th year after debut
                x_values = range(0, 8)  # 7 years after debut

                # Use a different color for each character
                plt.plot(x_values, counts, label=f"{first_name} ({show['tv_show_name']} {debut_year})",
                         color=colors[color_index % len(colors)])
                color_index += 1

    if filtered_characters:  # Only display the plot if there are characters who had 0 at debut
        plt.title('7-Year Popularity Trends for Names with Fewer Than 5 Babies at TV Show Debut')
        plt.xlabel('Years After Debut')
        plt.ylabel('Number of Babies')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.grid(True)
        plt.tight_layout()

        # Add the file name to the bottom right corner
        plt.text(1, 0.0, f"File: {file_name}", horizontalalignment='right', verticalalignment='bottom',
                 transform=plt.gca().transAxes, fontsize=6, color='black')

        plt.show()
    else:
        print("No characters with 0 at debut.")


# Main execution: Load name data and plot for each character
folder_path = r'C:\Users\ybars\OneDrive - huji.ac.il\HUJI Documents Shana BET\MATAR Bet\Data Mining 47717\babynames text files'  # Replace with your actual folder path
name_data = load_name_data(folder_path)


def load_shows_from_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


# Load the shows data, including the release year
shows_file_path = r'C:\Users\ybars\PycharmProjects\datamining\DM_Names_new_approach\output\tv_shows_new_release.json'
all_shows = load_shows_from_file(shows_file_path)

# Specify the file name you want to display on the graph
file_name = "line_graph5zeros.py"

# Run the plot
plot_jump_decline_over_7_years(all_shows, name_data, file_name)
