import pandas as pd
import re
import matplotlib.pyplot as plt

# Step 1: Load the dataset
file_path = './data/imdb-movies-dataset.csv'
df = pd.read_csv(file_path)

# Step 2: Create a new DataFrame with 'Director' and 'Cast' columns where 'Cast' is split into individual actors
df_exploded = df[['Director', 'Cast']].copy()
df_exploded['Cast'] = df_exploded['Cast'].str.split(', ')
df_exploded = df_exploded.explode('Cast')

# Step 3: Count the number of times each actor has worked with each director
director_actor_count = df_exploded.groupby(['Cast', 'Director']).size().reset_index(name='Count')

# Step 4: Create the two lists
# Actors who have worked with the same director at least 3 times (colored red)
actors_same_director = director_actor_count[director_actor_count['Count'] >= 3]['Cast'].unique()

# Actors who have worked with at least 3 different directors (colored blue)
actors_different_directors = director_actor_count.groupby('Cast')['Director'].nunique()
actors_different_directors = actors_different_directors[actors_different_directors >= 3].index.tolist()

# Step 5: Load the text file containing the IMDB list and rankings
file_path_txt = 'data/IMDB_top_100_actors.txt'
with open(file_path_txt, 'r', encoding='utf-8') as file:
    imdb_actor_list = file.readlines()

# Step 6: Extract actor names and their rankings from the text file
actor_pattern = re.compile(r'^[A-Z][a-z]+(?:\s[A-Z][a-z]+)*$')
actor_ranking_pattern = re.compile(r'^(\d+)\.\s([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)')

# Dictionary to hold the actor names and their corresponding rankings
actor_rankings = {}

for line in imdb_actor_list:
    match = actor_ranking_pattern.match(line.strip())
    if match:
        rank = int(match.group(1))
        actor_name = match.group(2)
        actor_rankings[actor_name] = rank

# Cross-reference the extracted actors with the earlier lists
actors_in_common_red = set(actor_rankings.keys()) & set(actors_same_director)
actors_in_common_blue = set(actor_rankings.keys()) & set(actors_different_directors)

# Step 7: Save the two lists as CSV files
actors_same_director_df = pd.DataFrame({'Actor': list(actors_in_common_red), 'Category': 'Red'})
actors_different_directors_df = pd.DataFrame({'Actor': list(actors_in_common_blue), 'Category': 'Blue'})
actors_same_director_df.to_csv('./output/shellys_request/actors_same_director_common.csv', index=False)
actors_different_directors_df.to_csv('./output/shellys_request/actors_different_directors_common.csv', index=False)

# Step 8: Prepare the data for the plot using the rankings from the "IMDB_top_100_actors.txt" file
sorted_actors = sorted(actor_rankings.items(), key=lambda x: x[1])
x_labels, y_values = zip(*sorted_actors)

# Filter to include only actors that are in the common lists (Red or Blue)
x_labels = [actor for actor in x_labels if actor in actors_in_common_red.union(actors_in_common_blue)]
y_values = [actor_rankings[actor] for actor in x_labels]

# Reverse the order of x_labels and y_values to make names ordered from right to left
x_labels = x_labels[::-1]
y_values = y_values[::-1]

# Determine the colors based on the list membership
colors = ['red' if actor in actors_in_common_red else 'blue' for actor in x_labels]

# Step 9: Plotting the graph
plt.figure(figsize=(10, 8))
plt.scatter(x_labels, y_values, c=colors)

# Adding labels and title
plt.xticks(rotation=90)
plt.xlabel('Actors')
plt.ylabel('Rankings (1-100)')
plt.title('Actor Rankings Based on IMDB List')
plt.grid(True)

# Reverse the Y-axis so that 1 is at the top and 100 is at the bottom
plt.gca().invert_yaxis()

# Adding the legend
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Same director 3+ times'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='3+ different directors')
])

# Show the plot
plt.tight_layout()
plt.savefig('./output/shellys_request/actor_rankings_plot.png', dpi=300, bbox_inches='tight')
plt.show()
