import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans

SIZE = 100

# Load the dataset
file_path = './data/movies.csv'
movies_df = pd.read_csv(file_path)

# Filter necessary columns
df = movies_df[['director', 'star']]

# Create a graph
G = nx.Graph()

# Add edges to the graph with a weight representing the number of collaborations
edge_weights = df.groupby(['director', 'star']).size().reset_index(name='weight')

# Select the top SIZE collaborations
top_edges = edge_weights.nlargest(SIZE, 'weight')

# Add edges to the graph
for index, row in top_edges.iterrows():
    G.add_edge(row['director'], row['star'], weight=row['weight'])

# Apply KMeans clustering to the directors
directors = df['director'].unique()
pivot_df = df.pivot_table(index='director', columns='star', aggfunc='size', fill_value=0)
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(pivot_df)
director_clusters = dict(zip(directors, clusters))

# Define pastel colors
pastel_blue = '#AEC6CF'
pastel_red = '#FFB6C1'

# Color nodes based on whether they are directors or stars
node_colors = []
node_sizes = []
for node in G.nodes():
    if node in director_clusters:
        node_colors.append(pastel_blue)  # Pastel color for directors
        node_sizes.append(1000)  # Larger size for directors
    else:
        node_colors.append(pastel_red)  # Pastel color for stars
        node_sizes.append(300)  # Smaller size for stars

# Get edge widths based on the weight
edge_widths = [G[u][v]['weight'] for u, v in G.edges()]

# Draw the graph
plt.figure(figsize=(15, 15))
pos = nx.spring_layout(G, k=0.15, iterations=20)
nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=node_sizes, font_size=10, font_color='black', edge_color=edge_widths, edge_cmap=plt.cm.Blues, width=2)

# Create a legend for the directors and stars
handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=pastel_blue, markersize=10, label='Directors'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=pastel_red, markersize=10, label='Stars')
]
plt.legend(handles=handles, title='Node Types')

# Set the title dynamically
plt.title(f'Network of Top {SIZE} Director-Star Collaborations with Clusters')
plt.show()