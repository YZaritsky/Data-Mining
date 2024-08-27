import pandas as pd
import matplotlib.pyplot as plt
import re

# Load the CSV file
file_path = './output/top_10_box_office_movies_1977_2023_with_villains_origins.csv'
df = pd.read_csv(file_path)

# Define regions and their corresponding countries or locations
regions = {
    'Islamic Countries': ['iran', 'iraq', 'afghanistan', 'syria', 'pakistan', 'saudi', 'egypt', 'turkey', 'libya',
                          'islamic', 'arab'],
    'USA': ['united states', 'usa', 'u.s.', 'america', 'gotham', 'new york', 'california', 'texas', 'illinois',
            'los angeles', 'boston', 'amityville', 'haddonfield', 'metropolis', 'smallville', 'derry', 'maine'],
    'Germany': ['germany', 'german', 'deutsch'],
    'Russian/Ukrainian': ['russia', 'russian', 'ussr', 'soviet', 'ukraine', 'ukrainian', 'stalingrad', 'moscow', 'kyiv',
                          'kiev']
}


# Assign each origin to a region
def assign_region(origin):
    origin_lower = str(origin).lower()
    for region, terms in regions.items():
        if any(re.search(r'\b' + re.escape(term) + r'\b', origin_lower) for term in terms):
            return region
    return 'Other'


df['Region'] = df['Origin'].apply(assign_region)

# Filter out rows where region is 'Other'
df = df[df['Region'] != 'Other']

# Count villains per region over the years
villain_counts = df.groupby(['Year', 'Region']).size().unstack(fill_value=0)

# Part 1 & 2: Line Plot of Villain Trends by Region
plt.figure(figsize=(14, 8))
for region in regions.keys():
    if region in villain_counts.columns:
        plt.plot(villain_counts.index, villain_counts[region], label=region)

plt.xlabel('Year')
plt.ylabel('Number of Villains')
plt.title('Trend of Villains by Region Over Time')
plt.legend()
plt.grid(True)
output_trend_path = "./output_10year_heatmaps/villains_trend_by_region.png"
plt.savefig(output_trend_path, bbox_inches='tight')
plt.close()

print(f"Trend visualization saved to {output_trend_path}")

# Part 3: Geopolitical Event Data (manually defined)
geopolitical_events = {
    '1979': 'Iran Hostage Crisis',
    '1989': 'Fall of Berlin Wall',
    '2001': '9/11 Attacks',
    '2003': 'Iraq War',
    '2014': 'Ukraine Crisis',
    '2022': 'Russia-Ukraine Conflict'
}

# Add a column for geopolitical events
df['Geopolitical_Event'] = df['Year'].apply(lambda x: geopolitical_events.get(str(x), None))

# Filter data for countries with conflicts with the USA
conflict_regions = list(regions.keys())  # Use all defined regions
df_conflict = df[df['Region'].isin(conflict_regions)]

# Count villains in conflict regions by year
conflict_villain_counts = df_conflict.groupby(['Year', 'Region']).size().unstack(fill_value=0)

# Part 4: Visualize Impact of Geopolitical Events
plt.figure(figsize=(14, 8))
for region in conflict_regions:
    if region in conflict_villain_counts.columns:
        plt.plot(conflict_villain_counts.index, conflict_villain_counts[region], label=region)

# Mark geopolitical events on the plot
for event_year, event_name in geopolitical_events.items():
    plt.axvline(x=int(event_year), color='gray', linestyle='--')

    # Adjust the y-position of the text to avoid overlapping with data points
    plt.text(int(event_year), plt.ylim()[1] * 0.85, event_name, rotation=90, verticalalignment='center', fontsize=10)

# Add some extra space around the plot to avoid overlapping
plt.subplots_adjust(right=0.9, top=0.9)

plt.xlabel('Year')
plt.ylabel('Number of Villains')
plt.title('Impact of Geopolitical Events on Villain Origins from Conflict Regions')
plt.legend()
plt.grid(True)
output_conflict_path = "./output_10year_heatmaps/villains_impact_geopolitical_events.png"
plt.savefig(output_conflict_path, bbox_inches='tight')
plt.close()

print(f"Geopolitical event impact visualization saved to {output_conflict_path}")