import json
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from shapely.geometry import Point
import time
import pandas as pd

# Load the JSON data
with open('./output/villains_data.json', 'r') as f:
    data = json.load(f)

# Extract place of birth information and filter out empty entries
places_of_birth = [
    {
        'name': villain['name'],
        'place_of_birth': villain['place_of_birth'],
        'universe': villain['universe']
    }
    for villain in data if villain.get('place_of_birth', '')
]

# Geocode places of birth to get coordinates
geolocator = Nominatim(user_agent="villain_locator", timeout=10)


def geocode(place, retries=3):
    try:
        location = geolocator.geocode(place)
        if location:
            return Point(location.longitude, location.latitude)
    except (GeocoderTimedOut, GeocoderUnavailable) as e:
        if retries > 0:
            time.sleep(2)  # Delay before retrying
            return geocode(place, retries - 1)
        else:
            print(f"Geocoding failed for {place} after several retries: {e}")
    return None


geo_data = []
for place in places_of_birth:
    point = geocode(place['place_of_birth'])
    if point:
        geo_data.append({
            'name': place['name'],
            'universe': place['universe'],
            'geometry': point
        })

# Create a GeoDataFrame
gdf = gpd.GeoDataFrame(geo_data)

# Create a DataFrame for heat map data
heatmap_data = pd.DataFrame([(point.y, point.x) for point in gdf.geometry], columns=['lat', 'lon'])

# Load the low resolution world map from a local path
world = gpd.read_file("./needed_files/ne_110m_admin_0_countries.shp")

# Drop "Antarctica" from the dataframe
world = world[world['CONTINENT'] != 'Antarctica']

# Initialize a final empty figure with higher DPI for higher resolution
fig, ax = plt.subplots(figsize=(20, 10), dpi=300)

# Start by plotting a map of the world
world.boundary.plot(ax=ax, color="black", linewidth=0.5)

# Plot the heatmap
sns.kdeplot(
    data=heatmap_data,
    x='lon',
    y='lat',
    fill=True,
    cmap='Reds',
    alpha=0.6,
    ax=ax,
    levels=100,
    thresh=0.01
)

# Turn off axis ticks
ax.set_xticks([])
ax.set_yticks([])

# Set the plot title
plt.title("Villains' Places of Birth Heatmap")

# Save the plot
output_path = "./output/villains_places_of_birth_heatmap.png"
plt.savefig(output_path, bbox_inches='tight')
plt.show()

print(f"Heatmap saved to {output_path}")
