import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

# Function to get the top 10 box office movies for a given year
def get_top_10_movies(year):
    url = f"https://www.boxofficemojo.com/year/{year}/?grossesOption=totalGrosses"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the table containing the top movies
    table = soup.find('table')
    rows = table.find_all('tr')[1:11]  # Skip the header row and get the top 10

    movies = []
    for row in rows:
        cols = row.find_all('td')
        rank = cols[0].text.strip()
        title = cols[1].text.strip()
        gross = cols[5].text.strip()
        movies.append([year, rank, title, gross])

    return movies

# Scrape data for multiple years and save to a DataFrame
all_movies = []
for year in range(1977, 2024):
    try:
        top_movies = get_top_10_movies(year)
        all_movies.extend(top_movies)
        print(f"Successfully scraped data for {year}")
    except Exception as e:
        print(f"Failed to scrape data for {year}: {e}")

# Convert to DataFrame and save to CSV
df = pd.DataFrame(all_movies, columns=['Year', 'Rank', 'Title', 'Gross'])

# Create an output directory if it doesn't exist
output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)

# Save the CSV in the output folder
output_path = os.path.join(output_folder, 'top_10_box_office_movies_1977_2023.csv')
df.to_csv(output_path, index=False)

print("Data scraping completed and saved to CSV.")