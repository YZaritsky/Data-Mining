import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Step 1: Load the CSV data from the file
df = pd.read_csv('output/top_10_box_office_movies_1977_2023_with_villains_origins.csv')

# Step 2: Define regions and exclude USA villains
regions = {
    'Islamic Countries': ['Iran', 'Iraq', 'Afghanistan', 'Syria', 'Pakistan', 'Saudi Arabia', 'Egypt', 'Turkey',
                          'Libya', 'Islamic'],
    'Communist Asia': ['China', 'North Korea'],
    'Russian/Ukrainian': ['Russia', 'USSR', 'Soviet Union', 'Ukraine', 'Stalingrad', 'Moscow', 'Kyiv']
}

# Define geopolitical conflicts data
geopolitical_conflicts = {
    'Islamic Countries': [(1979, 1981), (1991, 1993), (2001, 2001), (2002, 2015)],  # Iran Hostage Crisis, gulf war, 9/11, Iraq and Afganistan Wars
    'Communist Asia': [(2006, 2018), (2020, 2024)],  # North Korea Nuclear threat, China marked as greatest threat to the USA
    'Russian/Ukrainian': [(1980, 1985), (2014, 2016), (2022, 2024)]  # Cold war tention, Ukraine Crisis, Russia-Ukraine Conflict
}


# Function to assign region and determine conflict status
def assign_region_and_conflict(row):
    origin = str(row['Origin']) if not pd.isna(row['Origin']) else ''
    year = row['Year']

    # Determine the region
    for region, countries in regions.items():
        if any(country in origin for country in countries):
            # Check if the year falls within any conflict period
            for start, end in geopolitical_conflicts.get(region, []):
                if start <= year <= (end if end != 'present' else 9999):  # Handle 'present' as ongoing
                    return region, 1  # In conflict
            return region, 0  # Not in conflict

    return 'Other', 0  # Not in conflict and not from a defined region


# Apply the function to assign region and conflict status
df['Region'], df['In_Conflict'] = zip(*df.apply(assign_region_and_conflict, axis=1))

# Filter out 'Other' regions and villains from the USA
df = df[df['Region'] != 'Other']

# Step 3: Splitting Data for Training and Testing
X = df[['Year', 'In_Conflict']]
y = df['In_Conflict']

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 4: Training the Model with Random Forest
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Step 5: Making Predictions for 2025
new_data_2025 = pd.DataFrame({
    'Year': [2025],
    'In_Conflict': [1]  # Assuming ongoing conflicts
})

# Predicting for 2025
prediction_2025 = model.predict_proba(new_data_2025)[0][1] * 100  # Probability for conflict region

print(
    f"Predicted probability that the villain in 2025 will be from a region in conflict with the USA: {prediction_2025:.2f}%")

# Step 6: Evaluating the Model on Test Data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Update this part to avoid the warning
labels = [0, 1]  # These are the possible labels for 'In_Conflict'
conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)
class_report = classification_report(y_test, y_pred, labels=labels, zero_division=0)

print(f"Test Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Step 7: Visualizing Yearly Probabilities
yearly_probabilities = df.groupby('Year')['In_Conflict'].mean() * 100  # Convert to percentage

plt.figure(figsize=(12, 6))
plt.plot(yearly_probabilities.index, yearly_probabilities.values, marker='o', label='Historical Data')
plt.scatter(2025, prediction_2025, color='red', label=f'2025 Prediction: {prediction_2025:.2f}%',
            s=100)  # Add prediction for 2025

plt.xlabel('Year')
plt.ylabel('Probability of Villain from Conflict Region (%)')
plt.title('Probability of Villain from Region in Conflict with USA (1977-2024)')
plt.grid(True)
plt.ylim(0, 100)  # Set the Y-axis from 0 to 100%
plt.legend()

# Save the plot to a file
plt.savefig('output_prediction.png', bbox_inches='tight')

plt.show()
