import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load the CSV data from the file
df = pd.read_csv('output/top_10_box_office_movies_1977_2023_with_villains_origins.csv')

# Step 2: Define regions
regions = {
    'Islamic Countries': ['Iran', 'Iraq', 'Afghanistan', 'Syria', 'Pakistan', 'Saudi Arabia', 'Egypt', 'Turkey', 'Libya', 'Islamic'],
    'USA': ['United States', 'USA', 'Gotham City', 'New York', 'California', 'Texas', 'Illinois', 'Los Angeles', 'Long Island', 'Boston', 'Amityville', 'Haddonfield', 'Metropolis', 'Amity Island', 'Smallville', 'Derry, Maine', 'Stark Industries', 'Canada', 'Toronto', 'Montreal'],
    'Germany': ['Germany', 'West Germany', 'East Germany'],
    'Russian/Ukrainian': ['Russia', 'USSR', 'Soviet Union', 'Ukraine', 'Stalingrad', 'Moscow', 'Kyiv']
}

# Step 3: Data Processing
df['place_of_birth'] = df['Origin'].replace('', 'Unknown')
df['Region'] = df['Origin'].apply(lambda x: next((region for region, places in regions.items() if x in places), 'Other'))
df['Region_Code'] = df['Region'].astype('category').cat.codes

# Step 4: Splitting Data for Training and Testing
X = df[['Region_Code', 'Year']]
y = df['Region_Code']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Training the Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Step 6: Evaluating the Model using accuracy, confusion matrix, and classification report
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Step 7: Making Predictions on New Data (Examples)
for region in regions:
    new_data = pd.DataFrame({
        'Region_Code': [df['Region'].astype('category').cat.categories.get_loc(region)],
        'Year': [2025]  # Example year for prediction
    })
    prediction_proba = model.predict_proba(new_data)
    print(f"Prediction for a villain from {region} region in 2025: {prediction_proba}")