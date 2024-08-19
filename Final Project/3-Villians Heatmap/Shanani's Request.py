import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the JSON data from the file
with open('./output/villains_data.json', 'r') as file:
    data = json.load(file)

# Step 2: Convert JSON data to DataFrame
df = pd.DataFrame(data)

# Step 3: Data Processing
# Assuming we want to predict if the character belongs to the "Marvel" universe
df['Is_Marvel'] = df['universe'] == 'Prime Marvel Universe'
df['Is_Marvel'] = df['Is_Marvel'].astype(int)

# Convert species to categorical codes (you might have other processing needs)
df['Species_Code'] = df['species'].astype('category').cat.codes

# Filling missing places of birth with a placeholder and converting to categorical codes
df['place_of_birth'].replace('', 'Unknown', inplace=True)
df['Place_Code'] = df['place_of_birth'].astype('category').cat.codes

# Step 4: Splitting Data for Training and Testing
X = df[['Species_Code', 'Place_Code']]
y = df['Is_Marvel']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Training the Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Step 6: Evaluating the Model using accuracy with binary predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Step 7: Making Predictions on New Data (Examples:)
# Example 1: Human from New York
new_data = pd.DataFrame({
    'Species_Code': [93],  # Human
    'Place_Code': [386]    # New York
})
prediction_proba = model.predict_proba(new_data)
print(f"Prediction for Human from New York: {prediction_proba}")

# Example 2: Alien from an unknown place
new_data = pd.DataFrame({
    'Species_Code': [1],   # Alien
    'Place_Code': [607]    # Unknown
})
prediction_proba = model.predict_proba(new_data)
print(f"Prediction for Alien from Unknown: {prediction_proba}")

# Example 3: A Demon from Tokyo, Japan
new_data = pd.DataFrame({
    'Species_Code': [57],  # Demon
    'Place_Code': [271]    # Tokyo, Japan
})
prediction_proba = model.predict_proba(new_data)
print(f"Prediction for Demon from Tokyo, Japan: {prediction_proba}")

# Example 4: A Robot from Atlantis
new_data = pd.DataFrame({
    'Species_Code': [158],  # Robot
    'Place_Code': [66]      # Atlantis
})
prediction_proba = model.predict_proba(new_data)
print(f"Prediction for Robot from Atlantis: {prediction_proba}")

# Example 5: An Elder from a fictional place
new_data = pd.DataFrame({
    'Species_Code': [73],  # Elder
    'Place_Code': [303]    # Korriban (fictional planet)
})
prediction_proba = model.predict_proba(new_data)
print(f"Prediction for Elder from Korriban: {prediction_proba}")

# Example 6: A Cosmic Entity from Earth
new_data = pd.DataFrame({
    'Species_Code': [39],  # Cosmic Entity
    'Place_Code': [170]    # Earth
})
prediction_proba = model.predict_proba(new_data)
print(f"Prediction for Cosmic Entity from Earth: {prediction_proba}")

# Example 7: Leprechaun from Ireland
new_data = pd.DataFrame({
    'Species_Code': [135],  # Leprechaun
    'Place_Code': [263]     # Ireland
})
prediction_proba = model.predict_proba(new_data)
print(f"Prediction for Leprechaun from Ireland: {prediction_proba}")

# Example 8: A Saiyan from Krypton
new_data = pd.DataFrame({
    'Species_Code': [165],  # Saiyan
    'Place_Code': [305]     # Krypton
})
prediction_proba = model.predict_proba(new_data)
print(f"Prediction for Saiyan from Krypton: {prediction_proba}")

# Example 9: A Kryptonian from Gotham City
new_data = pd.DataFrame({
    'Species_Code': [126],  # Kryptonian
    'Place_Code': [221]     # Gotham City
})
prediction_proba = model.predict_proba(new_data)
print(f"Prediction for Kryptonian from Gotham City: {prediction_proba}")

# Example 10: A Human // Zombie from Asgard
new_data = pd.DataFrame({
    'Species_Code': [105],  # Human // Zombie
    'Place_Code': [63]      # Asgard
})
prediction_proba = model.predict_proba(new_data)
print(f"Prediction for Human // Zombie from Asgard: {prediction_proba}")

# Example 11: Human from different places
places = [386, 321, 242]  # New York, London, Hawaii
for place in places:
    new_data = pd.DataFrame({
        'Species_Code': [93],  # Human
        'Place_Code': [place]
    })
    prediction_proba = model.predict_proba(new_data)
    print(f"Prediction for Human from place code {place}: {prediction_proba}")

# Example 12: Unknown species from Unknown place
new_data = pd.DataFrame({
    'Species_Code': [0],  # Unknown species
    'Place_Code': [607]   # Unknown place
})
prediction_proba = model.predict_proba(new_data)
print(f"Prediction for Unknown species from Unknown place: {prediction_proba}")

print("\n*************************************\n")

# Additional Examples for Other Interesting Combinations
# 1. Alien from Krypton
new_data = pd.DataFrame({
    'Species_Code': [1],   # Alien
    'Place_Code': [305]    # Krypton
})
prediction_proba = model.predict_proba(new_data)
print(f"Prediction for Alien from Krypton: {prediction_proba}")

# 2. Cosmic Entity from Earth
new_data = pd.DataFrame({
    'Species_Code': [39],  # Cosmic Entity
    'Place_Code': [170]    # Earth
})
prediction_proba = model.predict_proba(new_data)
print(f"Prediction for Cosmic Entity from Earth: {prediction_proba}")

# 3. Mutant from Ancient Egypt
new_data = pd.DataFrame({
    'Species_Code': [145],  # Mutant
    'Place_Code': [39]      # Ancient Egypt
})
prediction_proba = model.predict_proba(new_data)
print(f"Prediction for Mutant from Ancient Egypt: {prediction_proba}")

# 4. Demon from Tokyo, Japan
new_data = pd.DataFrame({
    'Species_Code': [57],  # Demon
    'Place_Code': [271]    # Tokyo, Japan
})
prediction_proba = model.predict_proba(new_data)
print(f"Prediction for Demon from Tokyo, Japan: {prediction_proba}")

# 5. Human // Altered from New York
new_data = pd.DataFrame({
    'Species_Code': [94],  # Human // Altered
    'Place_Code': [386]    # New York
})
prediction_proba = model.predict_proba(new_data)
print(f"Prediction for Human // Altered from New York: {prediction_proba}")

# 6. Robot from Cybertron
new_data = pd.DataFrame({
    'Species_Code': [158],  # Robot
    'Place_Code': [140]     # Cybertron
})
prediction_proba = model.predict_proba(new_data)
print(f"Prediction for Robot from Cybertron: {prediction_proba}")

# 7. Vampire from Transylvania
new_data = pd.DataFrame({
    'Species_Code': [181],  # Vampire
    'Place_Code': [592]     # Transylvania
})
prediction_proba = model.predict_proba(new_data)
print(f"Prediction for Vampire from Transylvania: {prediction_proba}")

# 8. Cyborg from a Post-Apocalyptic Setting (Example Place: Dystopian Future)
new_data = pd.DataFrame({
    'Species_Code': [44],  # Cyborg
    'Place_Code': [315]    # Example Place: Dystopian Future
})
prediction_proba = model.predict_proba(new_data)
print(f"Prediction for Cyborg from Dystopian Future: {prediction_proba}")

# 9. God // Eternal from Olympus
new_data = pd.DataFrame({
    'Species_Code': [84],  # God / Eternal
    'Place_Code': [412]    # Olympus
})
prediction_proba = model.predict_proba(new_data)
print(f"Prediction for God // Eternal from Olympus: {prediction_proba}")

# 10. Asgardian from Jotunheim

new_data = pd.DataFrame({
    'Species_Code': [27],  # Asgardian
    'Place_Code': [274]    # Jotunheim
    })
prediction_proba = model.predict_proba(new_data)
print(f"Prediction for Asgardian from Jotunheim: {prediction_proba}")


# # Print the mapping between species and their corresponding codes
# species_mapping = df['species'].astype('category').cat.categories
# species_codes = df['species'].astype('category').cat.codes
#
# species_dict = dict(enumerate(species_mapping))
# print("Species Code Mapping:")
# for code, species in species_dict.items():
#     print(f"{code}: {species}")
#
#
#
# print("\n**************************************\n")
#
# # Print the mapping between place_of_birth and their corresponding codes
# place_mapping = df['place_of_birth'].astype('category').cat.categories
# place_dict = dict(enumerate(place_mapping))
# print("Place Code Mapping:")
# for code, place in place_dict.items():
#     print(f"{code}: {place}")



