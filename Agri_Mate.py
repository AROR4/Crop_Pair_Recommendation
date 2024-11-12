import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Load the main crop dataset
df = pd.read_csv('/Crop_recommendation.csv')
# Select the columns related to environmental and nutrient requirements
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# Standardize the feature columns for fair comparison
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Function to find the closest matching crops based on nutrient and environmental similarity using KNN
def find_similar_crops(input_crop, df, top_n=5, exclude_crops=None):
    if exclude_crops is None:
        exclude_crops = []  # Default to an empty list if no crops are specified

    # Get the feature values for the input crop
    input_features = df[df['label'] == input_crop][features].values
    if len(input_features) == 0:
        print("Crop not found in dataset.")
        return []

    input_features = input_features[0].reshape(1, -1)  # Reshape for KNN

    # Remove the excluded crops from the dataset (including the input crop)
    filtered_df = df[~df['label'].isin([input_crop] + exclude_crops)]
    filtered_features = filtered_df[features].values

    # Fit the KNN model
    knn = NearestNeighbors(n_neighbors=top_n)
    knn.fit(filtered_features)

    # Find the nearest neighbors
    distances, indices = knn.kneighbors(input_features)

    # Get the names of the nearest crops and filter out duplicates
    similar_crops = []
    unique_crops = set()  # To track duplicates
    for i, distance in zip(indices[0], distances[0]):
        crop_name = filtered_df.iloc[i]['label']
        if crop_name not in unique_crops:
            unique_crops.add(crop_name)
            similar_crops.append((crop_name, distance))

    return similar_crops

# Example usage
input_crop = 'jute'  # Example crop name to find similar crops
exclude_crops = []  # List of crops to exclude
similar_crops = find_similar_crops(input_crop, df, top_n=5, exclude_crops=exclude_crops)

# Print similar crops with their Euclidean distances
print(f"Best crop pairs for {input_crop}:")
for crop, dist in similar_crops:
    print(f"{crop}: Distance = {dist}")



X = df[features]
y = df['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")


input_crop = 'cotton'
similar_crops = find_similar_crops(input_crop, df, top_n=5, exclude_crops=[])

# Separate crops and distances, ensuring each is a string or float
unique_crops = [crop for crop, dist in similar_crops]
unique_distances = [float(dist) for crop, dist in similar_crops]

# Plotting the bar graph for the results
plt.figure(figsize=(10, 6))
plt.bar(unique_crops, unique_distances, color='skyblue')
plt.title(f"Similar Crops to {input_crop} with Distance")
plt.xlabel('Crops')
plt.ylabel('KNN Distance')
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()
