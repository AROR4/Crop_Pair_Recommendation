import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the dataset
df = pd.read_csv('/Crop_recommendation.csv')

# Define features for clustering and nutrient matching
env_features = ['temperature', 'humidity', 'ph', 'rainfall']
nutrient_features = ['N', 'P', 'K']
all_features = env_features + nutrient_features

# Scale environmental and nutrient features
scaler_env = StandardScaler()
scaler_nutrient = StandardScaler()
df[env_features] = scaler_env.fit_transform(df[env_features])
df[nutrient_features] = scaler_nutrient.fit_transform(df[nutrient_features])

# Step 1: Cluster crops based on environmental features
kmeans = KMeans(n_clusters=5, random_state=42)
df['environment_cluster'] = kmeans.fit_predict(df[env_features])

# Step 2: Train a nutrient-compatibility model for complementarity
def generate_compatibility_labels(df):
    df['nutrient_compatibility'] = ((df['N'] > df['N'].mean()) & (df['K'] < df['K'].mean())) | \
                                   ((df['N'] < df['N'].mean()) & (df['K'] > df['K'].mean()))
    return df

df = generate_compatibility_labels(df)
X_nutrients = df[nutrient_features]
y_compatibility = df['nutrient_compatibility']

# Split and train the KNN model for nutrient compatibility
X_train, X_test, y_train, y_test = train_test_split(X_nutrients, y_compatibility, test_size=0.3, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print(f"Nutrient Model Accuracy: {accuracy_score(y_test, knn.predict(X_test)) * 100:.2f}%")

# Step 3: Define the function to suggest compatible crop pair with nutrient inputs
def suggest_compatible_crop_pair(input_conditions, input_nutrients):
    # Standardize the input environmental and nutrient conditions
    input_env_scaled = scaler_env.transform([input_conditions])[0]
    input_nutrients_scaled = scaler_nutrient.transform([input_nutrients])[0]

    # Predict the closest environmental cluster
    closest_cluster = kmeans.predict([input_env_scaled])[0]
    cluster_crops = df[df['environment_cluster'] == closest_cluster]

    # Find compatible crops within the cluster based on nutrient inputs and complementarity
    compatible_crops = []
    for crop in cluster_crops['label'].unique():
        crop_conditions = cluster_crops[cluster_crops['label'] == crop][nutrient_features].mean().values
        if knn.predict([crop_conditions])[0] == 1:
            # Calculate nutrient closeness
            nutrient_diff = np.linalg.norm(crop_conditions - input_nutrients_scaled)
            compatible_crops.append((crop, nutrient_diff))

    # Sort compatible crops by nutrient closeness and return the top two
    compatible_crops.sort(key=lambda x: x[1])
    if len(compatible_crops) >= 2:
        return f"Suggested Crop Pair: {compatible_crops[0][0]} and {compatible_crops[1][0]}"
    else:
        return "No compatible crop pair found in this environmental cluster."

# Example input: [temperature, humidity, ph, rainfall] and [N, P, K]
input_conditions = [20.879744, 82.002744, 6.502985, 202.935536]
input_nutrients = [30, 82, 63]

# Suggest compatible crop pair
suggested_pair = suggest_compatible_crop_pair(input_conditions, input_nutrients)
print(suggested_pair)
