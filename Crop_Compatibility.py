import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('/Crop_recommendation.csv')
# Features and target columns
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
df[features] = StandardScaler().fit_transform(df[features])  # Standardize the features

# Create labels based on crop pair compatibility
# For this example, we'll create a simple binary label where crops are compatible (1) or not (0)
def generate_compatibility_labels(df, crop_1, crop_2):
    df['label_compatibility'] = (df['label'] == crop_1) | (df['label'] == crop_2)
    return df

# Example crop pair
crop_1 = 'rice'
crop_2 = 'jute'

# Prepare data for training
df = generate_compatibility_labels(df, crop_1, crop_2)

# Features (input conditions) and target (compatibility label)
X = df[features]
y = df['label_compatibility']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Function to predict compatibility for a given crop pair and input conditions
def predict_compatibility(input_conditions, crop_1, crop_2):
    input_conditions_scaled = StandardScaler().fit(df[features]).transform([input_conditions])[0]

    # Generate prediction
    compatibility_score = knn.predict([input_conditions_scaled])[0]

    # Output based on the model prediction
    if compatibility_score == 1:
        return f"Yes, the crops {crop_1} and {crop_2} are ideal to grow together."
    else:
        return f"No, the crops {crop_1} and {crop_2} are not ideal to grow together."

# Example input conditions: [N, P, K, temperature, humidity, ph, rainfall]
input_conditions = [90, 42, 43, 20.879744, 82.002744, 6.502985, 202.935536]  # Example for rice

# Predict the compatibility
result = predict_compatibility(input_conditions, crop_1, crop_2)
print(result)
