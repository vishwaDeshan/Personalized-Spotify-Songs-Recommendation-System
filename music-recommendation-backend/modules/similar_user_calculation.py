import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

# Step 1: Load the dataset from CSV
try:
    df = pd.read_csv('./datasets/user_profile.csv')
except FileNotFoundError:
    raise FileNotFoundError("User profiles CSV not found.")
except pd.errors.EmptyDataError:
    raise ValueError("The CSV file is empty or malformed.")

# Clean and preprocess the DataFrame
df = df.replace('', np.nan)
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')  # Converts non-numeric to NaN
df['Age'] = df['Age'].fillna(df['Age'].median())  # Fill NaN with median of 'Age'
df['Sex'] = df['Sex'].fillna('Unknown')  # Handle missing sex
df['Profession'] = df['Profession'].fillna('Unknown')  # Handle missing profession
df['Type of music you like to listen?'] = df['Type of music you like to listen?'].str.split(', ').apply(lambda x: [item.strip() for item in x if item.strip()])

# Focus on relevant columns for similarity: Age, Sex, Profession, and Type of music
relevant_columns = ['Age', 'Sex', 'Profession', 'Type of music you like to listen?']
df = df[relevant_columns]

# Encode categorical variables
le_sex = LabelEncoder()
le_profession = LabelEncoder()

df['Sex'] = le_sex.fit_transform(df['Sex'])
df['Profession'] = le_profession.fit_transform(df['Profession'])

# Convert music preferences into binary vectors (one-hot encoding)
music_types = set()
for music_list in df['Type of music you like to listen?']:
    music_types.update(music_list)

music_types = sorted(list(music_types))  # Unique music types

# Create binary columns for each music type
for music_type in music_types:
    df[music_type] = df['Type of music you like to listen?'].apply(lambda x: 1 if music_type in x else 0)

# Drop the original music list column
df = df.drop(columns=['Type of music you like to listen?'])

# Normalize numerical data (Age)
scaler = StandardScaler()
df['Age'] = scaler.fit_transform(df[['Age']])

# Prepare the feature matrix for clustering (all columns except non-numeric indices if any)
X = df.drop(columns=['Sex', 'Profession'])  # Keep numerical and binary columns for K-means

# Determine the optimal number of clusters
def find_optimal_clusters(X, max_k=10):
    inertias = []
    silhouette_scores = []
    K = range(2, max_k + 1)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    
    return inertias, silhouette_scores, K

inertias, silhouette_scores, K = find_optimal_clusters(X)

# Plot elbow curve (optional, for visualization)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(K, inertias, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')

plt.subplot(1, 2, 2)
plt.plot(K, silhouette_scores, 'bx-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score For Optimal k')

plt.tight_layout()
plt.show()

# Choose k=3 based on elbow or silhouette score (adjust if needed)
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

# Add cluster labels to the DataFrame
df['Cluster'] = kmeans.labels_

# Function to find similar users
def find_similar_users(age, sex, profession, music_preferences):
    # Preprocess input
    input_data = {
        'Age': [age],
        'Sex': [sex],
        'Profession': [profession],
        'Type of music you like to listen?': [music_preferences]
    }

    input_df = pd.DataFrame(input_data)

    # Encode categorical variables, with handling for unseen labels in 'Profession'
    input_df['Sex'] = le_sex.transform(input_df['Sex'])
    
    # Handle unseen 'Profession' values by adding a fallback 'Unknown' category
    try:
        input_df['Profession'] = le_profession.transform(input_df['Profession'])
    except KeyError:
        input_df['Profession'] = le_profession.transform(['Unknown'])

    # Create binary vectors for music preferences
    for music_type in music_types:
        input_df[music_type] = input_df['Type of music you like to listen?'].apply(lambda x: 1 if music_type in x else 0)

    # Drop the original music list column
    input_df = input_df.drop(columns=['Type of music you like to listen?'])

    # Normalize Age
    input_df['Age'] = scaler.transform(input_df[['Age']])

    # Prepare input features
    input_features = input_df[X.columns]

    # Predict the cluster
    input_cluster = kmeans.predict(input_features)[0]

    # Find users in the same cluster
    similar_users = df[df['Cluster'] == input_cluster]

    # Calculate distances to sort by similarity
    distances = euclidean_distances(input_features, X)
    similar_users['Distance'] = distances[0]
    similar_users = similar_users.sort_values(by='Distance')

    return similar_users[['Age', 'Sex', 'Profession'] + music_types.tolist()]

# Hardcoded input
input_age = 26
input_sex = 'Male'
input_profession = 'Undergraduate'  # This should work now even if 'Undergraduate' is unseen
input_music = ['Rock', 'Classical', 'Pop']

# Find similar users
similar_users = find_similar_users(input_age, input_sex, input_profession, input_music)

print("Similar Users for input (Age: 26, Sex: Male, Profession: Undergraduate, Music: Rock, Classical, Pop):")
print(similar_users.head(5))  # Show top 5 most similar users
