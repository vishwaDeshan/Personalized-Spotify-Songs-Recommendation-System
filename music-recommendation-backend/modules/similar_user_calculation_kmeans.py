import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Hardcoded input for the virtual user
input_age = 26
input_sex = 'Male'
input_profession = 'Undergraduate'  # Will map to 'Student'
input_music = ['Rock', 'Classical', 'Pop']  # 'Classical' will be mapped to 'Classic'

# Load the dataset from CSV
try:
    df = pd.read_csv('../datasets/user_profile.csv')
except FileNotFoundError:
    raise FileNotFoundError("User profiles CSV not found. Please adjust the file path.")
except pd.errors.EmptyDataError:
    raise ValueError("The CSV file is empty or malformed.")

# Select relevant columns
df = df[['Id', 'Name', 'Age', 'Sex', 'Profession', 'Type of music you like to listen?']]

# Rename the music column for simplicity
df = df.rename(columns={'Type of music you like to listen?': 'Genres'})

# Drop rows with missing values in key columns
df = df.dropna(subset=['Age', 'Sex', 'Profession', 'Genres'])

# Clean up the Genres column (remove trailing commas and split into lists)
df['Genres'] = df['Genres'].str.strip(',').str.split(', ')

# Adjust input to match dataset conventions (based on assumed dataset values)
input_music = ['Rock', 'Classic', 'Pop']  # 'Classical' mapped to 'Classic'
input_profession = 'Student'  # 'Undergraduate' mapped to 'Student'

# --- Feature Engineering ---

# Standardize Age
mean_age = df['Age'].mean()
std_age = df['Age'].std()
df['Age_std'] = (df['Age'] - mean_age) / std_age

# Encode Sex as binary (Male: 0, Female: 1)
df['Sex_binary'] = df['Sex'].map({'Male': 0, 'Female': 1})

# One-hot encode Profession
profession_dummies = pd.get_dummies(df['Profession'], prefix='Prof')

# Encode music genres using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
genres_dummies = pd.DataFrame(mlb.fit_transform(df['Genres']), 
                              columns=mlb.classes_, 
                              index=df.index)

# Combine features into a feature matrix
feature_matrix = pd.concat([df['Age_std'], df['Sex_binary'], profession_dummies, genres_dummies], axis=1)

# --- K-Means Clustering ---

# Choose number of clusters (e.g., 5)
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(feature_matrix)

# --- Create Virtual User ---

# Standardize input age
input_age_std = (input_age - mean_age) / std_age

# Encode input sex
input_sex_binary = 0 if input_sex == 'Male' else 1

# One-hot encode input profession
input_profession_dummies = pd.Series(0, index=profession_dummies.columns)
input_profession_dummies['Prof_' + input_profession] = 1

# Binary encode input music genres
input_genres_dummies = pd.Series(0, index=mlb.classes_)
for genre in input_music:
    if genre in mlb.classes_:
        input_genres_dummies[genre] = 1

# Combine into virtual user feature vector
virtual_user = np.concatenate([[input_age_std, input_sex_binary], 
                               input_profession_dummies.values, 
                               input_genres_dummies.values])

# --- Predict Cluster for Virtual User ---

virtual_user_cluster = kmeans.predict([virtual_user])[0]
print(f"Virtual user assigned to cluster: {virtual_user_cluster}")

# --- Find Users in the Same Cluster ---

similar_users = df[df['Cluster'] == virtual_user_cluster]

# Display details of similar users
print("\nSimilar Users in the Same Cluster:")
print(similar_users[['Id', 'Name', 'Age', 'Sex', 'Profession', 'Genres']])

# --- Visualization with t-SNE ---

# Apply t-SNE for 2D visualization
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(feature_matrix)

# Add t-SNE coordinates to the dataframe
df['tsne_1'] = tsne_results[:, 0]
df['tsne_2'] = tsne_results[:, 1]

# Plot the visualization
plt.figure(figsize=(10, 8))
for cluster in range(n_clusters):
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['tsne_1'], cluster_data['tsne_2'], 
                label=f'Cluster {cluster}', alpha=0.7)
    
# Highlight the virtual user's cluster
virtual_cluster_data = df[df['Cluster'] == virtual_user_cluster]
plt.scatter(virtual_cluster_data['tsne_1'], virtual_cluster_data['tsne_2'], 
            c='red', label='Virtual User\'s Cluster', edgecolor='black')

plt.legend()
plt.title('t-SNE Visualization of User Clusters')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()