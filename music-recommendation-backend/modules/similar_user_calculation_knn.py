# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Hardcoded input for the target user (you can modify these)
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

# Adjust input to match dataset conventions
input_music = ['Rock', 'Classic', 'Pop']  # 'Classical' mapped to 'Classic'
input_profession = 'Student'  # 'Undergraduate' mapped to 'Student'

# --- Feature Engineering ---

# Standardize Age (KNN is distance-based, so normalization is important)
scaler = StandardScaler()
df['Age_std'] = scaler.fit_transform(df[['Age']])

# Encode Sex as binary (Male: 0, Female: 1)
df['Sex_binary'] = df['Sex'].map({'Male': 0, 'Female': 1})

# One-hot encode Profession
profession_dummies = pd.get_dummies(df['Profession'], prefix='Prof')

# Encode music genres using MultiLabelBinarizer (handles multi-label data)
mlb = MultiLabelBinarizer()
genres_dummies = pd.DataFrame(mlb.fit_transform(df['Genres']), 
                              columns=mlb.classes_, 
                              index=df.index)

# Combine features into a single feature matrix
feature_matrix = pd.concat([df['Age_std'], df['Sex_binary'], profession_dummies, genres_dummies], axis=1)

# --- Prepare the Input User ---

# Standardize input age
input_age_std = scaler.transform([[input_age]])[0][0]

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

# Combine into input user feature vector
input_user = np.concatenate([[input_age_std, input_sex_binary], 
                             input_profession_dummies.values, 
                             input_genres_dummies.values])

# --- Apply KNN to Find Similar Users ---

# Use NearestNeighbors to find the K nearest neighbors
K = 5  # Number of similar users to return
knn = NearestNeighbors(n_neighbors=K, metric='euclidean')
knn.fit(feature_matrix)

# Find the K nearest neighbors to the input user
distances, indices = knn.kneighbors([input_user])

# Get the similar users' details
similar_users = df.iloc[indices[0]]

# Display details of similar users
print("Top 5 Similar Users:")
print(similar_users[['Id', 'Name', 'Age', 'Sex', 'Profession', 'Genres']])

# --- Visualization with t-SNE (Optional) ---

# Apply t-SNE for 2D visualization
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(feature_matrix)

# Add t-SNE coordinates to the dataframe
df['tsne_1'] = tsne_results[:, 0]
df['tsne_2'] = tsne_results[:, 1]

# Plot the visualization
plt.figure(figsize=(10, 8))
# Plot all users in gray
plt.scatter(df['tsne_1'], df['tsne_2'], c='gray', alpha=0.5, label='Other Users')
# Highlight similar users in red
similar_indices = indices[0]
plt.scatter(df.iloc[similar_indices]['tsne_1'], 
            df.iloc[similar_indices]['tsne_2'], 
            c='red', label='Similar Users', edgecolor='black')
plt.legend()
plt.title('t-SNE Visualization with Similar Users Highlighted')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()