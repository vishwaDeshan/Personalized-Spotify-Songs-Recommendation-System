# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Hardcoded input
input_age = 26
input_sex = 'Male'
input_profession = 'Undergraduate'  # Will map to 'Student'
input_music = ['Rock', 'Classical', 'Pop']  # Adjust 'Classical' to 'Classic'

# Load the dataset from CSV
try:
    df = pd.read_csv('../datasets/user_profile.csv')
except FileNotFoundError:
    raise FileNotFoundError("User profiles CSV not found.")
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
input_music = ['Rock', 'Classic', 'Pop']  # 'Classical' corrected to 'Classic'
input_profession = 'Student'  # Map 'Undergraduate' to 'Student'

# Compute maximum age difference for normalization
min_age = df['Age'].min()
max_age = df['Age'].max()
max_age_diff = max_age - min_age

# Function to compute similarity between a user and the input
def compute_similarity(row):
    # Age similarity: 1 - normalized absolute difference
    age_sim = 1 - abs(row['Age'] - input_age) / max_age_diff
    
    # Sex similarity: 1 if match, 0 if different
    sex_sim = 1 if row['Sex'] == input_sex else 0
    
    # Profession similarity: 1 if match with 'Student', 0 if different
    prof_sim = 1 if row['Profession'] == input_profession else 0
    
    # Music genre similarity: Jaccard similarity
    user_genres = set(row['Genres'])
    input_genres = set(input_music)
    intersection = user_genres & input_genres
    union = user_genres | input_genres
    genre_sim = len(intersection) / len(union) if union else 0
    
    # Overall similarity: average of all components
    overall_sim = (age_sim + sex_sim + prof_sim + genre_sim) / 4
    return overall_sim

# Compute similarity for all users
df['Similarity'] = df.apply(compute_similarity, axis=1)

# Sort by similarity (descending) and get top 5 similar users
top_n = 5
similar_users = df.sort_values(by='Similarity', ascending=False).head(top_n)

# Display details of similar users
print("Top 5 Similar Users:")
print(similar_users[['Id', 'Name', 'Age', 'Sex', 'Profession', 'Genres', 'Similarity']])

# --- Visualization Preparation ---

# Standardize Age for the feature matrix
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
similar_indices = similar_users.index
plt.scatter(df.loc[similar_indices, 'tsne_1'], 
            df.loc[similar_indices, 'tsne_2'], 
            c='red', label=f'Top {top_n} Similar Users')
plt.legend()
plt.title('t-SNE Visualization of User Profiles')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()