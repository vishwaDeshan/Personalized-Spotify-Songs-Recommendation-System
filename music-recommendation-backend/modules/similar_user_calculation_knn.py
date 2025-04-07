# similar_user_calculation_knn.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def calculate_similar_users(input_age, input_sex, input_profession, input_music, target_emotion='Angry', K=5):
    # Load the dataset from CSV
    try:
        df = pd.read_csv('../datasets/user_profile.csv')
    except FileNotFoundError:
        raise FileNotFoundError("User profiles CSV not found. Please adjust the file path.")
    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file is empty or malformed.")

    # Select relevant columns, including emotion-based preferences
    df = df[[
        'Id', 'Name', 'Age', 'Sex', 'Profession', 
        'Type of music you like to listen?', 
        'What type of music do you prefer to listen to when you\'re in a happy mood?',
        'What type of music do you prefer to listen to when you\'re sad?',
        'What type of music do you prefer to listen to when you\'re angry?',
        'What type of music do you prefer to listen to when you\'re in a relaxed mood?'
    ]]

    # Rename columns for simplicity
    df = df.rename(columns={
        'Type of music you like to listen?': 'Genres',
        'What type of music do you prefer to listen to when you\'re in a happy mood?': 'Happy_Preferences',
        'What type of music do you prefer to listen to when you\'re sad?': 'Sad_Preferences',
        'What type of music do you prefer to listen to when you\'re angry?': 'Angry_Preferences',
        'What type of music do you prefer to listen to when you\'re in a relaxed mood?': 'Relaxed_Preferences'
    })

    # Drop rows with missing values in key columns for similarity calculation
    df = df.dropna(subset=['Age', 'Sex', 'Profession', 'Genres'])

    # Clean up the Genres column (remove trailing commas and split into lists)
    df['Genres'] = df['Genres'].str.strip(',').str.split(', ')

    # Clean up the emotion preference columns (convert to lists, handle 'None' and missing values)
    emotion_columns = ['Happy_Preferences', 'Sad_Preferences', 'Angry_Preferences', 'Relaxed_Preferences']
    for col in emotion_columns:
        df[col] = df[col].fillna('').str.strip(',').str.split(', ')
        df[col] = df[col].apply(lambda x: [] if x == [''] else x)

    # Adjust input to match dataset conventions
    input_music = ['Rock', 'Classic', 'Pop']  # 'Classical' mapped to 'Classic'
    input_profession = 'Student'  # 'Undergraduate' mapped to 'Student'

    # --- Feature Engineering ---
    scaler = StandardScaler()
    df['Age_std'] = scaler.fit_transform(df[['Age']])
    df['Sex_binary'] = df['Sex'].map({'Male': 0, 'Female': 1})
    profession_dummies = pd.get_dummies(df['Profession'], prefix='Prof')
    mlb = MultiLabelBinarizer()
    genres_dummies = pd.DataFrame(mlb.fit_transform(df['Genres']), 
                                  columns=mlb.classes_, 
                                  index=df.index)
    feature_matrix = pd.concat([df['Age_std'], df['Sex_binary'], profession_dummies, genres_dummies], axis=1)

    # --- Prepare the Input User ---
    input_age_std = scaler.transform([[input_age]])[0][0]
    input_sex_binary = 0 if input_sex == 'Male' else 1
    input_profession_dummies = pd.Series(0, index=profession_dummies.columns)
    input_profession_dummies['Prof_' + input_profession] = 1
    input_genres_dummies = pd.Series(0, index=mlb.classes_)
    for genre in input_music:
        if genre in mlb.classes_:
            input_genres_dummies[genre] = 1
    input_user = np.concatenate([[input_age_std, input_sex_binary], 
                                 input_profession_dummies.values, 
                                 input_genres_dummies.values])

    # --- Apply KNN to Find Similar Users ---
    knn = NearestNeighbors(n_neighbors=K, metric='euclidean')
    knn.fit(feature_matrix)
    distances, indices = knn.kneighbors([input_user])
    similar_users = df.iloc[indices[0]]

    # Display details of similar users
    print("Top 5 Similar Users with Music Preferences for Each Emotion:")
    print(similar_users[[
        'Id', 'Name', 'Age', 'Sex', 'Profession', 'Genres',
        'Happy_Preferences', 'Sad_Preferences', 'Angry_Preferences', 'Relaxed_Preferences'
    ]])

    # Extract the preferences for the target emotion
    emotion_preferences = similar_users[f'{target_emotion}_Preferences'].tolist()
    print(f"\nSimilar Users' Preferences when {target_emotion}:", emotion_preferences)
    flattened_preferences = [pref for user_prefs in emotion_preferences for pref in user_prefs]
    print(f"Flattened {target_emotion} Preferences:", flattened_preferences)

    return flattened_preferences