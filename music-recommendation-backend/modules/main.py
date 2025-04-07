# main.py
from similar_user_calculation_knn import calculate_similar_users
from optimal_point_calculation import calculate_optimal_point

# Step 1: Calculate similar users and get their preferences
input_age = 26
input_sex = 'Male'
input_profession = 'Undergraduate'
input_music = ['Rock', 'Classical', 'Pop']
target_emotion = 'Angry'
preferred_mood = 'Calm'

emotion_preferences = calculate_similar_users(
    input_age=input_age,
    input_sex=input_sex,
    input_profession=input_profession,
    input_music=input_music,
    target_emotion=target_emotion,
    K=5
)

# Step 2: Calculate the optimal point using the preferences
final_valence, final_arousal = calculate_optimal_point(
    emotion_preferences=emotion_preferences,
    target_emotion=target_emotion,
    preferred_mood=preferred_mood
)