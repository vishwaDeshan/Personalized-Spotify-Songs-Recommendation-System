from similar_user_calculation_knn import calculate_similar_users
from optimal_point_calculation import calculate_optimal_point

# Calculate similar users and get their preferences
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

# weights for the Center of Gravity method
w_t = 0.2  # Weight for target emotion
w_p = 0.3  # Weight for preferences
w_m = 0.5  # Weight for preferred mood

# Calculate the optimal point using the preferences and weights
final_valence, final_arousal = calculate_optimal_point(
    emotion_preferences=emotion_preferences,
    target_emotion=target_emotion,
    preferred_mood=preferred_mood,
    w_t=w_t,
    w_p=w_p,
    w_m=w_m
)

print(f"Final Optimal Point - Valence: {final_valence}, Arousal: {final_arousal}")