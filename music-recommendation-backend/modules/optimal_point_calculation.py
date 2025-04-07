# optimal_point_calculation.py
from collections import Counter
import matplotlib.pyplot as plt

def calculate_optimal_point(emotion_preferences, target_emotion='Angry', preferred_mood='Calm'):
    # Step 1: Aggregate music preferences
    music_type_counts = Counter(emotion_preferences)
    total_count = sum(music_type_counts.values())
    preference_vector = {music_type: count / total_count for music_type, count in music_type_counts.items()}
    print("Preference Vector:", preference_vector)

    # Step 2: Map music types to valence and arousal
    music_type_to_emotion = {
        'Sad Music': {'valence': -0.5, 'arousal': -0.3},
        'Joyful Music': {'valence': 0.7, 'arousal': 0.5},
        'Relaxing Music': {'valence': 0.3, 'arousal': -0.7},
        'Energetic Music': {'valence': 0.5, 'arousal': 0.8},
        'Aggressive Music': {'valence': -0.3, 'arousal': 0.7}
    }

    # Calculate initial valence and arousal
    valence = 0.0
    arousal = 0.0
    for music_type, weight in preference_vector.items():
        if music_type in music_type_to_emotion:
            valence += weight * music_type_to_emotion[music_type]['valence']
            arousal += weight * music_type_to_emotion[music_type]['arousal']
    print(f"Initial Valence: {valence}, Initial Arousal: {arousal}")

    # Step 3: Adjust for target emotion
    emotion_to_valence_arousal = {
        'Angry': {'valence': -0.7, 'arousal': 0.8},
        'Happy': {'valence': 0.8, 'arousal': 0.6},
        'Sad': {'valence': -0.6, 'arousal': -0.4},
        'Calm': {'valence': 0.4, 'arousal': -0.6}
    }

    target_valence = emotion_to_valence_arousal[target_emotion]['valence']
    target_arousal = emotion_to_valence_arousal[target_emotion]['arousal']

    weight_target = 0.3
    weight_preferences = 0.7
    adjusted_valence = (weight_target * target_valence) + (weight_preferences * valence)
    adjusted_arousal = (weight_target * target_arousal) + (weight_preferences * arousal)
    print(f"Adjusted Valence (after Target Emotion): {adjusted_valence}, Adjusted Arousal: {adjusted_arousal}")

    # Step 4: Adjust for preferred mood
    preferred_valence = emotion_to_valence_arousal[preferred_mood]['valence']
    preferred_arousal = emotion_to_valence_arousal[preferred_mood]['arousal']

    weight_adjusted = 0.4
    weight_preferred = 0.6
    final_valence = (weight_adjusted * adjusted_valence) + (weight_preferred * preferred_valence)
    final_arousal = (weight_adjusted * adjusted_arousal) + (weight_preferred * preferred_arousal)
    print(f"Final Optimal Point - Valence: {final_valence}, Arousal: {final_arousal}")

    # Step 5: Clip values to [-1, 1]
    final_valence = max(min(final_valence, 1.0), -1.0)
    final_arousal = max(min(final_arousal, 1.0), -1.0)
    print(f"Clipped Optimal Point - Valence: {final_valence}, Arousal: {final_arousal}")

    # Step 6: Visualize (optional)
    plt.figure(figsize=(8, 8))
    plt.scatter(final_valence, final_arousal, color='red', label='Optimal Point', s=100)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel('Valence')
    plt.ylabel('Arousal')
    plt.title('Optimal Point in Valence-Arousal Space')
    plt.legend()
    plt.grid(True)
    plt.show()

    return final_valence, final_arousal