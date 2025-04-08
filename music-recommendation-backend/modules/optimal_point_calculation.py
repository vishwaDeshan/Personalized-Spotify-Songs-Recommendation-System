from collections import Counter
import matplotlib.pyplot as plt

def calculate_optimal_point(emotion_preferences, target_emotion='Angry', preferred_mood='Calm', w_t=0.2, w_p=0.3, w_m=0.5):
    # Validate weights
    total_weight = w_t + w_p + w_m
    if not (0.99 <= total_weight <= 1.01):  # Allow small floating-point errors
        raise ValueError(f"Weights must sum to 1, but got w_t={w_t}, w_p={w_p}, w_m={w_m} (sum={total_weight})")

    # Step 1: Aggregate music preferences
    music_type_counts = Counter(emotion_preferences)
    total_count = sum(music_type_counts.values())
    preference_vector = {music_type: count / total_count for music_type, count in music_type_counts.items()}
    print("Preference Vector:", preference_vector)

    # Step 2: Map music types to valence and arousal for preference point
    music_type_to_emotion = {
        'Sad Music': {'valence': -0.5, 'arousal': -0.3},
        'Joyful Music': {'valence': 0.7, 'arousal': 0.5},
        'Relaxing Music': {'valence': 0.3, 'arousal': -0.7},
        'Energetic Music': {'valence': 0.5, 'arousal': 0.8},
        'Aggressive Music': {'valence': -0.3, 'arousal': 0.7}
    }

    # Calculate preference valence and arousal
    pref_valence = 0.0
    pref_arousal = 0.0
    for music_type, weight in preference_vector.items():
        if music_type in music_type_to_emotion:
            pref_valence += weight * music_type_to_emotion[music_type]['valence']
            pref_arousal += weight * music_type_to_emotion[music_type]['arousal']
    print(f"Preference Valence: {pref_valence}, Preference Arousal: {pref_arousal}")

    # Step 3: Define emotion to valence-arousal mapping
    emotion_to_valence_arousal = {
        'Angry': {'valence': -0.7, 'arousal': 0.8},
        'Happy': {'valence': 0.8, 'arousal': 0.6},
        'Sad': {'valence': -0.6, 'arousal': -0.4},
        'Calm': {'valence': 0.4, 'arousal': -0.6}
    }

    # Get target emotion and preferred mood points
    target_valence = emotion_to_valence_arousal[target_emotion]['valence']
    target_arousal = emotion_to_valence_arousal[target_emotion]['arousal']
    preferred_valence = emotion_to_valence_arousal[preferred_mood]['valence']
    preferred_arousal = emotion_to_valence_arousal[preferred_mood]['arousal']
    print(f"Target Valence: {target_valence}, Target Arousal: {target_arousal}")
    print(f"Preferred Valence: {preferred_valence}, Preferred Arousal: {preferred_arousal}")

    # Step 4: Log the weights for the Center of Gravity method
    print(f"Weights - Target: {w_t}, Preferences: {w_p}, Preferred Mood: {w_m}")

    # Step 5: Calculate the final optimal point using weighted average
    final_valence = (w_t * target_valence) + (w_p * pref_valence) + (w_m * preferred_valence)
    final_arousal = (w_t * target_arousal) + (w_p * pref_arousal) + (w_m * preferred_arousal)
    print(f"Final Optimal Point - Valence: {final_valence}, Arousal: {final_arousal}")

    # Step 6: Clip values to [-1, 1]
    final_valence = max(min(final_valence, 1.0), -1.0)
    final_arousal = max(min(final_arousal, 1.0), -1.0)
    print(f"Clipped Optimal Point - Valence: {final_valence}, Arousal: {final_arousal}")

    # Step 7: Visualize the result
    plt.figure(figsize=(8, 8))
    plt.scatter(final_valence, final_arousal, color='red', label='Optimal Point', s=100)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel('Valence')
    plt.ylabel('Arousal')
    plt.title('Optimal Point in Valence-Arousal Space (Center of Gravity)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return final_valence, final_arousal