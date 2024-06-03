import os

# Define the path to the fer2013 dataset folder
dataset_path = ".\\archive"

# Define the emotions
emotions = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]

# Initialize dictionaries to store the counts
train_counts = {emotion: 0 for emotion in emotions}
test_counts = {emotion: 0 for emotion in emotions}
sum = 0
# Count the images in the train folder
train_path = os.path.join(dataset_path, "train")
for emotion in emotions:
    emotion_path = os.path.join(train_path, emotion)
    train_counts[emotion] = len(os.listdir(emotion_path))
    sum += train_counts[emotion]

# Count the images in the test folder
test_path = os.path.join(dataset_path, "test")
for emotion in emotions:
    emotion_path = os.path.join(test_path, emotion)
    test_counts[emotion] = len(os.listdir(emotion_path))
    sum += test_counts[emotion]
print(sum)
# Find the maximum count of images across all emotions in the train folder
max_train_count = max(train_counts.values())

# Find the maximum count of images across all emotions in the test folder
max_test_count = max(test_counts.values())

# Calculate the number of images needed for each emotion in the train folder
train_needed = {emotion: max_train_count - count for emotion, count in train_counts.items()}

# Calculate the number of images needed for each emotion in the test folder
test_needed = {emotion: max_test_count - count for emotion, count in test_counts.items()}

# Find the maximum length of emotion names for proper alignment
max_emotion_length = max(len(emotion) for emotion in emotions)

# Print the results in a table-like format with aligned columns
print(f"{'Emotion':<{max_emotion_length}}\t{'Train Count':>12}\t{'Test Count':>11}\t{'Train Needed':>13}\t{'Test Needed':>12}")
print(f"{'-------':<{max_emotion_length}}\t{'-----------':>12}\t{'----------':>11}\t{'------------':>13}\t{'-----------':>12}")
for emotion in emotions:
    print(f"{emotion:<{max_emotion_length}}\t{train_counts[emotion]:>12}\t{test_counts[emotion]:>11}\t{train_needed[emotion]:>13}\t{test_needed[emotion]:>12}")