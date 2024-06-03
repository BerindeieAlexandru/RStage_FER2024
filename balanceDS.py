import os
import shutil
import random

# Define the path to the fer2013 dataset folder
dataset_path = "fer2013"
balanced_path = "fer2013_balanced"

# Define the emotions
emotions = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]

# Initialize dictionaries to store the counts
train_counts = {emotion: 0 for emotion in emotions}
test_counts = {emotion: 0 for emotion in emotions}

# Count the images in the train folder
train_path = os.path.join(dataset_path, "train")
for emotion in emotions:
    emotion_path = os.path.join(train_path, emotion)
    train_counts[emotion] = len(os.listdir(emotion_path))

# Count the images in the test folder
test_path = os.path.join(balanced_path, "test")
for emotion in emotions:
    emotion_path = os.path.join(test_path, emotion)
    test_counts[emotion] = len(os.listdir(emotion_path))

# Define the number of images needed for each emotion in the test folder
test_needed = {
    "angry": 816,
    "disgusted": 1663,
    "fear": 750,
    "happy": 0,
    "neutral": 541,
    "sad": 527,
    "surprised": 943
}

# Copy images from train to test to balance the test set
for emotion in emotions:
    src_emotion_dir = os.path.join(train_path, emotion)
    dst_emotion_dir = os.path.join(balanced_path, "test", emotion)

    # Ensure the destination emotion directory exists
    if not os.path.exists(dst_emotion_dir):
        os.makedirs(dst_emotion_dir)

    # Number of images to copy
    num_images_to_copy = test_needed[emotion]

    if num_images_to_copy > 0:
        # Get a list of images in the source directory
        images = os.listdir(src_emotion_dir)

        # Randomly select the required number of images
        selected_images = random.sample(images, num_images_to_copy)

        # Copy the selected images
        for img in selected_images:
            src_img_path = os.path.join(src_emotion_dir, img)
            dst_img_path = os.path.join(dst_emotion_dir, img)
            shutil.copy(src_img_path, dst_img_path)

print("Balanced test set created successfully.")
