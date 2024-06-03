import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Define the paths
dataset_path = 'fer2013_balanced'
emotions = ['angry', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised']


# Function to read images and convert to the required format
def load_images_from_folder(folder, emotion_label):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                resized_img = cv2.resize(img, (48, 48))
                pixels = ' '.join(map(str, resized_img.flatten()))
                images.append([emotion_label, pixels])
    return images


# Load the dataset
data = []
for emotion in emotions:
    emotion_folder = os.path.join(dataset_path, 'train', emotion)
    emotion_label = emotions.index(emotion)
    data.extend(load_images_from_folder(emotion_folder, emotion_label))

    emotion_folder = os.path.join(dataset_path, 'test', emotion)
    data.extend(load_images_from_folder(emotion_folder, emotion_label))

# Create a DataFrame
df = pd.DataFrame(data, columns=['emotion', 'pixels'])

# Split the dataset
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['emotion'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['emotion'])

# Add the 'Usage' column
train_df['Usage'] = 'Training'
val_df['Usage'] = 'PublicTest'
test_df['Usage'] = 'PrivateTest'

# Save to CSV
train_df.to_csv('train.csv', index=False)
val_df.to_csv('val.csv', index=False)
test_df.to_csv('test.csv', index=False)

print('CSV files created successfully!')
