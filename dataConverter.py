import os
import cv2
import h5py
import numpy as np
from sklearn.model_selection import train_test_split

# Define the folders containing the images
train_folder = "D:\\Alex\\Documents\\Master\\An 1\\Sem 2\\ResearchStage\\test\\fer2013_balanced\\train"
test_folder = "D:\\Alex\\Documents\\Master\\An 1\\Sem 2\\ResearchStage\\test\\fer2013_balanced\\test"

# Initialize lists to store images and labels
X = []
y = []

# Define the emotion categories
emotions = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]


# Function to load images from a given folder and append to the dataset
def load_images_from_folder(folder, emotion_label):
    for img_file in os.listdir(folder):
        img_path = os.path.join(folder, img_file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (112, 112))
            X.append(img)
            y.append(emotion_label)


# Load images from both train and test folders
for emotion in emotions:
    emotion_label = emotions.index(emotion)
    train_emotion_folder = os.path.join(train_folder, emotion)
    test_emotion_folder = os.path.join(test_folder, emotion)

    load_images_from_folder(train_emotion_folder, emotion_label)
    load_images_from_folder(test_emotion_folder, emotion_label)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split the combined dataset into train (80%), test (10%), and validation (10%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Save the datasets into an HDF5 file
with h5py.File("dataFixed112.h5", "w") as hf:
    hf.create_dataset("X_train", data=X_train)
    hf.create_dataset("y_train", data=y_train)
    hf.create_dataset("X_valid", data=X_valid)
    hf.create_dataset("y_valid", data=y_valid)
    hf.create_dataset("X_test", data=X_test)
    hf.create_dataset("y_test", data=y_test)
