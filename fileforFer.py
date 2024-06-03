import os
import cv2
import h5py
import numpy as np
from sklearn.model_selection import train_test_split

train_folder = "D:\\Alex\\Documents\\Master\\An 1\\Sem 2\\ResearchStage\\test\\fer2013_balanced\\train"
test_folder = "D:\\Alex\\Documents\\Master\\An 1\\Sem 2\\ResearchStage\\test\\fer2013_balanced\\test"

X_train = []
y_train = []
X_test = []
y_test = []

emotions = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]

for emotion in emotions:
    train_emotion_folder = os.path.join(train_folder, emotion)
    test_emotion_folder = os.path.join(test_folder, emotion)

    for img_file in os.listdir(train_emotion_folder):
        img_path = os.path.join(train_emotion_folder, img_file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (48, 48))
        X_train.append(img)
        y_train.append(emotions.index(emotion))

    for img_file in os.listdir(test_emotion_folder):
        img_path = os.path.join(test_emotion_folder, img_file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (48, 48))
        X_test.append(img)
        y_test.append(emotions.index(emotion))

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

with h5py.File("data_CompleteV2.h5", "w") as hf:
    hf.create_dataset("X_train", data=X_train)
    hf.create_dataset("y_train", data=y_train)
    hf.create_dataset("X_valid", data=X_valid)
    hf.create_dataset("y_valid", data=y_valid)
    hf.create_dataset("X_test", data=X_test)
    hf.create_dataset("y_test", data=y_test)