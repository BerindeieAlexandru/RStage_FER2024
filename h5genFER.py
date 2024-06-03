import os
import cv2
import numpy as np
import h5py

# Define the path to the dataset
dataset_path = 'path/to/dataset/folder/'
emotion_labels = ['angry', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised']
IMG_SHAPE = (120, 120, 3)


def load_data(dataset_path, emotion_labels, img_shape):
    X = []
    y = []

    for idx, emotion in enumerate(emotion_labels):
        emotion_folder = os.path.join(dataset_path, emotion)
        for img_file in os.listdir(emotion_folder):
            img_path = os.path.join(emotion_folder, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (img_shape[1], img_shape[0]))
                X.append(img)
                y.append(idx)

    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='int')

    return X, y


# Load training data
X_train, y_train = load_data(os.path.join(dataset_path, 'train'), emotion_labels, IMG_SHAPE)
# Load validation data
X_valid, y_valid = load_data(os.path.join(dataset_path, 'test'), emotion_labels, IMG_SHAPE)

# Normalize the data
X_train /= 255.0
X_valid /= 255.0

# Create HDF5 file
h5_path = 'dataBalanced.h5'
with h5py.File(h5_path, 'w') as h5f:
    h5f.create_dataset('X_train', data=X_train)
    h5f.create_dataset('y_train', data=y_train)
    h5f.create_dataset('X_valid', data=X_valid)
    h5f.create_dataset('y_valid', data=y_valid)

print("HDF5 file created successfully.")
