import cv2
import os


# Function to resize images and convert to grayscale
def preprocess_images(input_folder, output_folder, size=(48, 48)):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        # Read the image
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        # Skip non-image files
        if img is None:
            continue

        # Convert the image to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # If no faces are found, skip this image
        if len(faces) == 0:
            print(f"No faces found in {filename}")
            continue

        # Choose the largest face found (assuming there's only one face per image)
        max_area = 0
        max_face = faces[0]
        for (x, y, w, h) in faces:
            if w * h > max_area:
                max_area = w * h
                max_face = (x, y, w, h)

        x, y, w, h = max_face
        face_img = img[y:y + h, x:x + w]

        # Resize the extracted face image without losing aspect ratio
        aspect_ratio = size[0] / float(w)
        new_height = int(h * aspect_ratio)
        face_resized = cv2.resize(face_img, (size[0], new_height))

        # Convert the face image to grayscale
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

        # Write the preprocessed face image to the output folder
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, face_gray)

        print(f"Processed: {filename}")


# Specify input and output folders
input_folder = "extraDS"
output_folder = "output_images"

# Call the function to preprocess images
preprocess_images(input_folder, output_folder)
