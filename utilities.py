import cv2
import os
from diffusers import StableDiffusionPipeline
import torch
from torch import autocast
import random
from time import time

pipe = None


def load_model(modelid, device):
    global pipe
    pipe = StableDiffusionPipeline.from_pretrained(modelid, torch_dtype=torch.float16)
    pipe.to(device)


def generate_prompt():
    # Define lists of prompt components
    contexts = [
        "A realistic", "A stylized", "A cartoonish", "A surreal", "A futuristic", "A vintage",
        "A whimsical", "A cyberpunk", "An abstract", "A retro", "A dystopian", "A romantic",
        "A dramatic", "A gothic", "A celestial", "A minimalist", "A detailed",
        "A impressionistic", "A expressionistic", "A photorealistic", "A low poly", "A retro", "A baroque",
        "A renaissance", "A medieval", "A ancient", "A modern", "A post-apocalyptic", "A dystopian", "A utopian",
        "A dream-like", "A nightmarish"
    ]
    ages = [
        "young", "middle aged", "elderly", "teenage", "child", "adult", "senior", "adolescent", "youthful",
        "mature", "aged", "prime", "golden years", "elderly"
    ]
    nationalities = [
        "Japanese", "American", "British", "French", "Chinese", "German", "Italian", "Spanish",
        "Russian", "Korean", "Indian", "Brazilian", "Mexican", "Canadian", "Australian",
        "Egyptian", "Nigerian", "South African", "Saudi Arabian", "Turkish", "Polish",
        "Ukrainian", "Argentine", "Colombian", "Venezuelan", "Thai", "Vietnamese", "Filipino",
        "Indonesian", "Malaysian", "Singaporean", "Swedish", "Norwegian", "Danish", "Finnish",
        "Irish", "Scottish", "Dutch", "Belgian", "Swiss", "Austrian", "Greek", "Portuguese",
        "Iranian", "Israeli", "Moroccan", "Algerian", "Tunisian", "Kenyan", "Ethiopian",
        "Ghanaian", "Jamaican", "Haitian", "Cuban", "Puerto Rican", "Dominican", "Peruvian",
        "Chilean", "Ecuadorian", "Guatemalan", "Honduran", "Salvadoran", "Nicaraguan",
        "Costa Rican", "Panamanian", "Icelandic", "New Zealander", "Fijian", "Samoan", "Tongan"
    ]
    races = [
        "Asian", "Caucasian", "African", "Hispanic", "European",
        "Latino", "Black", "White", "Indigenous", "Middle Eastern",
        "Pacific Islander", "Mixed race", "Biracial", "Multiracial"
    ]
    genders = ["male", "female"]
    emotions = ["happy"]
    facial_expressions = ["happy or smiling facial expression"]

    # Randomly select words from each list to construct the prompt
    selected_context = random.choice(contexts)
    selected_gender = random.choice(genders)
    selected_emotion = random.choice(emotions)
    selected_expression = random.choice(facial_expressions)
    selected_age = random.choice(ages)
    selected_nationality = random.choice(nationalities)
    selected_race = random.choice(races)

    # Construct the prompt using the selected words
    prompt = (
        f"{selected_context} image with a happy, cheerful, deeply well timed single {selected_gender} that is {selected_emotion} and has a detailed {selected_expression}. Person is expressive, {selected_age}, {selected_nationality} and has a {selected_race} genetic. Focus on eyes, eyebrows, mouth and other facial factors to express the happines."
    )

    return prompt


def generate_images(output_directory, modelid, device, num_images=10):
    start = 0
    if pipe is None:
        load_model(modelid, device)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    seed = 0
    negative_prompts = "deformed iris, mask, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, "
    "anime, text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, "
    "duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, "
    "mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, "
    "disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, "
    "blurry face, blurry body, missing body parts, missing body, missing head, missing face, "
    "fused fingers, too many fingers, long neck, malformed facial features, cloned face, deformed face, tattoo"
    total_time = 0
    for i in range(start, start + num_images):
        # Generate a random seed for each image
        seed += 200
        prompt = generate_prompt()
        print(f"Generated prompt: {prompt}")
        torch.manual_seed(seed)
        start_time = time()
        with autocast(device):
            image = pipe(prompt, 512, 512, 40, None, 5.5, negative_prompts).images[0]
        end_time = time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time
        print(f"Time taken to generate image {i + 1}: {elapsed_time:.2f} seconds")
        output_filename = os.path.join(output_directory, f"image_{i + 1}.png")
        image.save(output_filename)
        print(f"Image {i + 1}/{num_images} saved as {output_filename} with seed {seed}")
    average_time = total_time / num_images
    print(f"Average time per image: {average_time:.2f} seconds")


def preprocess_images(input_image_path, output_folder, size=(48, 48)):
    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the input image
    img = cv2.imread(input_image_path)

    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=7, minSize=(50, 50))

    # If no faces are found, print a message and return
    if len(faces) == 0:
        print(f"No faces found in {input_image_path}")
        return

    # Iterate through all detected faces in the image
    for idx, (x, y, w, h) in enumerate(faces):
        face_img = img[y:y + h, x:x + w]

        # Resize the extracted face image without losing aspect ratio
        aspect_ratio = size[0] / float(w)
        new_height = int(h * aspect_ratio)
        face_resized = cv2.resize(face_img, (size[0], new_height))

        # Convert the face image to grayscale
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

        # Write the preprocessed face image to the output folder
        output_filename = f"face_{idx + 1}_of_{os.path.splitext(os.path.basename(input_image_path))[0]}_.png"
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, face_gray)

        print(f"Preprocessed face {idx + 1} in {input_image_path} saved as {output_filename}")
