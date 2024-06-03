#generator with more faces
import os
from utilities import preprocess_images, load_model, generate_images

modelid = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
# ["SG161222/Realistic_Vision_V6.0_B1_noVAE", "runwayml/stable-diffusion-v1-5", "stabilityai/stable-cascade", "UnfilteredAI/NSFW-gen-v2", "stabilityai/stable-diffusion-xl-base-1.0"]
device = "cuda"


def main():
    num_images = 5
    stop_word = "stop"
    start = 0
    while True:
        # user_input = input("Enter a prompt (type 'stop' to exit): ")
        # if user_input.lower() == stop_word:
        #     print("Exiting...")
        #     break
        # else:
        generate_images("generator_space", modelid, device, num_images)

        # Preprocess each image one by one
        for i in range(start, start + num_images):
            image_filename = f"image_{i + 1}.png"
            preprocess_images(os.path.join("generator_space", image_filename), "fer2013\\train\\happy\\")
            print(f"Preprocessing completed for {image_filename}.")

        print("Preprocessing completed for all images.")
        break


if __name__ == "__main__":
    main()
