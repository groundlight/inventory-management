#!/usr/bin/env python3
"""A script to create a video from a list of images based on the iq_ids.
It does the following operations:
1. Read the list of iq_ids from the txt file.
2. Collect image paths based on iq_ids.
3. Create a video from the images.
"""

import os
import argparse
import cv2
from tqdm import tqdm


def create_video(image_folder, iq_ids_file, output_video, fps=30):
    # Read the list of iq_ids from the txt file
    with open(iq_ids_file, "r") as f:
        iq_ids = [line.strip() for line in f.readlines()]

    # List to hold image paths
    image_paths = []

    # Collect image paths based on iq_ids
    for iq_id in iq_ids:
        image_file = f"bbox_{iq_id}.jpg"
        image_path = os.path.join(image_folder, image_file)
        if os.path.exists(image_path):
            image_paths.append(image_path)
        else:
            print(f"Image {image_file} not found, skipping.")

    if not image_paths:
        print("No images found to create video.")
        return

    # Determine the width and height from the first image
    first_image = cv2.imread(image_paths[0])
    height, width, layers = first_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # You can use other codecs like 'XVID'
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Iterate through images and write them to the video
    for image_path in tqdm(image_paths, desc="Creating Video"):
        img = cv2.imread(image_path)
        video.write(img)

    # Release the video writer object
    video.release()
    print(f"Video saved as {output_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-folder",
        type=str,
        help="Path to the folder containing the images",
        required=True,
    )
    parser.add_argument(
        "--iq-ids",
        type=str,
        help="Path to the txt file containing iq_ids",
        required=True,
    )
    parser.add_argument(
        "--output-video", type=str, help="Path to save the output video", required=True
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for the video",
        required=False,
    )

    args = parser.parse_args()

    create_video(args.image_folder, args.iq_ids, args.output_video, fps=args.fps)
