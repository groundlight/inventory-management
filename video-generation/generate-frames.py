#!/usr/bin/env python3
"""A script to draw bounding boxes on images and crop them based on the bounding boxes + expansion.
It does the following operations:
1. Get all the bounding boxes for a frame in the JSON file.
2. Draw the bounding boxes with the color red if the ML returns YES, and green if the ML returns NO.
3. Save the image with the bounding boxes drawn and a txt file with the number of cloud escalations per frame.
"""

import json
import os
import argparse
from PIL import Image, ImageDraw, ImageFont
from groundlight import Groundlight
from tqdm import tqdm

gl = Groundlight()


# Function to draw bounding boxes on the image and crop images based on bounding boxes
def process_and_draw_image(
    image, detector_id, rois, expansion=0.05, score_threshold=0.95
):
    width, height = image.size
    font = ImageFont.load_default(size=25)
    cropped_images = []

    # Create a copy of the image for drawing bounding boxes
    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)

    # Number of escalations to the cloud labeler
    escalations = 0

    for i, roi in enumerate(rois):
        bbox = roi["init_args"]["geometry"]["init_args"]
        label = roi["init_args"]["label"]
        score = roi["init_args"]["score"]

        top = bbox["top"]
        left = bbox["left"]
        right = bbox["right"]
        bottom = bbox["bottom"]

        # Skip if the score is below the threshold
        if score < score_threshold:
            continue

        # Convert percentages to pixels
        top_px = int(top * height)
        left_px = int(left * width)
        right_px = int(right * width)
        bottom_px = int(bottom * height)

        # Crop the image with expansion from the original image
        top_px_exp = max(0, int((top - expansion) * height))
        left_px_exp = max(0, int((left - expansion) * width))
        right_px_exp = min(width, int((right + expansion) * width))
        bottom_px_exp = min(height, int((bottom + expansion) * height))

        cropped_image = image.crop(
            (left_px_exp, top_px_exp, right_px_exp, bottom_px_exp)
        )

        # Send cropped image to Groundlight
        iq = gl.submit_image_query(detector=detector_id, image=cropped_image, wait=30)

        if iq.result.label == "YES" and (
            iq.result.confidence is None
            or iq.result.confidence >= iq.confidence_threshold
        ):
            # Draw red if the ML returns YES
            draw.rectangle(
                [left_px, top_px, right_px, bottom_px], outline="red", width=4
            )
            label_text = f"Touching Item"
            draw.text((left_px, top_px - 25), label_text, fill="red", font=font)
        else:
            # Draw green if the ML returns NO
            draw.rectangle(
                [left_px, top_px, right_px, bottom_px], outline="green", width=4
            )
            label_text = f"Person"
            draw.text((left_px, top_px - 25), label_text, fill="red", font=font)

        # If the confidence is None, it means that the ML answer is from the cloud labeler, therefore we increment the escalations
        if iq.result.confidence is None:
            escalations += 1

        cropped_images.append((i, cropped_image))

    return image_with_boxes, cropped_images, escalations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=str, help="Path to the frames", required=True)
    parser.add_argument(
        "--save", type=str, help="Path to save the processed frames", required=True
    )
    parser.add_argument(
        "--crops", type=str, help="Path to save the cropped frames", required=True
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.95,
        help="Confidence threshold of the bounding boxes to include",
        required=False,
    )
    parser.add_argument(
        "--detector-id",
        type=str,
        help="ID of the detector to perform binary classification",
        required=True,
    )
    parser.add_argument(
        "--iq-ids",
        type=str,
        help="Path to the txt file containing iq_ids to process",
        required=True,
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting index for processing frames",
        required=False,
    )

    args = parser.parse_args()

    # Load the JSON file
    json_path = os.path.join(args.frames, "answer_block.json")
    with open(json_path) as f:
        data = json.load(f)

    # Create a dictionary for quick lookup of data by iq_id
    data_dict = {
        data["data"]["iq_id"][str(i)]: (
            data["data"]["current_best_answer_rois"][str(i)],
            data["data"]["image_id"][str(i)],
        )
        for i in range(len(data["data"]["iq_id"]))
    }

    # Load the list of iq_ids from the txt file
    with open(args.iq_ids, "r") as f:
        iq_ids_to_process = [line.strip() for line in f.readlines()]

    # Create the save directories if they don't exist
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if not os.path.exists(args.crops):
        os.makedirs(args.crops)

    # Open the escalations file in append mode
    with open("escalations.txt", "a") as f:
        # Iterate through the list of iq_ids to process
        for idx in tqdm(
            range(args.start, len(iq_ids_to_process)), desc="Processing Images"
        ):
            iq_id = iq_ids_to_process[idx]

            if iq_id not in data_dict:
                print(f"iq_id {iq_id} not found in data.")
                continue

            rois, image_id = data_dict[iq_id]

            # Skip if there are no bounding boxes (null) or image_id is not found
            if rois is None or len(rois) == 0 or image_id is None:
                continue

            # Construct the image file name
            image_file = image_id.replace("/", "_") + ".jpg"
            image_path = os.path.join(args.frames, image_file)

            # Open the image
            if not os.path.exists(image_path):
                print(f"Image {image_file} not found.")
                continue
            image = Image.open(image_path)

            # Process the image: draw bounding boxes and get cropped images
            image_with_boxes, cropped_images, escalations = process_and_draw_image(
                image.copy(),
                args.detector_id,
                rois,
                expansion=0.05,
                score_threshold=args.confidence_threshold,
            )

            # Save the image with bounding boxes
            output_path = os.path.join(args.save, f"bbox_{iq_id}.jpg")
            image_with_boxes.save(output_path)

            # Save the cropped images
            for i, cropped_image in cropped_images:
                cropped_output_path = os.path.join(args.crops, f"crop_{iq_id}_{i}.jpg")
                cropped_image.save(cropped_output_path)

            # Save the number of escalations to the file
            f.write(f"{iq_id}, {escalations}\n")

            # Print the processed index to resume if interrupted
            print(f"Last processed index: {idx}")

    print("Bounding boxes drawn and images saved. Cropped images saved.")
