#!/usr/bin/env python3
"""A script to process a video file and send frames to Groundlight ML model for Object Detection.
It does the following operations:
1. Read the video file frame by frame.
2. Detect motion in the frame.
3. Send the frame to Groundlight ML model (if motion has been detected).
4. Draw bounding boxes around the detected objects.
"""

import cv2
import argparse
import time
import logging
import numpy as np
from framegrab import MotionDetector
from groundlight import Groundlight, ROI

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

detectors = {
    "inventory_demo_hand_touch": "det_2oixHCE1BdX15uwxl7Z7Abz12re",
    "inventory_demo_ai_kits": "det_2oiwHya6h2MVUNYRi84DQKhCYCr",
    "inventory_demo_bittle": "det_2oiwQlpe7tvIaPOgviPQ2Ilxom6",
    "inventory_demo_robot_dog_kit": "det_2oiweCx6nAu4UCQvZTb9GoCnZ3U",
    "inventory_demo_mega_pixel": "det_2oiwlm5FmbopszLADZqi1VCFRN7",
    "inventory_demo_realsense": "det_2oiwr8tPZIfUuY4WFoL8kSdnkbA",
    "inventory_demo_nvidia": "det_2oiwwOw8pUwaJ8ynysjOc4NBRCc",
    "inventory_demo_product_names": "det_2oixBZKhRd0hvQonPB30FBUaE9Y",
}


def create_cropped_frame(frame: np.ndarray, roi: ROI) -> np.ndarray:
    """
    Crop the frame based on the ROI.

    Args:
        frame (np.ndarray): The frame to crop.
        roi (ROI): The region of interest.

    Returns:
        np.ndarray: The cropped frame.
    """

    top = roi.geometry.top
    bottom = roi.geometry.bottom
    left = roi.geometry.left
    right = roi.geometry.right

    # Get the frame size
    height, width, _ = frame.shape

    # Convert percentages to pixels
    top_px = int(top * height)
    bottom_px = int(bottom * height)
    left_px = int(left * width)
    right_px = int(right * width)

    # Crop the frame
    cropped_frame = frame[top_px:bottom_px, left_px:right_px]

    return cropped_frame


def draw_bounding_boxes(frame: np.ndarray, rois: list[ROI], color: str = "green", text: str | None = None) -> np.ndarray:
    """
    Draw bounding boxes around the detected objects in the frame. If text is provided, it will be displayed on top of the bounding boxes.

    Args:
        frame (np.ndarray): The frame to draw on.
        rois (list[ROI]): The list of regions of interest.
        color (str): The color of the bounding boxes (default is 'green').
        text (str): The text to display on the frame (default is None).

    Returns:
        np.ndarray: The frame with bounding boxes drawn.
    """

    # Copy the frame
    annotated_frame = frame.copy()

    # Draw bounding boxes around the detected objects
    for roi in rois:
        top = roi.geometry.top
        bottom = roi.geometry.bottom
        left = roi.geometry.left
        right = roi.geometry.right

        # Get the frame size
        height, width, _ = frame.shape

        # Convert percentages to pixels
        top_px = int(top * height)
        bottom_px = int(bottom * height)
        left_px = int(left * width)
        right_px = int(right * width)

        if color == "green":
            color_tuple = (0, 255, 0)
        elif color == "red":
            color_tuple = (0, 0, 255)
        elif color == "blue":
            color_tuple = (255, 0, 0)

        # Draw the bounding box
        cv2.rectangle(annotated_frame, (left_px, top_px), (right_px, bottom_px), color_tuple, 2)

        # Add text to the frame if provided with the same color as the bounding boxes
        if text is not None:
            cv2.putText(annotated_frame, text, (left_px, top_px - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_tuple, 1)

    return annotated_frame


def process_video(video_file: str, output_folder: str, delay: float, pct_threshold: float, start_frame: int) -> None:
    """
    Process a video file and send frames to Groundlight ML model for Object Detection. This script will also filtered out frames with no motion detected. If there is no motion detected for that frame, it will apply the previous state of the frame (e.g. draw the previous bounding boxes results and prints the last status, ...etc).

    Args:
        video_file (str): The path to the video file.
        output_folder (str): The path to the output folder.
        delay (float): Delay between processing each frame in seconds.
        pct_threshold (float): Percentage threshold for motion detection.
        start_frame (int): The starting index of the frame to process.

    Returns:
        None
    """

    # Initialize Groundlight
    gl = Groundlight()

    # Open the video file
    video_capture = cv2.VideoCapture(video_file)

    motdet = MotionDetector(pct_threshold=pct_threshold)

    # Check if the video file opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open video file.")
        return

    # Get the total number of frames in the video
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Store the previous state of the frame
    rois_blue = []
    rois_red = []
    rois_green = []

    # Get the number of detected objects
    num_objects_ai_kits: int = 0
    num_objects_bittle: int = 0
    num_objects_robot_dog_kit: int = 0
    num_objects_mega_pixel: int = 0
    num_objects_realsense: int = 0
    num_objects_nvidia: int = 0

    # Group ROIs by colors
    rois_blue: list[ROI] = []
    rois_red: list[ROI] = []
    rois_green: list[ROI] = []

    # Create a list of product names that the hand is touching
    hand_touch: list[str] = []

    frame_number = 0
    while True:
        # Read a frame
        ret, frame = video_capture.read()

        # If the frame was read correctly, ret will be True
        if not ret:
            break

        if frame_number < start_frame:
            frame_number += 1
            continue

        if not motdet.motion_detected(frame):
            print(f"No motion detected in frame {frame_number}/{total_frames}")

            # Draw bounding boxes around the detected objects
            annotated_frame = draw_bounding_boxes(frame, rois_blue, color="blue")
            annotated_frame = draw_bounding_boxes(annotated_frame, rois_red, color="red")

            # Add "Touching Item" text on top of the bounding boxes
            annotated_frame = draw_bounding_boxes(annotated_frame, rois_green, color="green", text="Touching Item")

            # Create a black image on the side of the frame to display the product names that the hand is touching and also the number of each objects (start from the top left corner)
            output_frame = annotated_frame
            cv2.putText(output_frame, "HAND TOUCH ITEM", (frame.shape[1] - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            for i, product_name in enumerate(hand_touch):
                cv2.putText(output_frame, product_name, (frame.shape[1] - 180, 60 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.putText(output_frame, "INVENTORY LIST", (frame.shape[1] - 180, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(
                output_frame, f"AI KIT: {num_objects_ai_kits}", (frame.shape[1] - 180, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
            cv2.putText(output_frame, f"Bittle: {num_objects_bittle}", (frame.shape[1] - 180, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(
                output_frame,
                f"ROBOT DOG KIT: {num_objects_robot_dog_kit}",
                (frame.shape[1] - 180, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                output_frame, f"Mega-pixel: {num_objects_mega_pixel}", (frame.shape[1] - 180, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
            cv2.putText(
                output_frame, f"REALSENSE: {num_objects_realsense}", (frame.shape[1] - 180, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
            cv2.putText(output_frame, f"JETSON: {num_objects_nvidia}", (frame.shape[1] - 180, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            output_file = f"{output_folder}/frame_{frame_number}.jpg"
            cv2.imwrite(output_file, output_frame)

            frame_number += 1
            continue

        print(f"Motion detected in frame {frame_number}/{total_frames}")

        iq_ai_kits = gl.submit_image_query(detectors["inventory_demo_ai_kits"], frame, 30)
        iq_bittle = gl.submit_image_query(detectors["inventory_demo_bittle"], frame, 30)
        iq_robot_dog_kit = gl.submit_image_query(detectors["inventory_demo_robot_dog_kit"], frame, 30)
        iq_mega_pixel = gl.submit_image_query(detectors["inventory_demo_mega_pixel"], frame, 30)
        iq_realsense = gl.submit_image_query(detectors["inventory_demo_realsense"], frame, 30)
        iq_nvidia = gl.submit_image_query(detectors["inventory_demo_nvidia"], frame, 30)
        iq_product_names = gl.submit_image_query(detectors["inventory_demo_product_names"], frame, 30)

        # Get the number of detected objects
        num_objects_ai_kits: int = iq_ai_kits.result.count
        num_objects_bittle: int = iq_bittle.result.count
        num_objects_robot_dog_kit: int = iq_robot_dog_kit.result.count
        num_objects_mega_pixel: int = iq_mega_pixel.result.count
        num_objects_realsense: int = iq_realsense.result.count
        num_objects_nvidia: int = iq_nvidia.result.count
        num_objects_product_names: int = iq_product_names.result.count

        # Group ROIs by colors
        rois_blue: list[ROI] = iq_product_names.rois
        rois_red: list[ROI] = []
        rois_green: list[ROI] = []

        # Create a list of product names that the hand is touching
        hand_touch: list[str] = []

        # Crop the frame based on the ROIs and send the cropped frames to Groundlight binary detector to see if a hand is touching the object
        # If the hand is touching the object, add to the rois_green list, otherwise add to the rois_red list
        if num_objects_ai_kits > 0:
            for roi in iq_ai_kits.rois:
                cropped_frame = create_cropped_frame(frame, roi)
                iq_hand_touch = gl.submit_image_query(detectors["inventory_demo_hand_touch"], cropped_frame, 30)
                if iq_hand_touch.result.label == "YES":
                    hand_touch.append("AI KIT")
                    rois_green.append(roi)
                else:
                    rois_red.append(roi)

        if num_objects_bittle > 0:
            for roi in iq_bittle.rois:
                cropped_frame = create_cropped_frame(frame, roi)
                iq_hand_touch = gl.submit_image_query(detectors["inventory_demo_hand_touch"], cropped_frame, 30)
                if iq_hand_touch.result.label == "YES":
                    hand_touch.append("Bittle")
                    rois_green.append(roi)
                else:
                    rois_red.append(roi)

        if num_objects_robot_dog_kit > 0:
            for roi in iq_robot_dog_kit.rois:
                cropped_frame = create_cropped_frame(frame, roi)
                iq_hand_touch = gl.submit_image_query(detectors["inventory_demo_hand_touch"], cropped_frame, 30)
                if iq_hand_touch.result.label == "YES":
                    hand_touch.append("ROBOT DOG KIT")
                    rois_green.append(roi)
                else:
                    rois_red.append(roi)

        if num_objects_mega_pixel > 0:
            for roi in iq_mega_pixel.rois:
                cropped_frame = create_cropped_frame(frame, roi)
                iq_hand_touch = gl.submit_image_query(detectors["inventory_demo_hand_touch"], cropped_frame, 30)
                if iq_hand_touch.result.label == "YES":
                    hand_touch.append("Mega-pixel")
                    rois_green.append(roi)
                else:
                    rois_red.append(roi)

        if num_objects_realsense > 0:
            for roi in iq_realsense.rois:
                cropped_frame = create_cropped_frame(frame, roi)
                iq_hand_touch = gl.submit_image_query(detectors["inventory_demo_hand_touch"], cropped_frame, 30)
                if iq_hand_touch.result.label == "YES":
                    hand_touch.append("REALSENSE")
                    rois_green.append(roi)
                else:
                    rois_red.append(roi)

        if num_objects_nvidia > 0:
            for roi in iq_nvidia.rois:
                cropped_frame = create_cropped_frame(frame, roi)
                iq_hand_touch = gl.submit_image_query(detectors["inventory_demo_hand_touch"], cropped_frame, 30)
                if iq_hand_touch.result.label == "YES":
                    hand_touch.append("JETSON")
                    rois_green.append(roi)
                else:
                    rois_red.append(roi)

        # Draw bounding boxes around the detected objects
        annotated_frame = draw_bounding_boxes(frame, rois_blue, color="blue")
        annotated_frame = draw_bounding_boxes(annotated_frame, rois_red, color="red")

        # Add "Touching Item" text on top of the bounding boxes
        annotated_frame = draw_bounding_boxes(annotated_frame, rois_green, color="green", text="Touching Item")

        # Create a black image on the side of the frame to display the product names that the hand is touching and also the number of each objects (start from the top left corner)
        output_frame = annotated_frame
        cv2.putText(output_frame, "HAND TOUCH ITEM", (frame.shape[1] - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        for i, product_name in enumerate(hand_touch):
            cv2.putText(output_frame, product_name, (frame.shape[1] - 180, 60 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(output_frame, "INVENTORY LIST", (frame.shape[1] - 180, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(output_frame, f"AI KIT: {num_objects_ai_kits}", (frame.shape[1] - 180, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(output_frame, f"Bittle: {num_objects_bittle}", (frame.shape[1] - 180, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(
            output_frame,
            f"ROBOT DOG KIT: {num_objects_robot_dog_kit}",
            (frame.shape[1] - 180, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            output_frame, f"Mega-pixel: {num_objects_mega_pixel}", (frame.shape[1] - 180, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        cv2.putText(
            output_frame, f"REALSENSE: {num_objects_realsense}", (frame.shape[1] - 180, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        cv2.putText(output_frame, f"JETSON: {num_objects_nvidia}", (frame.shape[1] - 180, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Save the frame to the output folder
        output_file = f"{output_folder}/frame_{frame_number}.jpg"
        cv2.imwrite(output_file, output_frame)

        frame_number += 1

        # Add delay between frames
        time.sleep(delay)

    # Release the video capture object
    video_capture.release()
    print("Video processing completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video file and send frames to Groundlight ML model.")
    parser.add_argument("--video-file", type=str, required=True, help="Path to the video file")
    parser.add_argument("--output-folder", type=str, required=True, help="Path to the output folder")
    parser.add_argument(
        "--delay",
        type=float,
        default=0,
        help="Delay between processing each frame in seconds",
    )
    parser.add_argument(
        "--pct-threshold",
        type=float,
        default=0.5,
        help="Percentage threshold for motion detection",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="The starting index of the frame to process",
    )

    args = parser.parse_args()

    process_video(
        video_file=args.video_file,
        output_folder=args.output_folder,
        delay=args.delay,
        pct_threshold=args.pct_threshold,
        start_frame=args.start_frame,
    )
