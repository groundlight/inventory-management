#!/usr/bin/env python3
"""A script to process a video file and send frames to Groundlight ML model for Object Detection.
It does the following operations:
1. Read the video file frame by frame.
2. Detect motion in the frame.
3. Send the frame to Groundlight ML model (if motion has been detected).
4. Save the IQ IDs of the processed frames.
"""

import cv2
import argparse
import time
import logging
from framegrab import MotionDetector
from groundlight import Groundlight

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def process_video(
    video_file_path, detector_id, delay, output_file, pct_threshold: float = 1.0
):
    # Initialize Groundlight
    gl = Groundlight()

    # Open the video file
    video_capture = cv2.VideoCapture(video_file_path)

    motdet = MotionDetector(pct_threshold=pct_threshold)

    # Check if the video file opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open video file.")
        return

    # Get the total number of frames in the video
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames_processed = 0

    frame_number = 0
    with open(output_file, "w") as f:
        while True:
            # Read a frame
            ret, frame = video_capture.read()

            # If the frame was read correctly, ret will be True
            if not ret:
                break

            if not motdet.motion_detected(frame):
                print(f"No motion detected in frame {frame_number}/{total_frames}")
                frame_number += 1
                continue

            print(f"Motion detected in frame {frame_number}/{total_frames}")
            # Send the frame to Groundlight
            try:
                iq = gl.ask_async(detector_id, frame)
                f.write(f"{iq.id}\n")
                print(f"Frame {frame_number} processed: {iq.id}")
                total_frames_processed += 1
            except Exception as e:
                print(f"Error processing frame {frame_number}: {e}")

            frame_number += 1

            # Add delay between frames
            time.sleep(delay)

    # Release the video capture object
    video_capture.release()
    print("Video processing completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a video file and send frames to Groundlight ML model."
    )
    parser.add_argument(
        "--video-file", type=str, required=True, help="Path to the video file"
    )
    parser.add_argument(
        "--detector-id", type=str, required=True, help="ID of the detector"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.8,
        help="Delay between processing each frame in seconds",
    )
    parser.add_argument(
        "--pct-threshold",
        type=float,
        default=1.0,
        help="Percentage threshold for motion detection",
    )
    parser.add_argument(
        "--output-file", type=str, required=True, help="File to save the IQ IDs"
    )

    args = parser.parse_args()

    process_video(
        args.video_file,
        args.detector_id,
        args.delay,
        args.output_file,
        args.pct_threshold,
    )
