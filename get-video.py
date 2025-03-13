import cv2
import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Create a video from a series of images.")
parser.add_argument("--frames-dir", required=True, help="Directory containing the frames")
parser.add_argument("--fps", type=int, default=30, help="Output video fps (default is 30)")
parser.add_argument("--output-path", required=True, help="Output video file location")

# Parse the arguments
args = parser.parse_args()

# Directory containing the images
image_dir = args.frames_dir

# Output video file
output_video = args.output_path

# Frame rate
frame_rate = args.fps

# Get list of image files and sort them by the number in the filename
image_files = sorted(
    [f for f in os.listdir(image_dir) if f.startswith("frame_") and f.endswith(".jpg")], key=lambda x: int(x.split("_")[1].split(".")[0])
)

# Read the first image to get the frame size
first_image_path = os.path.join(image_dir, image_files[0])
frame = cv2.imread(first_image_path)
height, width, layers = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

# Loop through all images and write them to the video
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    frame = cv2.imread(image_path)
    video.write(frame)

# Release the video writer object
video.release()

print(f"Video saved as {output_video}")
