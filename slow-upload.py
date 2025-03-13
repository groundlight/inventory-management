#!/usr/bin/env python3
"""A script to upload frames to a detector in a controlled manner.
You can specify the delay between uploads.
Also, you can shuffle consistently across runs, and resume from where you left off.
"""

import argparse
import os
import random
import time
import cv2

from groundlight import Groundlight
from imgcat import imgcat
from PIL import Image
from tqdm.auto import tqdm

def upload_frames(frames_path: str, detector_id: str, start: int, count: int, seed: int, delay: float, preview: bool):
    if detector_id:
        gl = Groundlight()
        detector = gl.get_detector(detector_id)

    # First find all the frames
    frames = [f for f in os.listdir(frames_path) if f.lower().endswith(('.jpg', '.jpeg'))]
    if len(frames) == 0:
        raise ValueError(f"No frames found in {frames_path}")
    frames.sort()
    print(f"Found {len(frames)} frames in {frames_path}")

    if seed != 0:
        # Shuffle with the requested seed.
        rng = random.Random(seed)
        rng.shuffle(frames)

    subset = range(start, start + count)
    next_idx = start
    try:
        for idx in tqdm(subset):
            frame = frames[idx]
            img_path = os.path.join(frames_path, frame)
            img = cv2.imread(img_path)
            if preview:
                print(f"Previewing {frame}")
                img = Image.open(img_path)
                imgcat(img)
            if detector_id:
                resp = gl.ask_ml(detector, image=img)
                print(f"Uploaded {frame} to detector {detector_id} with response {resp}")
            else:
                print(f"No detector specified.  Not uploading {frame}.")
            next_idx = idx + 1
            if delay >= 1:
                print(f"Waiting for {delay} seconds before uploading the next frame...")
            else:
                pass  # Don't bother them with a message if it's super short.
            time.sleep(delay)
    finally:
        if next_idx == start + count:
            print(f"\n\nRun was completed.")
            if next_idx < len(frames):
                print(f"To continue, use the following arguments:")
                print(f"  --start {next_idx} --count {count} --shuffle {seed}")
            else:
                print(f"\n\nThat's the end of this set of frames")
        else:
            print(f"\n\nRun was not completed. To resume this run, use the following arguments:")
            print(f"  --start {next_idx} --count {count} --shuffle {seed}")


def upload(frame_path: str, detector: str="Detector"):
    print(f"Uploading {frame_path} to detector {detector}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=str, help="Path to the frames")
    parser.add_argument("--detector-id", type=str, help="ID of the detector")
    parser.add_argument("--start", type=int, default=0, help="Starting frame index")
    parser.add_argument("--count", type=int, default=10, help="Number of frames to upload")
    parser.add_argument("--shuffle", type=int, default=0, help="Seed for the random number generator; 0 for no-shuffle")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between uploads (sec)")
    parser.add_argument("--preview", action="store_true", help="Preview the frames")

    args = parser.parse_args()
    upload_frames(frames_path=args.frames, detector_id=args.detector_id, start=args.start, count=args.count, seed=args.shuffle, delay=args.delay, preview=args.preview)
