#!/usr/bin/env python3

import argparse
import os
import cv2
import framegrab
import imgcat
from tqdm.auto import tqdm

class FrameDecoder:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Error opening video file")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames: {self.total_frames}")
        self.motdet = None  # motion detector

    def configure_motdet(self, pct_threshold: float = 1.0):
        """Configure motion detection, with the threshold being % of pixels 
        that must change to be considered motion."""
        self.motdet = framegrab.MotionDetector(pct_threshold=pct_threshold)

    def decode(self, save_to: str | None = None, preview: bool = True):
        if save_to:
            os.makedirs(save_to, exist_ok=True)

        count = 0
        progress = tqdm(range(self.total_frames), desc="Extracting frames")
        for frame_num in progress:
            ret, bgr_frame = self.cap.read()
            if not ret:
                break
            if self.motdet:
                if self.motdet.motion_detected(bgr_frame):
                    print(f"Motion detected at frame {frame_num}.  Count of motion frames now {count+1}.")
                else:
                    continue
            count += 1
            if preview:
                print(f"Frame {frame_num}")
                # convert bgr to rgb
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                imgcat.imgcat(rgb_frame)
            if save_to:
                fn = f"{save_to}/frame_{frame_num:06d}.jpg"
                cv2.imwrite(fn, bgr_frame)
        print(f"Extracted {count} frames out of {self.total_frames}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("--save-to", type=str, default=None, help="Optional path to save the frames")
    parser.add_argument("--preview", action="store_true", help="Show the frames")
    parser.add_argument("--pct-threshold", type=float, default=None, help="Percentage threshold for motion detection")
    args = parser.parse_args()

    decoder = FrameDecoder(video_path=args.video_path)
    if args.pct_threshold:
        decoder.configure_motdet(pct_threshold=args.pct_threshold)
    decoder.decode(save_to=args.save_to, preview=args.preview)
