# Person Detection + Touching Shelf Video Generation

For running this demo, you will need to br able to download all datasets from Groundlight API. The dataset will return a JSON file `answer-block.json` that contains all the ROIs for that dataset with the images associate with them.

## Train Object Detection Model

`labeler.py` can be used to train a object detection detector from another detector that has the dataset labeled by the labelers (need to download the entire dataset first). Example command to run the script can be something like:

```bash
poetry run python labeler.py --frames /path_to_frames_folder --detector-id id_of_the_detector_to_train
```

# Generate Video

Assuming we have an existing detector to download all the pre-labeled dataset, the script can be ran in the following order:
1. `video-uploader.py`: Upload the video frame-by-frame to Groundlight API, creating a list of IQ IDs in `ids.txt` so that we can reference them in order when reconstructing the video.
2. `generate-frames.py`: Load the list of IQ IDs from a file, extract all the bounding boxes (you will need to download the entire dataset after finishing step 1 to get the ROIs), send cropped images to Groundlight to get binary classification results (e.g. Is the person holding an item on the shelf?), and create a same frame with the bounding boxes drawn on it with different colors. It also saves a txt file with the number of cloud escalations for each frame.
3. `video-uploader.py`: Reconstruct a new video with the bounding boxes frames.