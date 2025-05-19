<p align="center">
<img src="images/gl_logo.png" width="400px">
</p>


# Inventory Management using Video Analytics

Building a complete inventory management solution generally involves connecting to an ERP or inventory management system.  However, a good first step in building such a system is to train visual detectors to identify objects of interest, and annotate a video so you can see how well it's working.  This demo will walk you through the process of training a detector and annotating a video.

<div style="padding:56.25% 0 0 0;position:relative;"><iframe src="https://player.vimeo.com/video/1036091427?badge=0&amp;autopause=0&amp;player_id=0&amp;app_id=58479&amp;autoplay=1&amp;muted=1&amp;loop=1" frameborder="0" allow="autoplay; fullscreen; picture-in-picture; clipboard-write; encrypted-media" style="position:absolute;top:0;left:0;width:100%;height:100%;" title="Inventory Monitoring with Vision AI | Groundlight AI"></iframe></div><script src="https://player.vimeo.com/api/player.js"></script>

## Installation

The code supports installing the dependencies using both `uv` and `pip`.

```bash
# Clone this repo
git clone git@github.com:groundlight/inventory-management.git
```

Then install the dependencies using `uv` or `pip`

### Using `uv`

```bash
cd inventory-management
uv venv
uv sync --no-build-isolation
```

### Using `pip`

```bash
cd inventory-management
pip install -r requirements.txt
```

## Generating the data

### Videos

Download them, and put them into `data/videos` and DVC.

### Frames

First, consider how much motion detection you want, and preview.  Something like 0.1% will be very sensitive, and catch any tiny motion.  Something like 2% will catch fewer more representative frames.

```bash
./video-to-frames.py data/videos/videoname.mp4 \
    --pct-threshold 0.1 \
    --preview
```

Run the script to generate the frames:

```bash
./video-to-frames.py data/videos/videoname.mp4 \
    --pct-threshold 0.1 \
    --save-to data/frames/videoname-0.1
```

The generated frames will be saved to `data/frames/videoname-0.1`

