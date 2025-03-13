# Inventory Management using Video Analytics

Building a complete inventory management solution generally involves connecting to an ERP or inventory management system.  However, a good first step in building such a system is to train visual detectors to identify objects of interest, and annotate a video so you can see how well it's working.  This demo will walk you through the process of training a detector and annotating a video.

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

