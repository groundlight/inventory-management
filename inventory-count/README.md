# Retail Inventory Counting Demo


## Set up

To use this demo, you need to have a Groundlight account and a project with the following detectors, as configured in the `generate-frames.py` script.


```
    "inventory_demo_hand_touch": "det_2o...",
    "inventory_demo_ai_kits": "det_2o...",
    "inventory_demo_bittle": "det_2o...",
    "inventory_demo_robot_dog_kit": "det_2o...",
    "inventory_demo_mega_pixel": "det_2o...",
    "inventory_demo_realsense": "det_2o...",
    "inventory_demo_nvidia": "det_2o...",
    "inventory_demo_product_names": "det_2o...",
```

The queries should be configured as follows:

- `inventory_demo_hand_touch`: (BINARY mode)  Query = "Is there a real hand touching the boxes IN THE CENTER of the image? Answer NO if unclear or hand is touching another box that is not at the center."

- `inventory_demo_ai_kits`: (COUNT mode)  Query = "How many Bittle kits are in the picture?"

- `inventory_demo_bittle`: (COUNT mode)  Query = "How many Robot Dog Kits are in the picture? (Return item that has the name "Robot Dog")"

- `inventory_demo_robot_dog_kit`: (COUNT mode)  Query = "How many Robot Dog Kits are in the picture? (Return item that has the name "Robot Dog")"

- `inventory_demo_mega_pixel`: (COUNT mode)  Query = "How many Mega Pixel boxes are in the picture?"

- `inventory_demo_realsense`: (COUNT mode)  Query = "How many realsense camera boxes are in the picture?"

- `inventory_demo_nvidia`: (COUNT mode)  Query = "How many Nvidia (Jetson) device boxes are in the picture?"

- `inventory_demo_product_names`: (COUNT mode)  Query = "How many product names are in the picture? Draw a box on the product name for each boxes"

In every case, the confidence-threshold was set to 0.8.
