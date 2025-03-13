#!/usr/bin/env python3
"""A script to label images based on the bounding boxes and the answer.
It does the following operations:
1. Load the JSON file containing the bounding boxes and answers.
2. Iterate through the images and label them based on the bounding boxes and answers (to Groundlight).
"""

import requests
import json
import os
import argparse
import logging
import time
from abc import ABC
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    field_validator,
    model_validator,
)
from groundlight import Groundlight
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class StrictBaseModel(BaseModel, ABC):
    """
    pydantic BaseModel that has some stricter config settings and some helper methods.

    "cls" and "init_args" are reserved fields, do not use them in sub-models.

    Some benefits we get by using pydantic:
    * type checking and auto conversion of input data where possible (helps ensure we have json serializability where we want it)
    * lots of helpful built in methods (dict(), __repr__(), __eq__(), copy(), dumps(), etc.)
    * `Field` can be used for many basic data validation tasks (e.g. min/max length, regex, etc.)
    """

    def __init__(self, **data):
        """
        Extend pydantic's __init__() to handle our custom "cls" and "init_args" format.
        """
        if "cls" in data:
            assert (
                data["cls"] == self.__class__.__name__
            ), f"expecting 'cls' type {self.__class__.__name__}, got {data}"
            super().__init__(**data.get("init_args", {}))
        else:
            super().__init__(**data)

    def dict(self, **kwargs) -> dict:
        """
        Extend pydantic's dict() to optionally output in our custom "cls" and "init_args" format.
        """
        self.model_dump(**kwargs)

    def model_dump(self, with_cls_field=False, **kwargs) -> dict:
        """
        Extend pydantic's model_dump() to optionally output in our custom "cls" and "init_args" format.
        """
        if with_cls_field:
            return {
                "cls": self.__class__.__name__,
                "init_args": super().model_dump(**kwargs),
            }
        return super().model_dump(**kwargs)

    model_config = ConfigDict(extra="forbid", validate_default=True, frozen=True)


class BBoxGeometry(StrictBaseModel):
    """
    Bounding box for a region
    """

    model_config = ConfigDict(frozen=False)

    left: NonNegativeFloat  # location of the left edge of the box (commonly called x1)
    top: NonNegativeFloat  # location of the top edge (commonly called y1, measured from the top of the img)
    right: NonNegativeFloat  # location of the right edge (commonly called x2)
    bottom: NonNegativeFloat  # location of the bottom edge (commonly called y2, measured from the top of the img)
    version: str = "2.0"  # current default version

    # TODO: add better validation that unit-scale ROIs are defined correctly (bottom, right <= 1)

    @property
    def width(self):
        return self.right - self.left

    @property
    def height(self):
        return self.bottom - self.top

    @property
    def midpoint(self):
        "Return the center of the box (x, y)"
        return (0.5 * (self.left + self.right), 0.5 * (self.top + self.bottom))

    @property
    def coords(self):
        "return a tuple with just the coords of the bounding box (left, top, right, bottom)"
        return (self.left, self.top, self.right, self.bottom)  # x1, y1, x2, y2

    @property
    def area(self):
        return self.width * self.height

    @property
    def is_unit_scale(self):
        return self.left >= 0 and self.top >= 0 and self.right <= 1 and self.bottom <= 1

    def to_unit_scale(self, width: int, height: int) -> "BBoxGeometry":
        """
        Convert the bounding box from pixel scale to unit scale.
        """
        # TODO: Keep track of image aspect ratio!
        if self.is_unit_scale:
            logger.warning(
                "BBoxGeometry is already in unit scale. Skipping conversion."
            )
            return self
        return BBoxGeometry(
            left=self.left / width,
            top=self.top / height,
            right=self.right / width,
            bottom=self.bottom / height,
        )

    def to_pixel_scale(self, width: int, height: int) -> "BBoxGeometry":
        """
        Convert the bounding box from unit scale to pixel scale.
        """
        if not self.is_unit_scale:
            logger.warning(
                "BBoxGeometry is already in pixel scale. Skipping conversion."
            )
            return self
        return BBoxGeometry(
            left=round(self.left * width),
            top=round(self.top * height),
            right=round(self.right * width),
            bottom=round(self.bottom * height),
        )

    @model_validator(mode="before")
    @classmethod
    def validate_left_and_right(cls, values):
        "check that left coordinate is less than the right coordinate"
        if "left" in values and "right" in values and values["left"] > values["right"]:
            raise ValueError(
                f"BBoxGeometry left coordinate must be less than right: {values}"
            )
        return values

    @model_validator(mode="before")
    @classmethod
    def validate_top_and_bottom(cls, values):
        "check that bottom coordinate is less than the top coordinate"
        if "top" in values and "bottom" in values and values["top"] > values["bottom"]:
            raise ValueError(
                f"BBoxGeometry top coordinate must be less than bottom: {values}"
            )
        return values

    @field_validator("version")
    @classmethod
    def update_version_and_warn(cls, v):
        CURRENT_VERSION = "2.0"
        if v != CURRENT_VERSION:
            logger.warning(
                f"Updating BBoxGeometry version from {v} to {CURRENT_VERSION}"
            )
        return CURRENT_VERSION


class ROI(StrictBaseModel, frozen=False):
    """
    Class describing a region of interest in an image.
    """

    label: str  # String class label for what is contained in the region
    geometry: BBoxGeometry  # Geometry of the region
    score: float = 1.0  # 1.0 -> perfect annotation that we are highly confident is correct, 0.0 -> bad annotation that we know is wrong
    version: str = "2.0"  # current version format

    def dict(self, **kwargs) -> dict:
        return self.model_dump(**kwargs)

    def model_dump(self, with_cls_field=False, **kwargs) -> dict:
        if with_cls_field:
            # Ugly workaround to support cls and init_args on the Geometry submodel.
            # We should get away from this pattern in the future and just use
            # models that are not ambiguous.
            init_args = super().model_dump(**kwargs)
            init_args["geometry"] = self.geometry.model_dump(with_cls_field=True)
            return {"cls": self.__class__.__name__, "init_args": init_args}
        return super().model_dump(**kwargs)

    @field_validator("version")
    @classmethod
    def update_version_and_warn(cls, v):
        CURRENT_VERSION = "2.0"
        return CURRENT_VERSION


# Function to send Image Query to GroundLight and label the image
def label_image(detector_id: str, image: Image, rois, answer: str):
    # Instantiate Groundlight client and get the detector
    if detector_id:
        gl = Groundlight()
        detector = gl.get_detector(detector_id)

    # Send image query to GroundLight
    iq = gl.ask_async(detector=detector, image=image, human_review="NEVER")

    # Sleep for a bit to give the GroundLight API time to process the image
    time.sleep(0.8)

    # Parse the rois to the format expected by GroundLight
    parsed_rois = []

    for roi in rois:
        bbox = roi["init_args"]["geometry"]["init_args"]
        label = roi["init_args"]["label"]
        score = roi["init_args"]["score"]
        version = roi["init_args"]["version"]

        top = bbox["top"]
        left = bbox["left"]
        right = bbox["right"]
        bottom = bbox["bottom"]

        parsed_rois.append(
            ROI(
                version="1.0",
                label=label,
                score=score,
                geometry=BBoxGeometry(
                    left=left, top=top, right=right, bottom=bottom, version=version
                ),
            )
        )

    parsed_rois = [
        parsed_roi.model_dump(with_cls_field=True) for parsed_roi in parsed_rois
    ]

    # Label the image with the label and rois
    url_string = "https://api.groundlight.ai/device-api/labels"
    token = os.getenv("GROUNDLIGHT_API_TOKEN")
    headers = {
        "x-api-token": f"{token}",
        "Content-Type": "application/json",
    }

    response = requests.request(
        method="POST",
        url=url_string,
        headers=headers,
        json={
            "label": answer,
            "posicheck_id": iq.id,
            "rois": parsed_rois,
            "annotations_requested": f"BINARY_CLASSIFICATION,BOUNDING_BOXES",
        },
    )

    if response.status_code != 200:
        logger.error(f"Failed to label image with status code: {response.status_code}")
        logger.error(response.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=str, help="Path to the frames", required=True)
    parser.add_argument(
        "--detector-id", type=str, help="ID of the detector", required=True
    )

    args = parser.parse_args()

    # Load the JSON file
    json_path = os.path.join(args.frames, "answer_block.json")
    with open(json_path) as f:
        data = json.load(f)

    # Iterate through the data with a progress bar
    for idx, image_id in tqdm(
        data["data"]["image_id"].items(), desc="Processing Images"
    ):
        rois = data["data"]["current_best_answer_rois"][idx]
        answer = data["data"]["current_best_answer_str"][idx]

        # Label class has a different name than the answer from the JSON file
        label = "PASS" if answer == "YES" else "FAIL"

        # Skip if there are no bounding boxes (null)
        if rois is None or len(rois) == 0:
            continue

        # Construct the image file name
        image_file = image_id.replace("/", "_") + ".jpg"
        image_path = os.path.join(args.frames, image_file)

        # Open the image
        if not os.path.exists(image_path):
            print(f"Image {image_file} not found.")
            continue
        image = Image.open(image_path)

        label_image(detector_id=args.detector_id, image=image, rois=rois, answer=label)

    print("Image successfully labeled")
