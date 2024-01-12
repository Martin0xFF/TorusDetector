from dataclasses import dataclass
import os
import glob

import torch
import numpy as np
from PIL import Image


@dataclass
class TrainOptions:
    data_path = ""

    # with Image.open(img_path) as img:
    #     img_data = np.array(img)


def rename_images(path_glob="data/*bayer*.png"):
    # Rename images from data set.
    found_images = glob.glob(path_glob)

    for i, img_path in enumerate(found_images):
        img_data = None
        new_img_path = os.path.join(os.path.dirname(img_path), f"bayer_{i:06d}.png")
        os.rename(img_path, new_img_path)


def TorusData(path_glob=""):
    pass


def train():
    pass


if __name__ == "__main__":
    rename_images()
