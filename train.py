from dataclasses import dataclass
import os
import glob
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

from PIL import Image, ImageDraw


class TrainOptions:
    def __init__(self, tl, md, lf, opt, tb, log):
        self.training_loader = tl
        self.model = md
        self.loss_fn = lf
        self.optimizer = opt
        self.tb_writer = tb
        self.log = log


class TrainRig:
    def __init__(self, train_options):
        self.options = train_options

        self.model = self.options.model
        self.loss_fn = self.options.loss_fn
        self.training_loader = self.options.training_loader
        self.optimizer = self.options.optimizer
        self.log = self.options.log
        self.tb_writer = self.options.tb_writer

    def train(self, epochs):
        for epoch in range(epochs):
            self._train_one_epoch(epoch)

    def _train_one_epoch(self, epoch_index):
        batch_loss = 0.0
        for i, data in enumerate(self.training_loader):
            inputs, labels = data
            self.optimizer.zero_grad()

            # Compute the loss and its gradients
            loss = self.loss_fn(model(inputs), labels)
            loss.backward()
            # Adjust learning weights
            self.optimizer.step()

            batch_loss += loss
        if self.log:
            avg_epoch_loss = batch_loss / (i + 1)  # loss per batch
            print(f"Epoch {epoch_index} loss: {avg_epoch_loss}")
            if self.tb_writer is not None:
                tb_x = epoch_index * len(self.training_loader) + i + 1
                self.tb_writer.add_scalar("Loss/train", avg_batch_loss, tb_x)
            debug_loss = 0.0
        return batch_loss


def rename_images(path_glob="data/*bayer*.png", new_name="bayer"):
    # Rename images from data set.
    found_images = glob.glob(path_glob)

    for i, img_path in enumerate(found_images):
        img_data = None
        new_img_path = os.path.join(
            os.path.dirname(img_path), f"{new_name}_{i:06d}.png"
        )
        os.rename(img_path, new_img_path)


class TorusData(Dataset):
    def __init__(self, annotations, img_dir="data"):
        self.img_dir = img_dir
        self.annotations = list(annotations.values())

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_path = os.path.join("data", self.annotations[idx]["filename"])
        return LoadImage(image_path), LoadBox(self.annotations[idx]["regions"])


def LoadBox(annotation_region):
    if len(annotation_region) != 0:
        box = annotation_region[0]["shape_attributes"]
        box_data = torch.tensor(
            [box["x"], box["y"], box["width"], box["height"]],
            dtype=torch.float32,
        )
    else:
        box_data = torch.zeros(
            (4,),
            dtype=torch.float32,
        )
    return box_data


def LoadImage(img_path):
    with Image.open(img_path) as im:
        image_data = torch.tensor(np.array(im, dtype=np.float32).transpose(2, 1, 0))
    return image_data


class SingleTorus(nn.Module):
    def __init__(self):
        super(SingleTorus, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 57 * 89, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 57 * 89)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def Inspect(model, path_glob="data/*color*.png"):
    # Rename images from data set.
    found_images = glob.glob(path_glob)

    for i, img_path in enumerate(found_images):
        print(f"Viewing Image: {i}")
        out_box = model(LoadImage(img_path)).type(torch.int32)[0, :].numpy()
        x, y, w, h = out_box[:4]
        bounds = ((x, y), (x + w, y + h))
        if (out_box < 0).any():
            print(f"Skipping: {img_path} because bounds are negative.")
            continue

        with Image.open(img_path) as im:
            draw_handle = ImageDraw.Draw(im)
            draw_handle.rectangle(bounds, outline="red")
            im = im.rotate(180, Image.NEAREST, expand=1)
            im.save(img_path.replace("data", "output"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="train", description="trains basic torus detectors for fun", epilog="---"
    )

    parser.add_argument(
        "-t",
        "--task",
        required=True,
        choices=["train", "inspect"],
    )
    args = parser.parse_args()

    if args.task == "train":
        # Learning Rate
        alpha = 1e-4
        num_anchors = 1
        model = SingleTorus()
        loss = torch.nn.MSELoss()
        opt = torch.optim.Adam(model.parameters(), lr=alpha)

        json_path = "data/torus_ann.json"
        annotations = None
        with open(json_path) as js:
            annotations = json.load(js)
        tloader = DataLoader(TorusData(annotations), batch_size=20)

        to = TrainOptions(
            tloader,
            model,
            torch.nn.MSELoss(),
            opt,
            None,
            True,
        )

        tr = TrainRig(to)
        tr.train(100)

        model_path = "model.pt"
        torch.save(model.state_dict(), model_path)

    elif args.task == "inspect":
        st = SingleTorus()
        st.load_state_dict(torch.load("model.pt"))
        st.eval()
        Inspect(st)
    elif args.task == 'auto-annotation':
        # TODO(ttran): Support auto annotation feature
        pass
