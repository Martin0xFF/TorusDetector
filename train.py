from dataclasses import dataclass
import os
import glob
import json
import argparse
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

from PIL import Image, ImageDraw


class TrainOptions:
    def __init__(self, train, test, md, lf, opt, tb, log, device='cpu'):
        self.training_loader = train
        self.testing_loader = test
        self.model = md
        self.loss_fn = lf
        self.optimizer = opt
        self.tb_writer = tb
        self.log = log
        self.device = device


class TrainRig:
    def __init__(self, train_options):
        self.options = train_options

        self.model = self.options.model
        self.loss_fn = self.options.loss_fn
        self.training_loader = self.options.training_loader
        self.testing_loader = self.options.testing_loader
        self.optimizer = self.options.optimizer
        self.log = self.options.log
        self.tb_writer = self.options.tb_writer
        self.device = self.options.device

        self.model.to(self.device)

    def train(self, epochs):
        best_train_loss = np.inf
        best_test_loss = np.inf
        for epoch in range(epochs):
            print(f"\nEpoch number: {epoch} ------------------")
            cur_train_loss = self._train_one_epoch(epoch)
            if cur_train_loss < best_train_loss:
                print("New Smallest Train Loss found, dumping model")
                best_train_loss = cur_train_loss
                torch.save(self.model.state_dict(), "train_model.pt")


            if self.testing_loader is not None:
                cur_test_loss = self.test()
                if cur_test_loss < best_test_loss:
                    print("New Smallest Test Loss found, dumping model")
                    best_test_loss = cur_test_loss
                    torch.save(self.model.state_dict(), "test_model.pt")

    def _train_one_epoch(self, epoch_index):
        self.model.train()
        batch_loss = 0.0
        for i, data in enumerate(self.training_loader):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            self.optimizer.zero_grad()
            # Compute the loss and its gradients
            loss = self.loss_fn(self.model(inputs), labels)
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

    def test(self):
        self.model.eval()
        batch_loss = 0.0
        for i, data in enumerate(self.testing_loader):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            # Compute the loss and its gradients
            loss = self.loss_fn(self.model(inputs), labels)

            batch_loss += loss
        if self.log:
            avg_loss = batch_loss / (i + 1)  # loss per batch
            print(f"Test loss: {avg_loss}")
        return batch_loss


def rename_images(path_glob="data/*bayer*.png", new_name="bayer", start_index=0):
    # Rename images from data set.
    found_images = glob.glob(path_glob)

    for i, img_path in enumerate(found_images):
        img_data = None
        new_img_path = os.path.join(
            os.path.dirname(img_path), f"{new_name}_{i+start_index:06d}.png"
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

class TorusAutoData(Dataset):
    def __init__(self, img_dir="field_images"):
        self.img_dir = img_dir
        self.image_paths = glob.glob(f"{img_dir}/*.jpg") + glob.glob(f"{img_dir}/*.png")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # h,w - 240,368 from pi camera
        resize_fn = transforms.Resize((240, 368), antialias=True)
        img_data = resize_fn(LoadImage(image_path))
        img_data = img_data[:3, ...]
        return img_data, img_data

def LoadBox(annotation_region):
    box_data = torch.zeros(5, 5)
    for i, region in enumerate(annotation_region):
        box = region["shape_attributes"]
        box_data[i, :] = torch.tensor(
            [box["x"], box["y"], box["width"], box["height"], 1],
            dtype=torch.float32,
        )
        if i > 4:
            break
    return box_data


def LoadImage(img_path):
    with Image.open(img_path) as im:
        return transforms.functional.to_tensor(im)
        # image_data = torch.tensor(np.array(im, dtype=np.float32).transpose(2, 1, 0))
    # return image_data

class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors: List[Tuple[int, int]], num_classes: int, new_coords: bool):
        """
        Create a YOLO layer

        :param anchors: List of anchors
        :param num_classes: Number of classes
        :param new_coords: Whether to use the new coordinate format from YOLO V7
        """
        super(YOLOLayer, self).__init__()
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.new_coords = new_coords
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.no = num_classes + 5  # number of outputs per anchor
        self.grid = torch.zeros(1)  # TODO

        anchors = torch.tensor(list(chain(*anchors))).float().view(-1, 2)
        self.register_buffer('anchors', anchors)
        self.register_buffer(
            'anchor_grid', anchors.clone().view(1, -1, 1, 1, 2))
        self.stride = None

    def forward(self, x: torch.Tensor, img_size: int) -> torch.Tensor:
        """
        Forward pass of the YOLO layer

        :param x: Input tensor
        :param img_size: Size of the input image
        """
        stride = img_size // x.size(2)
        self.stride = stride
        bs, _, ny, nx = x.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        x = x.view(bs, self.num_anchors, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        if not self.training:  # inference
            if self.grid.shape[2:4] != x.shape[2:4]:
                self.grid = self._make_grid(nx, ny).to(x.device)

            x[..., 0:2] = (x[..., 0:2].sigmoid() + self.grid) * stride  # xy
            x[..., 2:4] = torch.exp(x[..., 2:4]) * self.anchor_grid # wh
            x[..., 4:] = x[..., 4:].sigmoid() # conf, cls
            x = x.view(bs, -1, self.no)

        return x


class SingleTorus(nn.Module):
    def __init__(self):
        super(SingleTorus, self).__init__()

        self.dropout = nn.Dropout(0.1)

        self.conv1 = nn.Conv2d(3, 16, 3)
        self.batch_norm1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 3)
        self.batch_norm2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 30, 3)
        self.batch_norm3 = nn.BatchNorm2d(30)

        self.fc1 = nn.Linear(6 * 362 * 234, 16)
        self.fc2 = nn.Linear(16, 5)
        self.norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def forward(self, x):
        x = self.norm(x)
        x = self.batch_norm1(F.relu(self.dropout(self.conv1(x))))
        x = self.batch_norm2(F.relu(self.dropout(self.conv2(x))))
        x = self.batch_norm3(F.relu(self.dropout(self.conv3(x))))
        x = x.view(-1, 5, 6 * 362 * 234)
        x = F.relu(self.dropout(self.fc1(x)))
        x = self.fc2(x)
        return x


def Inspect(model, path_glob="data/*color*.png"):
    found_images = glob.glob(path_glob)

    for i, img_path in enumerate(found_images):
        print(f"Viewing Image: {i}")
        out_box = model(LoadImage(img_path)[None,...]).type(torch.int32)[0, :].numpy()
        with Image.open(img_path) as im:
            for i in range(5):
                x, y, w, h = out_box[i, :4]
                bounds = ((x, y), (x + w, y + h))
                if w <= 0 or h <= 0:
                    continue
                print(bounds)

                if (out_box[i, :4] < 0).any():
                    print(f"Skipping: {img_path} because bounds are negative.")
                    continue

                draw_handle = ImageDraw.Draw(im)
                draw_handle.rectangle(bounds, outline="red")
            im = im.rotate(180, Image.NEAREST, expand=1)
            im.save(img_path.replace("data", "output"))


def AutoAnnotate(model, path_glob="data/*color*.png"):
    found_images = glob.glob(path_glob)
    boxes = {}

    for i, img_path in enumerate(found_images):
        print(f"Viewing Image: {i}")
        out_box = model(LoadImage(img_path)).type(torch.int32)[0, :].numpy()
        x, y, w, h = out_box[:4]

        if (out_box < 0).any():
            print(f"Skipping: {img_path} because bounds are negative.")
            boxes[img_path] = [-1, -1, -1, -1]
            continue
        else:
            boxes[img_path] = [int(l) for l in out_box]

    with open("output/boxes.json", "w") as file:
        file.write(json.dumps(boxes))


def Review(path_glob="data/*color*.png"):
    boxes = {}

    with open("output/boxes.json", "r") as file:
        boxes = json.load(file)

    new_boxes = {}
    for img_path, box in boxes.items():
        if box[0] < 0:
            print(f"Skipping: {img_path} because bounds are negative.")
        else:
            with Image.open(img_path) as im:
                draw_handle = ImageDraw.Draw(im)
                draw_handle.rectangle(
                    ((box[0], box[1]), (box[0] + box[2], box[1] + box[3])),
                    outline="red",
                )
                im = im.rotate(180, Image.NEAREST, expand=1)
                im.show()
                i = input("Keep? [Y/n]")

                if i == "Y" or i == "":
                    im.save(img_path.replace("data", "output"))
                    new_boxes[img_path] = box

    with open("output/boxes.json", "w") as file:
        file.write(json.dumps(new_boxes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="train", description="trains basic torus detectors for fun", epilog="---"
    )

    parser.add_argument(
        "-t",
        "--task",
        required=True,
        choices=["train", "inspect", "auto-annotation", "review", "rename"],
    )

    parser.add_argument(
        "-e",
        "--epoch",
        type=int,
        default=10
    )

    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        default="test_model.pt"
    )

    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        choices = ["mps", "cuda", "cpu"]
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
            annotations = list(json.load(js).items())

        random.seed(2702)
        torch.manual_seed(2702)
        random.shuffle(annotations)

        train_ratio = int(0.9 * len(annotations))
        training_loader = DataLoader(
            TorusData(dict(annotations[:train_ratio])), batch_size=train_ratio
        )
        print(f"Training Dataset Batch Size: {len(training_loader)}")

        testing_loader = DataLoader(
            TorusData(dict(annotations[train_ratio:])), batch_size=10
        )
        print(f"Test Dataset Batch Size: {len(testing_loader)}")

        to = TrainOptions(
            training_loader,
            testing_loader,
            model,
            loss,
            opt,
            None,
            True,
            args.device
        )

        tr = TrainRig(to)
        tr.train(args.epoch)

    elif args.task == "inspect":
        st = SingleTorus()
        st.load_state_dict(torch.load(args.model_name))
        st.eval()
        Inspect(st)

    elif args.task == "auto-annotation":
        st = SingleTorus()
        st.load_state_dict(torch.load("model.pt"))
        st.eval()
        AutoAnnotate(st)

    elif args.task == "review":
        Review()

    elif args.task == "rename":
        rename_images(
            path_glob="new_data/*color*.png",
            new_name="color",
            start_index=43,
        )
