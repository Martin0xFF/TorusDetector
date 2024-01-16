import json
import argparse
import glob
import random
import torch

from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

from PIL import Image, ImageDraw

from unet import UNet
from train import TorusAutoData, TrainOptions, TrainRig, LoadImage


class TorusAutoEncoder(nn.Module):
    def __init__(self):
        super(TorusAutoEncoder, self).__init__()
        layers = 3

        # Encoder
        en_modules = []
        en_modules.append(
            nn.Sequential(
                nn.Conv2d(3, 16, 7),
                nn.BatchNorm2d(16),
                nn.ReLU(),
            )
        )

        for i in range(layers):
            conv_block = nn.Sequential(
                nn.Conv2d(16, 16, 3),
                nn.BatchNorm2d(16),
                nn.ReLU(),
            )
            en_modules.append(conv_block)

        # Decoder
        de_modules = []

        for i in range(layers):
            convt_block = nn.Sequential(
                nn.ConvTranspose2d(16, 16, 3),
                nn.BatchNorm2d(16),
                nn.ReLU(),
            )
            de_modules.append(convt_block)

        de_modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(16, 3, 7),
                nn.BatchNorm2d(3),
                nn.ReLU(),
            )
        )

        self.encoder = nn.Sequential(*en_modules)
        self.decoder = nn.Sequential(*de_modules)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def InspectAutoEncoder(model, path_glob="data/*color*.png"):
    found_images = glob.glob(path_glob)

    for i, img_path in enumerate(found_images):
        print(f"Viewing Image: {i}")
        out_im = (255 * model(LoadImage(img_path)[None, ...])).type(torch.uint8).numpy()
        out_im = out_im[0, ...].transpose([2, 1, 0])
        im = Image.fromarray(out_im).save(img_path.replace("data", "output"))


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

    parser.add_argument("-e", "--epoch", type=int, default=10)

    parser.add_argument("-m", "--model-name", type=str, default="train_model.pt")

    parser.add_argument(
        "-d", "--device-name", type=str, default="cpu", choices=["cpu", "mps", "cuda"]
    )

    args = parser.parse_args()

    if args.task == "train":
        random.seed(2702)
        torch.manual_seed(2702)

        # Learning Rate
        alpha = 1e-3
        model = TorusAutoEncoder()

        loss = torch.nn.MSELoss()
        opt = torch.optim.Adam(model.parameters(), lr=alpha)

        json_path = "data/torus_ann.json"
        annotations = None
        with open(json_path) as js:
            annotations = list(json.load(js).items())

        random.shuffle(annotations)

        training_loader = DataLoader(TorusAutoData("field_images"), batch_size=50)
        print(f"Training Dataset Batch Size: {len(training_loader)}")

        to = TrainOptions(
            training_loader,
            None,
            model,
            loss,
            opt,
            None,
            True,
            "mps",
        )

        tr = TrainRig(to)
        tr.train(args.epoch)

    elif args.task == "inspect":
        tae = TorusAutoEncoder()
        tae.load_state_dict(torch.load(args.model_name))
        tae.eval()
        InspectAutoEncoder(tae)
