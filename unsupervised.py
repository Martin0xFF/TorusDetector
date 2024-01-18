import json
import argparse
import glob
import random
import torch
import math

from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

from PIL import Image, ImageDraw

from train import TorusAutoData, TrainOptions, TrainRig, LoadImage


def round_repeats(repeats, multiplier):
    """Round number of filters based on depth multiplier."""
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


class MBConvBlock(nn.Module):
    def __init__(self, inp, final_oup, k, s, expand_ratio, se_ratio, has_se=False):
        super(MBConvBlock, self).__init__()

        self._momentum = 0.01
        self._epsilon = 1e-3
        self.input_filters = inp
        self.output_filters = final_oup
        self.stride = s
        self.expand_ratio = expand_ratio
        self.has_se = has_se
        self.id_skip = True  # skip connection and drop connect

        # Expansion phase
        oup = inp * expand_ratio  # number of output channels
        if expand_ratio != 1:
            self._expand_conv = nn.Conv2d(
                in_channels=inp, out_channels=oup, kernel_size=1, bias=False
            )
            self._bn0 = nn.BatchNorm2d(
                num_features=oup, momentum=self._momentum, eps=self._epsilon
            )

        # Depthwise convolution phase
        self._depthwise_conv = nn.Conv2d(
            in_channels=oup,
            out_channels=oup,
            groups=oup,  # groups makes it depthwise
            kernel_size=k,
            padding=(k - 1) // 2,
            stride=s,
            bias=False,
        )
        self._bn1 = nn.BatchNorm2d(
            num_features=oup, momentum=self._momentum, eps=self._epsilon
        )

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(inp * se_ratio))
            self._se_reduce = nn.Conv2d(
                in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1
            )
            self._se_expand = nn.Conv2d(
                in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1
            )

        # Output phase
        self._project_conv = nn.Conv2d(
            in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False
        )
        self._bn2 = nn.BatchNorm2d(
            num_features=final_oup, momentum=self._momentum, eps=self._epsilon
        )
        self._relu = nn.ReLU6(inplace=True)

    def forward(self, x, drop_connect_rate=None):
        """
        :param x: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        identity = x
        if self.expand_ratio != 1:
            x = self._relu(self._bn0(self._expand_conv(x)))
        x = self._relu(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._relu(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        if (
            self.id_skip
            and self.stride == 1
            and self.input_filters == self.output_filters
        ):
            # if drop_connect_rate:
            #    x = drop_connect(x, drop_connect_rate, training=self.training)
            x += identity  # skip connection
        return x


def round_filters(filters, multiplier, divisor=8, min_width=None):
    """Calculate and round number of filters based on width multiplier."""
    if not multiplier:
        return filters
    filters *= multiplier
    min_width = min_width or divisor
    new_filters = max(min_width, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


class TorusAutoEncoder(nn.Module):
    def __init__(self):
        super(TorusAutoEncoder, self).__init__()

        widthi_multiplier = 1.0
        depth_multiplier = 1.0

        momentum = 0.01
        epsilon = 1e-3

        mb_block_settings = [
            # repeat|kernal_size|stride|expand|input|output|se_ratio
            [1, 3, 1, 1, 32, 16, 0.25],
            [2, 3, 2, 6, 16, 24, 0.25],
            # [2, 5, 2, 6, 24, 40, 0.25],
            # [3, 3, 2, 6, 40, 80, 0.25],
            # [3, 5, 1, 6, 80, 112, 0.25],
            # [4, 5, 2, 6, 112, 192, 0.25],
            # [1, 3, 1, 6, 192, 320, 0.25],
        ]

        # Encoder
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=momentum, eps=epsilon),
            nn.ReLU6(inplace=True),
        )

        self.blocks = nn.ModuleList([])
        for i, stage_setting in enumerate(mb_block_settings):
            stage = nn.ModuleList([])
            (
                num_repeat,
                kernal_size,
                stride,
                expand_ratio,
                input_filters,
                output_filters,
                se_ratio,
            ) = stage_setting
            # Update block input and output filters based on width multiplier.
            input_filters = (
                input_filters
                if i == 0
                else round_filters(input_filters, widthi_multiplier)
            )
            output_filters = round_filters(output_filters, widthi_multiplier)
            num_repeat = (
                num_repeat
                if i == 0 or i == len(mb_block_settings) - 1
                else round_repeats(num_repeat, depth_multiplier)
            )

            # The first block needs to take care of stride and filter size increase.
            stage.append(
                MBConvBlock(
                    input_filters,
                    output_filters,
                    kernal_size,
                    stride,
                    expand_ratio,
                    se_ratio,
                    has_se=False,
                )
            )
            if num_repeat > 1:
                input_filters = output_filters
                stride = 1
            for _ in range(num_repeat - 1):
                stage.append(
                    MBConvBlock(
                        input_filters,
                        output_filters,
                        kernal_size,
                        stride,
                        expand_ratio,
                        se_ratio,
                        has_se=False,
                    )
                )

            self.blocks.append(stage)

        self.feature_encoder = nn.Sequential(
            nn.Linear(24 * 60 * 92, 64), nn.ReLU6(), nn.Linear(64, 64)
        )

        # Decoder
        de_modules = []

        self.feature_decoder = nn.Sequential(nn.ReLU6(), nn.Linear(64, 24 * 60 * 92))

        de_modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(24, 24, 3, stride=2),
                nn.BatchNorm2d(24),
                nn.ReLU6(),
            )
        )

        de_modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(24, 24, 3, stride=2, padding=1),
                nn.BatchNorm2d(24),
                nn.ReLU6(),
            )
        )

        de_modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(24, 24, 2, stride=1, padding=1),
                nn.BatchNorm2d(24),
                nn.ReLU6(),
            )
        )

        de_modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(24, 3, 3, stride=1, padding=1),
            )
        )
        self.decoder = nn.Sequential(*de_modules)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.blocks:
            for block in stage:
                x = block(x)

        x = x.reshape(-1, 24 * 60 * 92)
        x = self.feature_encoder(x)
        x = self.feature_decoder(x)
        x = x.reshape(-1, 24, 60, 92)
        x = self.decoder(x)
        return x


def InspectAutoEncoder(model, path_glob="data/*color*.png", max_num_images=0):
    found_images = glob.glob(path_glob)

    for i, img_path in enumerate(found_images):
        print(f"Viewing Image: {i}")
        x = LoadImage(img_path)
        resize_fn = transforms.Resize((240, 368), antialias=True)
        in_im = resize_fn(LoadImage(img_path)[None, ...])
        out_im = (255 * model(in_im)).type(torch.uint8).numpy()

        out_im = out_im[0, ...].transpose([1, 2, 0])
        in_im = (255 * in_im[0, ...]).type(torch.uint8).numpy().transpose([1, 2, 0])

        i_im = Image.fromarray(in_im)
        im = Image.fromarray(out_im)

        dst = Image.new("RGB", (i_im.width + im.width, im.height))
        dst.paste(i_im, (0, 0))
        dst.paste(im, (im.width, 0))
        dst.save("output/" + img_path.split("/")[-1])

        if max_num_images and max_num_images == i:
            break


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
    parser.add_argument("--cont", default=False, action="store_true")

    args = parser.parse_args()

    if args.task == "train":
        random.seed(2702)
        torch.manual_seed(2702)

        # Learning Rate
        alpha = 1e-3
        model = TorusAutoEncoder()
        if args.cont == True:
            print(f"picking up from previous model : {args.model_name} ")
            model.load_state_dict(
                torch.load(args.model_name, map_location=torch.device(args.device_name))
            )

        loss = torch.nn.MSELoss()
        opt = torch.optim.Adam(model.parameters(), lr=alpha)

        json_path = "data/torus_ann.json"
        annotations = None
        with open(json_path) as js:
            annotations = list(json.load(js).items())

        random.shuffle(annotations)

        training_loader = DataLoader(
            TorusAutoData("/home/mabel/Projects/detect/data/coco/train2017"),
            batch_size=75,
        )
        print(f"Training Dataset Batch Size: {len(training_loader)}")

        to = TrainOptions(
            training_loader,
            None,
            model,
            loss,
            opt,
            None,
            True,
            "cuda",
        )

        tr = TrainRig(to)
        tr.train(args.epoch)

    elif args.task == "inspect":
        tae = TorusAutoEncoder()
        tae.load_state_dict(torch.load(args.model_name, map_location=torch.device(args.device_name)))
        tae.eval()
        InspectAutoEncoder(
            tae,
            "field_images/*.jpg",
            max_num_images=50
            # tae, "data/*color*.png", max_num_images=20
        )
