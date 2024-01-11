import os
import argparse
import urllib.request
import zipfile
from typing import Literal

dataset_names = [
    "apple2orange",
    "summer2winter_yosemite",
    "horse2zebra",
    "monet2photo",
    "cezanne2photo",
    "ukiyoe2photo",
    "vangogh2photo",
    "maps",
    "facades",
    "iphone2dslr_flower",
    "ae_photos",
]
DatasetNames = Literal[
    "apple2orange",
    "summer2winter_yosemite",
    "horse2zebra",
    "monet2photo",
    "cezanne2photo",
    "ukiyoe2photo",
    "vangogh2photo",
    "maps",
    "facades",
    "iphone2dslr_flower",
    "ae_photos",
]


def progress(block_num: int, block_size: int, total_size: int):
    downloaded = block_num * block_size
    print(f"\r{downloaded}/{total_size} ({(downloaded/total_size) * 100:.2f}%)")


def download_dataset(name: DatasetNames):
    filename = f"{name}.zip"
    urllib.request.urlretrieve(
        f"http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/{filename}",
        filename,
        progress,
    )
    print("Unzipping...")
    with zipfile.ZipFile(filename, "r") as zf:
        zf.extractall()
    os.remove(filename)


def main():
    parser = argparse.ArgumentParser("CycleGAN dataset downloader")
    parser.add_argument(
        "-ds",
        "--dataset",
        dest="dataset_name",
        choices=dataset_names,
        help="name of the dataset to download",
        required=True,
    )
    args = parser.parse_args()
    download_dataset(args.dataset_name)


if __name__ == "__main__":
    main()
