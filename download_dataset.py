import os
import urllib.request
import zipfile
from typing import Literal

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


download_dataset("horse2zebra")
