import os
import argparse
import torchvision
import safetensors.torch
from download_dataset import DatasetNames, dataset_names


def to_safetensors(dataset_name: DatasetNames, destination_folder: str = "./"):
    for subdirectory in ("trainA", "trainB"):
        directory = os.path.join(dataset_name, subdirectory)
        safetensors.torch.save_file(
            {
                file: torchvision.io.read_image(
                    os.path.join(directory, file), torchvision.io.ImageReadMode.RGB
                )
                for file in os.listdir(directory)
            },
            os.path.join(destination_folder, subdirectory + ".safetensors"),
        )


def main():
    parser = argparse.ArgumentParser("CycleGAN dataset to safetensors")
    parser.add_argument(
        "-ds",
        "--dataset",
        dest="dataset_name",
        choices=dataset_names,
        help="name of the dataset to use for converting to savetensors",
        required=True,
    )
    parser.add_argument(
        "-dst",
        "--destination",
        dest="destination_folder",
        type=str,
        default="./",
        help="where to save the safetensors",
    )
    args = parser.parse_args()
    to_safetensors(args.dataset_name, args.destination_folder)


if __name__ == "__main__":
    main()
