import os
import random
import argparse
import torch
import safetensors.torch
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from download_dataset import DatasetNames, dataset_names
from main import (
    VisionGANDataset,
    Resnet_k,
    get_inverse_transform,
    get_checkpoint_epochs,
)


def visualize_results(
    dataset_path: DatasetNames,
    checkpoints_folder: str = "./checkpoints",
    num_photos: int = 5,
    grid: bool = False,
    save: bool = False,
    only_latest: bool = False,
):
    train_dataset_A = VisionGANDataset(os.path.join(dataset_path, "trainA"))
    train_dataset_B = VisionGANDataset(os.path.join(dataset_path, "trainB"))

    generator_A = Resnet_k(9)
    generator_B = Resnet_k(9)

    inv_transform = get_inverse_transform()

    fig: Figure
    ax0: plt.Axes
    ax1: plt.Axes
    ax2: plt.Axes
    ax3: plt.Axes

    with torch.no_grad():
        for checkpoint_epoch in get_checkpoint_epochs(checkpoints_folder):
            safetensors.torch.load_model(
                generator_A,
                os.path.join(
                    checkpoints_folder,
                    "generator_A_" + checkpoint_epoch,
                ),
            )
            safetensors.torch.load_model(
                generator_B,
                os.path.join(
                    checkpoints_folder,
                    "generator_B_" + checkpoint_epoch,
                ),
            )
            if grid:
                reals_B = torch.stack([random.choice(train_dataset_B) for _ in range(num_photos)], dim=0)
                fakes_A = generator_A(reals_B)
                reals_A = torch.stack([random.choice(train_dataset_A) for _ in range(num_photos)], dim=0)
                fakes_B = generator_B(reals_A)

                grid_B = make_grid(inv_transform(torch.concat((reals_B, fakes_A), dim=0)), nrow=num_photos)
                grid_A = make_grid(inv_transform(torch.concat((reals_A, fakes_B), dim=0)), nrow=num_photos)

                fig, (ax0, ax1) = plt.subplots(
                    2, figsize=(15, 8), constrained_layout=True
                )
                fig.suptitle("Epoch: " + checkpoint_epoch)

                ax0.imshow(grid_B.permute(1, 2, 0))
                ax1.imshow(grid_A.permute(1, 2, 0))

                if save:
                    plt.savefig(checkpoint_epoch + "_grid.png")
                else:
                    plt.show()
            else:
                for i in range(num_photos):
                    real_B = random.choice(train_dataset_B)
                    fake_A = generator_A(real_B)
                    real_A = random.choice(train_dataset_A)
                    fake_B = generator_B(real_A)

                    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(
                        2, 2, figsize=(15, 8), constrained_layout=True
                    )
                    ax0.set_title("Real")
                    ax1.set_title("Fake")

                    fig.suptitle("Epoch: " + checkpoint_epoch)
                    ax0.imshow(inv_transform(real_B).permute(1, 2, 0))
                    ax1.imshow(inv_transform(fake_A).permute(1, 2, 0))
                    ax2.imshow(inv_transform(real_A).permute(1, 2, 0))
                    ax3.imshow(inv_transform(fake_B).permute(1, 2, 0))

                    if save:
                        plt.savefig(checkpoint_epoch + "_" + str(i) + ".png")
                    else:
                        plt.show()

            if only_latest:
                break


def main():
    parser = argparse.ArgumentParser("CycleGAN visualizer")
    parser.add_argument(
        "-ds",
        "--dataset",
        dest="dataset_name",
        choices=dataset_names,
        help="name of the dataset to use for visualization",
        required=True,
    )
    parser.add_argument(
        "-chd",
        "--checkpoint-dir",
        dest="checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="from where to load the model checkpoints",
    )
    parser.add_argument(
        "-np",
        "--num-photos",
        dest="num_photos",
        type=int,
        default=5,
        help="how many photos to display",
    )
    parser.add_argument(
        "-ol",
        "--only-latest",
        dest="only_latest",
        action="store_true",
        help="whether to display only the latest epoch",
    )
    parser.add_argument(
        "-g",
        "--grid",
        dest="grid",
        action="store_true",
        help="whether to show photos as a grid",
    )
    parser.add_argument(
        "-sp",
        "--save-photos",
        dest="save_photos",
        action="store_true",
        help="whether to save the photos instead of displaying",
    )
    args = parser.parse_args()
    visualize_results(
        args.dataset_name,
        args.checkpoint_dir,
        args.num_photos,
        args.grid,
        args.save_photos,
        args.only_latest,
    )


if __name__ == "__main__":
    main()
