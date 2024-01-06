import os
import random
import torch
import safetensors.torch
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from main import (
    VisionGANDataset,
    Resnet_k,
    get_inverse_transform,
    get_checkpoint_epochs,
)


def visualize_results(
    dataset_path: str,
    num_photos: int = 5,
    save: bool = False,
    only_latest: bool = False,
    checkpoints_folder: str = "./checkpoints",
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


visualize_results("./horse2zebra", only_latest=True)
