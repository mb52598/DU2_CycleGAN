import os
import random
import safetensors.torch
import matplotlib
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from main import VisionGANDataset, Resnet_k, get_inverse_transform


def get_state_dict(checkpoint_file: str):
    return {
        k.lstrip("_orig_mod."): v
        for k, v in safetensors.torch.load_file(checkpoint_file).items()
    }


def visualize_results(dataset_path: str, checkpoints_folder: str = "./checkpoints"):
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

    for checkpoint_epoch in sorted(
        set(map(lambda x: x.rsplit("_", 1)[1], os.listdir(checkpoints_folder))), reverse=True, key=lambda x: int(x)
    ):
        index = random.randint(0, min(len(train_dataset_A), len(train_dataset_B)) - 1)
        real_B = train_dataset_B[index]
        fake_A = generator_A(real_B)
        real_A = train_dataset_A[index]
        fake_B = generator_B(real_A)

        generator_A.load_state_dict(
            get_state_dict(
                os.path.join(
                    checkpoints_folder,
                    "generator_A_" + checkpoint_epoch,
                )
            )
        )
        generator_B.load_state_dict(
            get_state_dict(
                os.path.join(
                    checkpoints_folder,
                    "generator_B_" + checkpoint_epoch,
                )
            )
        )

        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
        ax0.set_title("Real")
        ax1.set_title("Fake")

        fig.suptitle(checkpoint_epoch)
        ax0.imshow(inv_transform(real_B).permute(1, 2, 0))
        ax1.imshow(inv_transform(fake_A).permute(1, 2, 0))
        ax2.imshow(inv_transform(real_A).permute(1, 2, 0))
        ax3.imshow(inv_transform(fake_B).permute(1, 2, 0))

        plt.show()


visualize_results("./horse2zebra")
