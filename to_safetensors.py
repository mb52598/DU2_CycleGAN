import os
import torchvision
import safetensors.torch
from main import get_transform


def to_safetensors(directory: str, filename: str):
    transform = get_transform()
    safetensors.torch.save_file(
        {
            file: transform(
                torchvision.io.read_image(
                    os.path.join(directory, file), torchvision.io.ImageReadMode.RGB
                )
            ).contiguous()
            for file in os.listdir(directory)
        },
        filename,
    )


to_safetensors("./horse2zebra/trainA", "train_A.safetensors")
to_safetensors("./horse2zebra/trainB", "train_B.safetensors")
