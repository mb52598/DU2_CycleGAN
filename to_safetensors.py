import os
import torchvision
import safetensors.torch


def to_safetensors(directory: str, filename: str):
    safetensors.torch.save_file(
        {
            file: torchvision.io.read_image(
                os.path.join(directory, file), torchvision.io.ImageReadMode.RGB
            )
            for file in os.listdir(directory)
        },
        filename,
    )


to_safetensors("./horse2zebra/trainA", "train_A.safetensors")
to_safetensors("./horse2zebra/trainB", "train_B.safetensors")
