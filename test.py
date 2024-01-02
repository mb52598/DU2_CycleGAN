import random
import safetensors.torch
import torchvision
from main import VisionGANDataset, Resnet_k

train_dataset_A = VisionGANDataset('./horse2zebra/trainA')
train_dataset_B = VisionGANDataset('./horse2zebra/trainB')

generator_A = Resnet_k(9)
generator_B = Resnet_k(9)

generator_A.load_state_dict({k.lstrip('_orig_mod.'): v for k, v in safetensors.torch.load_file('./checkpoints/generator_A_20.safetensors').items()})
generator_B.load_state_dict({k.lstrip('_orig_mod.'): v for k, v in safetensors.torch.load_file('./checkpoints/generator_B_20.safetensors').items()})

index = random.randint(0, min(len(train_dataset_A), len(train_dataset_B)) - 1)

torchvision.io.write_png(train_dataset_A.inverse_transform(train_dataset_B[index]), 'urimg_A.png')
torchvision.io.write_png(train_dataset_A.inverse_transform(generator_A(train_dataset_B[index])), 'myimg_A.png')
torchvision.io.write_png(train_dataset_B.inverse_transform(train_dataset_A[index]), 'urimg_B.png')
torchvision.io.write_png(train_dataset_B.inverse_transform(generator_B(train_dataset_A[index])), 'myimg_B.png')
