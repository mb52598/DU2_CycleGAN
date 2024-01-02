import os
import sys
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.transforms import v2
from accelerate import Accelerator
from typing import Any, cast


class VisionGANDataset(Dataset[torch.Tensor]):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    mean_tensor = torch.tensor(mean).reshape(3, 1, 1)
    std_tensor = torch.tensor(std).reshape(3, 1, 1)
    images: list[str]

    def __init__(self, images_folder: str):
        super().__init__()
        self.images = list(
            map(lambda x: os.path.join(images_folder, x), os.listdir(images_folder))
        )
        
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.get_default_dtype(), scale=True),
                v2.Normalize(self.mean, self.std),
            ]
        )
        self.inverse_transform = v2.Compose(
            [v2.Lambda(self._inverse), v2.ToDtype(torch.uint8, scale=True)]
        )

    def _inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std_tensor + self.mean_tensor

    def __len__(self) -> int:
        return len(self.images)

    def getImage(self, index: int) -> torch.Tensor:
        return torchvision.io.read_image(
            self.images[index], torchvision.io.ImageReadMode.RGB
        )

    def __getitem__(self, index: int) -> torch.Tensor:
        image = self.getImage(index)
        transformed_image = self.transform(image)
        return transformed_image


class ImageBuffer:
    tensors: list[torch.Tensor]
    size: int

    def __init__(self, size: int):
        self.tensors = []
        self.size = size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.tensors.append(x)
        if len(self.tensors) > self.size:
            self.tensors.pop(0)
        # if random.getrandbits(1):
        #     result = self.tensors.copy()
        #     result[-1] = self.tensors[random.randint(0, len(self.tensors) - 2)]
        #     return torch.stack(result)
        return torch.concat(self.tensors)


class ImageBufferFast(nn.Module):
    tensors: torch.Tensor

    def __init__(self, buffer_size: int, *size: int):
        super().__init__()
        self.register_buffer("tensors", torch.zeros(buffer_size, *size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.tensors = torch.roll(self.tensors, shifts=-1, dims=0)
        self.tensors[-1] = x
        return self.tensors


class ImageBufferUltraFast(nn.Module):
    tensors: torch.Tensor

    def __init__(self, buffer_size: int, *size: int):
        super().__init__()
        self.register_buffer("tensors", torch.zeros(buffer_size, *size))
        self.index = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.tensors[self.index] = x
        self.index += 1
        self.index %= self.tensors.size(0)
        return self.tensors


def Resnet_c7s1_k(in_k: int, out_k: int, output_transform: nn.Module):
    return nn.Sequential(
        nn.Conv2d(
            in_k, out_k, kernel_size=7, stride=1, padding=3, padding_mode="reflect"
        ),
        nn.InstanceNorm2d(out_k),
        output_transform,
    )


def Resnet_dk(in_k: int, out_k: int):
    return nn.Sequential(
        nn.Conv2d(in_k, out_k, kernel_size=3, stride=2),
        nn.InstanceNorm2d(out_k),
        nn.ReLU(inplace=True),
    )


class Resnet_Rk(nn.Module):
    def __init__(self, in_k: int, out_k: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_k, out_k, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_k, out_k, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr = self.model(x)
        return xr + x  # Residual connection


def Resnet_uk(in_k: int, out_k: int, output_padding: int = 0):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_k, out_k, kernel_size=3, stride=2, output_padding=output_padding
        ),
        nn.InstanceNorm2d(out_k),
        nn.ReLU(inplace=True),
    )


def Resnet_k(k: int):
    return nn.Sequential(
        Resnet_c7s1_k(3, 64, output_transform=nn.ReLU(inplace=True)),
        Resnet_dk(64, 128),
        Resnet_dk(128, 256),
        *(Resnet_Rk(256, 256) for _ in range(k)),
        Resnet_uk(256, 128),
        Resnet_uk(128, 64, output_padding=1),
        Resnet_c7s1_k(64, 3, output_transform=nn.Tanh()),
    )


def UNet_encoder(in_k: int, out_k: int, batch_norm: bool = True):
    return nn.Sequential(
        nn.Conv2d(in_k, out_k, kernel_size=4, stride=2),
        *((nn.BatchNorm2d(out_k),) if batch_norm else ()),
        nn.LeakyReLU(0.2, inplace=True),
    )


def UNet_decoder(
    in_k: int,
    out_k: int,
    dropout: bool = True,
    output_padding: int = 0,
    output_transform: nn.Module = nn.ReLU(inplace=True),
):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_k, out_k, kernel_size=4, stride=2, output_padding=output_padding
        ),
        nn.BatchNorm2d(out_k),
        *((nn.Dropout(0.5),) if dropout else ()),
        output_transform,
    )


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = UNet_encoder(3, 64, batch_norm=False)
        self.e2 = UNet_encoder(64, 128)
        self.e3 = UNet_encoder(128, 256)
        self.e4 = UNet_encoder(256, 512)
        self.e5 = UNet_encoder(512, 512)
        self.e6 = UNet_encoder(512, 512)
        self.e7 = UNet_encoder(512, 512)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding="same"),
            nn.ReLU(),
        )
        self.d1 = UNet_decoder(1024, 512)
        self.d2 = UNet_decoder(1024, 512)
        self.d3 = UNet_decoder(1024, 512)
        self.d4 = UNet_decoder(1024, 256, dropout=False)
        self.d5 = UNet_decoder(512, 128, dropout=False)
        self.d6 = UNet_decoder(256, 64, dropout=False)
        self.d7 = UNet_decoder(128, 3, dropout=False, output_transform=nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xe1 = self.e1(x)
        xe2 = self.e2(xe1)
        xe3 = self.e3(xe2)
        xe4 = self.e4(xe3)
        xe5 = self.e5(xe4)
        xe6 = self.e6(xe5)
        xe7 = self.e7(xe6)
        xb = self.bottleneck(xe7)
        xd1 = self.d1(torch.concat((xb, xe7), dim=-3))
        xd2 = self.d2(torch.concat((xd1, xe6), dim=-3))
        xd3 = self.d3(torch.concat((xd2, xe5), dim=-3))
        xd4 = self.d4(torch.concat((xd3, xe4), dim=-3))
        xd5 = self.d5(torch.concat((xd4, xe3), dim=-3))
        xd6 = self.d6(torch.concat((xd5, xe2), dim=-3))
        xd7 = self.d7(torch.concat((xd6, xe1), dim=-3))
        return xd7


def PatchGAN_Ck(in_k: int, out_k: int, instance_norm: bool = True):
    return nn.Sequential(
        nn.Conv2d(in_k, out_k, kernel_size=4, stride=2),
        *((nn.InstanceNorm2d(out_k),) if instance_norm else ()),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )


def PatchGAN():
    return nn.Sequential(
        PatchGAN_Ck(3, 64, instance_norm=False),
        PatchGAN_Ck(64, 128),
        PatchGAN_Ck(128, 256),
        PatchGAN_Ck(256, 512),
        nn.Conv2d(512, 1, kernel_size=4, stride=2),
    )


def PixelGAN():
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=1, stride=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(64, 128, kernel_size=1, stride=1),
        nn.InstanceNorm2d(128),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(128, 1, kernel_size=1, stride=1),
    )


class GeneratorLoss(nn.Module):
    tensor_1: torch.Tensor

    def __init__(self, lambda_param: float, lambda_ident: float):
        super().__init__()
        self.register_buffer("tensor_1", torch.ones(1, 1, 6, 6))
        self.lambda_param = lambda_param
        self.lambda_ident = lambda_ident

    def forward(
        self,
        x_A: torch.Tensor,
        x_B: torch.Tensor,
        d_A_fake_A: torch.Tensor,
        d_B_fake_B: torch.Tensor,
        g_A_x_A: torch.Tensor,
        g_B_x_B: torch.Tensor,
        g_B_fake_A: torch.Tensor,
        g_A_fake_B: torch.Tensor,
    ) -> torch.Tensor:
        # loss_gan_A = torch.mean(
        #     torch.sum(
        #         torch.square(d_A_fake_A - 1),
        #         dim=(-3, -2, -1),
        #     )
        # )
        # loss_gan_B = torch.mean(
        #     torch.sum(
        #         torch.square(d_B_fake_B - 1),
        #         dim=(-3, -2, -1),
        #     )
        # )
        # loss_identity_A = torch.mean(torch.sum(torch.abs(g_A_x_A - x_A)), dim=(-3, -2, -1))
        # loss_identity_B = torch.mean(torch.sum(torch.abs(g_B_x_B - x_B)), dim=(-3, -2, -1))
        # loss_cyc_A = torch.mean(
        #     torch.sum(torch.abs(g_B_fake_A - x_B), dim=(-3, -2, -1))
        # )
        # loss_cyc_B = torch.mean(
        #     torch.sum(torch.abs(g_A_fake_B - x_A), dim=(-3, -2, -1))
        # )
        loss_gan_A = nnF.mse_loss(d_A_fake_A, self.tensor_1)
        loss_gan_B = nnF.mse_loss(d_B_fake_B, self.tensor_1)
        loss_identity_A = nnF.l1_loss(g_A_x_A, x_A)
        loss_identity_B = nnF.l1_loss(g_B_x_B, x_B)
        loss_cyc_A = nnF.l1_loss(g_B_fake_A, x_B)
        loss_cyc_B = nnF.l1_loss(g_A_fake_B, x_A)
        return 36 * (loss_gan_A + loss_gan_B) + 196_608 * self.lambda_param * (
            loss_cyc_A
            + loss_cyc_B
            + self.lambda_ident * (loss_identity_A + loss_identity_B)
        )


class DiscriminatorLoss(nn.Module):
    tensor_0: torch.Tensor
    tensor_1: torch.Tensor

    def __init__(self, buffer_size: int):
        super().__init__()
        self.register_buffer("tensor_0", torch.zeros(buffer_size, 1, 6, 6))
        self.register_buffer("tensor_1", torch.ones(1, 1, 6, 6))

    def forward(
        self,
        d_A_x_A: torch.Tensor,
        d_B_x_B: torch.Tensor,
        d_A_fakes_A: torch.Tensor,
        d_B_fakes_B: torch.Tensor,
    ) -> torch.Tensor:
        # loss_real_A = torch.mean(
        #     torch.sum(torch.square(d_A_x_A - 1), dim=(-3, -2, -1))
        # )
        # loss_real_B = torch.mean(
        #     torch.sum(torch.square(d_B_x_B - 1), dim=(-3, -2, -1))
        # )
        # loss_fake_A = torch.mean(
        #     torch.sum(torch.square(d_A_fakes_A), dim=(-3, -2, -1))
        # )
        # loss_fake_B = torch.mean(
        #     torch.sum(torch.square(d_B_fakes_B), dim=(-3, -2, -1))
        # )
        loss_real_A = nnF.mse_loss(d_A_x_A, self.tensor_1)
        loss_real_B = nnF.mse_loss(d_B_x_B, self.tensor_1)
        loss_fake_A = nnF.mse_loss(d_A_fakes_A, self.tensor_0)
        loss_fake_B = nnF.mse_loss(d_B_fakes_B, self.tensor_0)
        return 18 * (loss_real_A + loss_real_B + loss_fake_A + loss_fake_B)


def main(filename: str, checkpoints_folder: str = "./checkpoints"):
    torch.backends.cudnn.benchmark = True

    lr = 0.0002
    batch_size = 1
    epochs = 200
    lambda_param = 10
    lambda_ident = 0.5
    buffer_size = 50

    accelerator = Accelerator(step_scheduler_with_optimizer=False)
    device = cast(Any, accelerator.device)

    train_dataset_A = VisionGANDataset("./horse2zebra/trainA")
    train_dataset_B = VisionGANDataset("./horse2zebra/trainB")

    train_dataloader_A = DataLoader(
        train_dataset_A, batch_size, shuffle=True, pin_memory=True
    )
    train_dataloader_B = DataLoader(
        train_dataset_B, batch_size, shuffle=True, pin_memory=True
    )

    generator_A = Resnet_k(9)
    generator_B = Resnet_k(9)
    discriminator_A = PatchGAN()
    discriminator_B = PatchGAN()

    generator_loss = GeneratorLoss(lambda_param, lambda_ident)
    discriminator_loss = DiscriminatorLoss(buffer_size)

    buffer_A = ImageBufferUltraFast(buffer_size, 3, 256, 256)
    buffer_B = ImageBufferUltraFast(buffer_size, 3, 256, 256)

    generator_optimizer = optim.Adam(
        itertools.chain(generator_A.parameters(), generator_B.parameters()),
        lr,
        betas=(0.5, 0.999),
    )
    discriminator_optimizer = optim.Adam(
        itertools.chain(discriminator_A.parameters(), discriminator_B.parameters()),
        lr,
        betas=(0.5, 0.999),
    )

    generator_scheduler = optim.lr_scheduler.LinearLR(
        generator_optimizer, start_factor=1, end_factor=0, total_iters=100
    )
    discriminator_scheduler = optim.lr_scheduler.LinearLR(
        discriminator_optimizer, start_factor=1, end_factor=0, total_iters=100
    )

    if not os.path.exists(checkpoints_folder):
        os.mkdir(checkpoints_folder)
    
    models = [generator_A, generator_B, discriminator_A, discriminator_B, generator_loss, discriminator_loss, buffer_A, buffer_B]
    
    (
        generator_A,
        generator_B,
        discriminator_A,
        discriminator_B,
        generator_loss,
        discriminator_loss,
        buffer_A,
        buffer_B,
        generator_optimizer,
        discriminator_optimizer,
        train_dataloader_A,
        train_dataloader_B,
        generator_scheduler,
        discriminator_scheduler,
    ) = cast(
        Any,
        accelerator.prepare(
            *[model.to(memory_format=torch.channels_last) for model in models],
            generator_optimizer,
            discriminator_optimizer,
            train_dataloader_A,
            train_dataloader_B,
            generator_scheduler,
            discriminator_scheduler,
        ),
    )
    generator_A: nn.Module
    generator_B: nn.Module
    discriminator_A: nn.Module
    discriminator_B: nn.Module
    generator_loss: nn.Module
    discriminator_loss: nn.Module
    buffer_A: nn.Module
    buffer_B: nn.Module
    generator_optimizer: optim.Optimizer
    discriminator_optimizer: optim.Optimizer
    train_dataloader_A: DataLoader[torch.Tensor]
    train_dataloader_B: DataLoader[torch.Tensor]
    generator_scheduler: optim.lr_scheduler.LRScheduler
    discriminator_scheduler: optim.lr_scheduler.LRScheduler

    NumberOfDatapoints = min(len(train_dataloader_A), len(train_dataloader_B))
    
    for epoch in range(1, epochs + 1):
        accelerator.print(f"== EPOCH: {epoch}/{epochs} ==")
        gen_losses = torch.zeros(1, device=device)
        disc_losses = torch.zeros(1, device=device)
        x_A: torch.Tensor
        x_B: torch.Tensor
        for x_A, x_B in zip(train_dataloader_A, train_dataloader_B):
            x_A = x_A.to(memory_format=torch.channels_last)
            x_B = x_B.to(memory_format=torch.channels_last)
            # Generate images
            fake_A: torch.Tensor = generator_A(x_B)
            fake_B: torch.Tensor = generator_B(x_A)
            # Generator loss
            discriminator_A.requires_grad_(False)
            discriminator_B.requires_grad_(False)
            #
            gen_loss = generator_loss(
                x_A,
                x_B,
                discriminator_A(fake_A),
                discriminator_B(fake_B),
                generator_A(x_A),
                generator_B(x_B),
                generator_B(fake_A),
                generator_A(fake_B),
            )
            gen_losses += gen_loss
            generator_optimizer.zero_grad(set_to_none=True)
            accelerator.backward(gen_loss)
            generator_optimizer.step()
            # Discriminator loss
            discriminator_A.requires_grad_(True)
            discriminator_B.requires_grad_(True)
            #
            fakes_A = buffer_A(fake_A.detach())
            fakes_B = buffer_B(fake_B.detach())
            disc_loss = discriminator_loss(
                discriminator_A(x_A),
                discriminator_B(x_B),
                discriminator_A(fakes_A),
                discriminator_B(fakes_B),
            )
            disc_losses += disc_loss
            discriminator_optimizer.zero_grad(set_to_none=True)
            accelerator.backward(disc_loss)
            discriminator_optimizer.step()
        if epoch >= 100:
            generator_scheduler.step()
            discriminator_scheduler.step()
        if epoch % 10 == 0:
            accelerator.save_model(
                generator_A, os.path.join(checkpoints_folder, f"generator_A_{epoch}")
            )
            accelerator.save_model(
                generator_B, os.path.join(checkpoints_folder, f"generator_B_{epoch}")
            )
            accelerator.save_model(
                discriminator_A,
                os.path.join(checkpoints_folder, f"discriminator_A_{epoch}"),
            )
            accelerator.save_model(
                discriminator_B,
                os.path.join(checkpoints_folder, f"discriminator_B_{epoch}"),
            )
        accelerator.print("Generator loss: ", (gen_losses / NumberOfDatapoints).item())
        accelerator.print(
            "Discriminator loss: ", (disc_losses / NumberOfDatapoints).item()
        )


if __name__ == "__main__":
    main(*sys.argv)