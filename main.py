import os
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.transforms import v2
import safetensors
import safetensors.torch
import tqdm
import argparse
from typing import Any, cast, Callable, Iterable, TypeVar


def get_checkpoint_epochs(checkpoints_folder: str):
    return sorted(
        set(map(lambda x: x.rsplit("_", 1)[1], os.listdir(checkpoints_folder))),
        reverse=True,
        key=lambda x: int(x),
    )


def get_transform_params():
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    return mean, std


def get_transform():
    mean, std = get_transform_params()
    return v2.Compose(
        [
            v2.Resize(286, v2.InterpolationMode.BICUBIC, antialias=True),
            v2.RandomCrop(256),
            v2.RandomHorizontalFlip(),
            v2.ToDtype(torch.get_default_dtype(), scale=True),
            v2.Normalize(mean, std),
        ]
    )


def get_inverse_transform():
    mean, std = get_transform_params()
    mean_tensor = torch.tensor(mean).reshape(3, 1, 1)
    std_tensor = torch.tensor(std).reshape(3, 1, 1)

    def _inverse(x: torch.Tensor) -> torch.Tensor:
        return x * std_tensor + mean_tensor

    return v2.Compose([v2.Lambda(_inverse), v2.ToDtype(torch.uint8, scale=True)])


class VisionGANDataset(Dataset[torch.Tensor]):
    def __init__(self, images_folder: str):
        super().__init__()
        self.transform = get_transform()
        self.images = [
            os.path.join(images_folder, image) for image in os.listdir(images_folder)
        ]

    def __enter__(self):
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any):
        pass

    def __len__(self):
        return len(self.images)

    def get_image(self, index: int) -> torch.Tensor:
        return torchvision.io.read_image(
            self.images[index], torchvision.io.ImageReadMode.RGB
        )

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.transform(self.get_image(index))


class SafetensorsVisionGANDataset(Dataset[torch.Tensor]):
    def __init__(self, images_file: str, device: str):
        super().__init__()
        self.transform = get_transform()
        self.images_fp = safetensors.safe_open(images_file, "pt", device)

    def __enter__(self):
        self.images_fp = self.images_fp.__enter__()
        self.keys = self.images_fp.keys()
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any):
        self.images_fp.__exit__(type, value, traceback)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.transform(self.images_fp.get_tensor(self.keys[index]))


class FullVisionGANDataset(Dataset[torch.Tensor]):
    def __init__(self, images_folder: str, device: str | torch.device):
        self.transform = get_transform()
        self.tensors = [
            torchvision.io.read_image(
                os.path.join(images_folder, image), torchvision.io.ImageReadMode.RGB
            ).to(device)
            for image in os.listdir(images_folder)
        ]

    def __enter__(self):
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any):
        pass

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.transform(self.tensors[index])


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


def init_module(
    model: nn.Module, init: Callable[[nn.Module], Any], *classes: type[nn.Module]
):
    for m in model.modules():
        if isinstance(m, cast(Any, classes)):
            init(m)


def init_conv(model: nn.Module, init: Callable[[torch.Tensor], Any]):
    init_module(
        model,
        lambda x: init(cast(nn.Conv2d, x).weight),
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
    )


def Resnet_c7s1_k(
    in_k: int, out_k: int, instance_norm: bool, output_transform: nn.Module
):
    return nn.Sequential(
        nn.Conv2d(in_k, out_k, kernel_size=7, padding=3, padding_mode="reflect"),
        *((nn.InstanceNorm2d(out_k),) if instance_norm else ()),
        output_transform,
    )


def Resnet_dk(in_k: int, out_k: int):
    return nn.Sequential(
        nn.Conv2d(in_k, out_k, kernel_size=3, stride=2, padding=1),
        nn.InstanceNorm2d(out_k),
        nn.ReLU(inplace=True),
    )


class Resnet_Rk(nn.Module):
    def __init__(self, in_k: int, out_k: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_k, out_k, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(out_k),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_k, out_k, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(out_k),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr = self.model(x)
        return xr + x  # Residual connection


def Resnet_uk(in_k: int, out_k: int, output_padding: int = 0):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_k,
            out_k,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=output_padding,
        ),
        nn.InstanceNorm2d(out_k),
        nn.ReLU(inplace=True),
    )


def Resnet_k(k: int):
    return nn.Sequential(
        Resnet_c7s1_k(
            3, 64, instance_norm=True, output_transform=nn.ReLU(inplace=True)
        ),
        Resnet_dk(64, 128),
        Resnet_dk(128, 256),
        *(Resnet_Rk(256, 256) for _ in range(k)),
        Resnet_uk(256, 128, output_padding=1),
        Resnet_uk(128, 64, output_padding=1),
        Resnet_c7s1_k(64, 3, instance_norm=False, output_transform=nn.Tanh()),
    )


def UNet_encoder(in_k: int, out_k: int, batch_norm: bool = True):
    return nn.Sequential(
        nn.Conv2d(in_k, out_k, kernel_size=4, stride=2, bias=(not batch_norm)),
        *((nn.BatchNorm2d(out_k),) if batch_norm else ()),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
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
            in_k,
            out_k,
            kernel_size=4,
            stride=2,
            output_padding=output_padding,
            bias=False,
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
            nn.ReLU(inplace=True),
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
        # return 36 * (loss_gan_A + loss_gan_B) + 196_608 * self.lambda_param * (
        #     loss_cyc_A
        #     + loss_cyc_B
        #     + self.lambda_ident * (loss_identity_A + loss_identity_B)
        # )
        return (
            loss_gan_A
            + loss_gan_B
            + self.lambda_param
            * (
                loss_cyc_A
                + loss_cyc_B
                + self.lambda_ident * (loss_identity_A + loss_identity_B)
            )
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
        d_x: torch.Tensor,
        d_fakes: torch.Tensor,
    ) -> torch.Tensor:
        # loss_real = torch.mean(
        #     torch.sum(torch.square(d_x - 1), dim=(-3, -2, -1))
        # )
        # loss_fake = torch.mean(
        #     torch.sum(torch.square(d_fakes), dim=(-3, -2, -1))
        # )
        loss_real = nnF.mse_loss(d_x, self.tensor_1)
        loss_fake = nnF.mse_loss(d_fakes, self.tensor_0)
        # return 18 * (loss_real + loss_fake)
        return 0.5 * (loss_real + loss_fake)


def custom_collate(data: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(data).to(memory_format=torch.channels_last)


T1 = TypeVar("T1")
T2 = TypeVar("T2")


def zip_long(it1: Iterable[T1], it2: Iterable[T2]) -> Iterable[tuple[T1, T2]]:
    i1 = iter(it1)
    i2 = iter(it2)
    b1 = False
    b2 = False
    while True:
        try:
            n1 = next(i1)
        except StopIteration:
            b1 = True
            i1 = iter(it1)
            n1 = next(i1)
        try:
            n2 = next(i2)
        except StopIteration:
            b2 = True
            i2 = iter(it2)
            n2 = next(i2)
        if b1 and b2:
            break
        yield n1, n2


def train_model(checkpoints_folder: str, resume: bool, compile_models: bool):
    lr = 0.0002
    batch_size = 1
    epochs = 200
    lambda_param = 10
    lambda_ident = 0.5
    buffer_size = 50

    if not torch.cuda.is_available():
        raise RuntimeError("Cuda not available")
    if torch.cuda.device_count() > 1:
        print(
            "You have multiple cuda devices available, count: ",
            torch.cuda.device_count(),
        )
    device = torch.device("cuda", torch.cuda.current_device())

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    with FullVisionGANDataset(
        "./horse2zebra/trainA", device
    ) as train_dataset_A, FullVisionGANDataset(
        "./horse2zebra/trainB", device
    ) as train_dataset_B:
        train_dataloader_A = DataLoader(
            train_dataset_A, batch_size, shuffle=True, collate_fn=custom_collate
        )
        train_dataloader_B = DataLoader(
            train_dataset_B, batch_size, shuffle=True, collate_fn=custom_collate
        )

        generator_A = Resnet_k(9).to(device)
        generator_B = Resnet_k(9).to(device)
        discriminator_A = PatchGAN().to(device)
        discriminator_B = PatchGAN().to(device)

        for model in (generator_A, generator_B, discriminator_A, discriminator_B):
            init_conv(model, nn.init.xavier_normal_)

        generator_loss = GeneratorLoss(lambda_param, lambda_ident).to(device)
        discriminator_loss = DiscriminatorLoss(buffer_size).to(device)

        buffer_A = ImageBufferUltraFast(buffer_size, 3, 256, 256).to(device)
        buffer_B = ImageBufferUltraFast(buffer_size, 3, 256, 256).to(device)

        if compile_models:
            generator_A: nn.Module = torch.compile(generator_A)
            generator_B: nn.Module = torch.compile(generator_B)
            discriminator_A: nn.Module = torch.compile(discriminator_A)
            discriminator_B: nn.Module = torch.compile(discriminator_B)
            generator_loss: nn.Module = torch.compile(generator_loss)
            discriminator_loss: nn.Module = torch.compile(discriminator_loss)
            buffer_A: nn.Module = torch.compile(buffer_A)
            buffer_B: nn.Module = torch.compile(buffer_B)

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

        data_count = max(len(train_dataloader_A), len(train_dataloader_B))

        generator_A.train()
        generator_B.train()

        scaler = torch.cuda.amp.grad_scaler.GradScaler()

        writer = print

        if not os.path.exists(checkpoints_folder):
            os.mkdir(checkpoints_folder)

        start_epoch = 0

        if resume:
            checkpoint_epochs = get_checkpoint_epochs(checkpoints_folder)
            if len(checkpoint_epochs) == 0:
                raise RuntimeError("No checkpoints found")
            start_epoch = int(checkpoint_epochs[0])
            safetensors.torch.load_model(
                generator_A,
                os.path.join(checkpoints_folder, f"generator_A_{start_epoch}"),
            )
            safetensors.torch.load_model(
                generator_B,
                os.path.join(checkpoints_folder, f"generator_B_{start_epoch}"),
            )
            safetensors.torch.load_model(
                discriminator_A,
                os.path.join(checkpoints_folder, f"discriminator_A_{start_epoch}"),
            )
            safetensors.torch.load_model(
                discriminator_B,
                os.path.join(checkpoints_folder, f"discriminator_B_{start_epoch}"),
            )
            generator_optimizer.load_state_dict(
                torch.load(
                    os.path.join(
                        checkpoints_folder, f"generator_optimizer_{start_epoch}"
                    ),
                )
            )
            discriminator_optimizer.load_state_dict(
                torch.load(
                    os.path.join(
                        checkpoints_folder, f"discriminator_optimizer_{start_epoch}"
                    ),
                )
            )
            generator_scheduler.load_state_dict(
                torch.load(
                    os.path.join(
                        checkpoints_folder, f"generator_scheduler_{start_epoch}"
                    ),
                )
            )
            discriminator_scheduler.load_state_dict(
                torch.load(
                    os.path.join(
                        checkpoints_folder, f"discriminator_scheduler_{start_epoch}"
                    ),
                )
            )
            scaler.load_state_dict(
                torch.load(
                    os.path.join(checkpoints_folder, f"scaler_{start_epoch}"),
                )
            )

        x_A: torch.Tensor
        x_B: torch.Tensor

        with torch.no_grad():
            for _, x_A, x_B in zip(
                range(buffer_size), train_dataloader_A, train_dataloader_B
            ):
                buffer_A(generator_A(x_B))
                buffer_B(generator_B(x_A))

        for epoch in range(start_epoch + 1, epochs + 1):
            writer(f"== EPOCH: {epoch}/{epochs} ==")
            gen_losses = torch.zeros(1, device=device)
            disc_losses = torch.zeros(1, device=device)
            for x_A, x_B in tqdm.tqdm(zip_long(train_dataloader_A, train_dataloader_B)):
                generator_optimizer.zero_grad(set_to_none=True)
                #
                with torch.autocast(device_type=device.type, dtype=torch.float16):
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
                scaler.scale(gen_loss).backward()
                scaler.step(generator_optimizer)
                # Discriminator loss
                discriminator_A.requires_grad_(True)
                discriminator_B.requires_grad_(True)
                #
                discriminator_optimizer.zero_grad(set_to_none=True)
                #
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    fakes_A = buffer_A(fake_A.detach())
                    disc_loss_A = discriminator_loss(
                        discriminator_A(x_A),
                        discriminator_A(fakes_A),
                    )
                    fakes_B = buffer_B(fake_B.detach())
                    disc_loss_B = discriminator_loss(
                        discriminator_B(x_B),
                        discriminator_B(fakes_B),
                    )
                    disc_losses += disc_loss_A
                    disc_losses += disc_loss_B
                scaler.scale(disc_loss_A).backward()
                scaler.scale(disc_loss_B).backward()
                scaler.step(discriminator_optimizer)
                #
                scaler.update()
            if epoch >= 100:
                generator_scheduler.step()
                discriminator_scheduler.step()
            writer("Generator loss: ", (gen_losses / data_count).item())
            writer("Discriminator loss: ", (disc_losses / data_count).item())
            if epoch % 10 == 0:
                safetensors.torch.save_model(
                    generator_A,
                    os.path.join(checkpoints_folder, f"generator_A_{epoch}"),
                )
                safetensors.torch.save_model(
                    generator_B,
                    os.path.join(checkpoints_folder, f"generator_B_{epoch}"),
                )
                safetensors.torch.save_model(
                    discriminator_A,
                    os.path.join(checkpoints_folder, f"discriminator_A_{epoch}"),
                )
                safetensors.torch.save_model(
                    discriminator_B,
                    os.path.join(checkpoints_folder, f"discriminator_B_{epoch}"),
                )
                torch.save(
                    generator_optimizer.state_dict(),
                    os.path.join(checkpoints_folder, f"generator_optimizer_{epoch}"),
                )
                torch.save(
                    discriminator_optimizer.state_dict(),
                    os.path.join(
                        checkpoints_folder, f"discriminator_optimizer_{epoch}"
                    ),
                )
                torch.save(
                    generator_scheduler.state_dict(),
                    os.path.join(checkpoints_folder, f"generator_scheduler_{epoch}"),
                )
                torch.save(
                    discriminator_scheduler.state_dict(),
                    os.path.join(
                        checkpoints_folder, f"discriminator_scheduler_{epoch}"
                    ),
                )
                torch.save(
                    scaler.state_dict(),
                    os.path.join(checkpoints_folder, f"scaler_{epoch}"),
                )


def main():
    parser = argparse.ArgumentParser("CycleGAN")
    parser.add_argument(
        "-chd",
        "--checkpoint_dir",
        dest="checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="where to store the model checkpoints",
    )
    parser.add_argument(
        "-r",
        "--resume",
        dest="resume",
        action="store_true",
        help="whether to resume training from the checkpoints directory",
    )
    parser.add_argument(
        "-c",
        "--compile",
        dest="compile_model",
        action="store_true",
        help="whether to compile the model before running using the default torch.dynamo backend",
    )
    args = parser.parse_args()
    train_model(args.checkpoint_dir, args.resume, args.compile_model)


if __name__ == "__main__":
    main()
