import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import cycle
from torch.utils.data import DataLoader

from utils.memory_mapped_diffusion_dataset import MemoryMappedDiffusionDataset


def train():
    IMAGE_SIZE = 128
    BATCH_SIZE = 38

    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8, 16),
        flash_attn = True
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = IMAGE_SIZE,
        timesteps = 512,           # number of steps
        sampling_timesteps = 128    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    )

    # DATASET = "/media/bean/1c6e3fbe-7865-44ae-bf37-9bbb323c174c/scan_calisto_preprocessor/datasets/250_256_CRI_SLIDES_250_173003"
    DATASET = "/root/998_128_SLIDES_741206"
    model_dataset = MemoryMappedDiffusionDataset(DATASET, IMAGE_SIZE=IMAGE_SIZE)
    dataset_loader = DataLoader(model_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory = True, num_workers = 32)

    trainer = Trainer(
        diffusion,
        '/root/images',
        train_batch_size = BATCH_SIZE,
        train_lr = 8e-5,
        train_num_steps = 150000,         # total training steps
        gradient_accumulate_every = 1,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
    )

    trainer.dl = model_dataset
    trainer.accelerator.prepare(dataset_loader)
    trainer.dl = cycle(dataset_loader)
    
    trainer.train()

if __name__ == "__main__":
    train()
