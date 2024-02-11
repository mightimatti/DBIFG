import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import cycle
from torch.utils.data import DataLoader

from utils.memory_mapped_diffusion_dataset import MemoryMappedDiffusionDataset


def train():
    IMAGE_SIZE = 128
    BATCH_SIZE = 16

    model = Unet(
        dim = 64,
        channels=2,
        dim_mults = (1, 2, 4, 8),
        flash_attn = True
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = IMAGE_SIZE,
        timesteps = 256,           # number of steps
        sampling_timesteps = 64    # number of sampling timesteps
    )

    # DATASET = "/media/bean/1c6e3fbe-7865-44ae-bf37-9bbb323c174c/scan_calisto_preprocessor/datasets/250_256_CRI_SLIDES_250_173003"
    DATASET = "/media/bean/1c6e3fbe-7865-44ae-bf37-9bbb323c174c/tubcloud/998_128_SLIDES_741206"
    model_dataset = MemoryMappedDiffusionDataset(DATASET, IMAGE_SIZE=IMAGE_SIZE)
    dataset_loader = DataLoader(model_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory = True, num_workers = 4)

    trainer = Trainer(
        diffusion,
        '/media/bean/1c6e3fbe-7865-44ae-bf37-9bbb323c174c/tubcloud/images',
        train_batch_size = BATCH_SIZE,
        train_lr = 8e-5,
        train_num_steps = 160000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = False,                       # turn on mixed precision
        calculate_fid=False,
        save_and_sample_every = 1000,
    )

    trainer.dl = model_dataset
    trainer.accelerator.prepare(dataset_loader)
    trainer.dl = cycle(dataset_loader)
    trainer.load(125)
    
    trainer.train()


if __name__ == "__main__":
    train()
