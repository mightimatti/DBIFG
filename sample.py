import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.memory_mapped_scan_dataset import MemoryMappedScanDataset
from torchvision.utils import save_image, make_grid


"""
	CURRENTLY DOESN'T WORK

"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
EPOCHS = 1

model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        # random_fourier_features = True,
    )


model.load_state_dict(
    torch.load(
        "epoch_1.model",
        map_location=torch.device(device),
    ),
)
diffusion = GaussianDiffusion(model, image_size=256, timesteps=512)  # number of steps

model.to(device)
diffusion.to(device)
with torch.no_grad():
    sampled_images = diffusion.sample(batch_size=4)
# print(sampled_images.shape)  # (4, 3, 128, 128)
grid = make_grid(sampled_images, nrow=2,)
save_image(grid, "test.jpg")