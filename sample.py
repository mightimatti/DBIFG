import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.memory_mapped_scan_dataset import MemoryMappedScanDataset
from torchvision.utils import save_image, make_grid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 128

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

diffusion.load_state_dict(
    torch.load(
        "/media/bean/1c6e3fbe-7865-44ae-bf37-9bbb323c174c/DBIFG/checkpoints_vast/model-33.pt",
        map_location=torch.device(device),
    )["model"],
)


model.to(device)
diffusion.to(device)
with torch.no_grad():
    sampled_images = diffusion.ddim_sample((54,3,IMAGE_SIZE,IMAGE_SIZE))
# print(sampled_images.shape)  # (4, 3, 128, 128)
grid = make_grid(sampled_images, nrow=9,)
save_image(grid, "test_rgb.png")