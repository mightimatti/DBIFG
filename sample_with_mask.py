import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils.memory_mapped_scan_dataset import MemoryMappedScanDataset
from torchvision.utils import save_image, make_grid
from collections import defaultdict
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
EPOCHS = 1
IMAGE_SIZE = 256

schedule = [
295,
280,
285,
160,
165,
140,
145,
120,
130,
100,
108,
110,
65,
70,
46,
50,
36,
40,
29,
32,
25,
23,
19,
20,
17,
15,
14,
13,
14,
12,
13,
11,
12,
10,
11,
9,
10,
8,
9,
7,
8,
6,
7,
5,
6,
4,
5,
3,
4,
2,
3,
1,
]


class GaussianDiffWMask(GaussianDiffusion):
    def set_original_image(self, image):
        self.image = image

    def set_mask(self, mask):
        self.mask = mask

    def undo(self, image_after_model, t, debug=False):
        # calculate beta and alpha for current timestep
        beta = 1- self.alphas_cumprod[t]
        alpha = self.alphas_cumprod[t]

        # estimate previous image through adding noise
        noise = torch.sqrt(beta).view(-1,1,1,1) * torch.randn_like(image_after_model)
        return torch.sqrt(alpha).view(-1,1,1,1) * image_after_model + noise

    def p_sample(self, x, t: int, pred_xstart, x_self_cond=None):
        b, *_, device = *x.shape, self.device
        # create a tensor of Shape  (B, 1) containing T
        if pred_xstart is not None:
            gt_weight = torch.sqrt(self.alphas_cumprod[t])
            gt_part = gt_weight.view(-1,1,1,1)  * self.image

            noise_weight = torch.sqrt((1 - self.alphas_cumprod[t]))
            noise_part = noise_weight.view(-1,1,1,1)  * torch.randn_like(x)

            weighed_gt = gt_part + noise_part

            x = self.mask * (weighed_gt) + (1 - self.mask) * (x)

        # feed through the model predicting statistical values
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=t, x_self_cond=x_self_cond, clip_denoised=True
        )
        noise = torch.randn_like(x) if t.any() else 0.0  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise

        return pred_img, x_start

    def inpaint_with_mask(
        self,
        images,
        gt_keep_mask,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        # assert isinstance(shape, (tuple, list))

        self.set_original_image(images)
        self.set_mask(gt_keep_mask)

        image_after_step = torch.randn(*images.shape, device=device)

        pred_xstart = None

        idx_wall = -1
        sample_idxs = defaultdict(lambda: 0)

        # use TQDM for visualization
        time_pairs = tqdm(list(zip(schedule[:-1], schedule[1:])))

        pred_img = None
        x_start = None
        for t_last, t_cur in time_pairs:
            idx_wall += 1

            # shape (B, 1)
            t_last_t = torch.tensor(
                [t_last] * images.shape[0],
                device=device,  # pylint: disable=not-callable
            )

            # copy current version
            image_before_step = image_after_step.clone()

            # print(f"{t_cur}, {t_last}")
            if t_cur < t_last:
                # case regular diffusion
                gt_weight = torch.sqrt(self.alphas_cumprod[t_cur])
                gt_part = gt_weight * images

                noise_weight = torch.sqrt((1 - self.alphas_cumprod[t_cur]))
                noise_part = noise_weight * torch.randn_like(image_before_step)

                weighed_gt = gt_part + noise_part

                image_before_step = gt_keep_mask * (weighed_gt) + (1 - gt_keep_mask) * (
                    image_before_step
                )
               
                if t_cur == 0 :
                    yield image_before_step
                
                with torch.no_grad():
                    pred_img, x_start = self.p_sample(
                        image_after_step,
                        t_last_t,
                        pred_xstart=pred_xstart,
                    )

                    sample_idxs[t_cur] += 1

                    yield x_start

            else:
                t_shift = 1

                image_after_step = self.undo(
                    x_start,
                    t=t_last_t + t_shift,
                    debug=False,
                )
                pred_xstart = image_after_step


if __name__ == "__main__":
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        # flash_attn=True,
        # random_fourier_features = True,
    )

    model.load_state_dict(
        torch.load(
            "epoch_18.model",
            map_location=device,
        ),
    )

    diffusion = GaussianDiffWMask(
        model, image_size=IMAGE_SIZE, timesteps=300
    )  # number of steps

    model.to(device)
    diffusion.to(device)
    DATASET = "/media/bean/1c6e3fbe-7865-44ae-bf37-9bbb323c174c/scan_calisto_preprocessor/datasets/250_256_CRI_SLIDES_250_173003"
    model_dataset = MemoryMappedScanDataset(DATASET)


    for x in range(10, 30):
        nr = random.choice(range(len(model_dataset)))
        useful_9, masks_raw = model_dataset[nr:nr+9]

        masks = torch.zeros_like(masks_raw)
        masks = masks_raw

        # useful_9[
        #     masks_itn
        # ] = 0.0

        # masks[
        #     :,
        #     :,
        #     44:196,
        #     44:196,
        # ] = 0.0
        print(f"grid shape: {useful_9.shape}")
        print(f"masks shape: {masks.shape}")

        useful_9 = useful_9.to(device)
        grid = make_grid(
            useful_9,
            nrow=3,
        )
        save_image(grid, f"{x:02d}__input.jpg")

        masks = masks.to(device)
        grid = make_grid(
            masks,
            nrow=3,
        )
        save_image(grid, f"{x:02d}__masks.jpg")

        # exit(1)
        
        with torch.no_grad():
            inpainting = diffusion.inpaint_with_mask(useful_9, masks)
            for idx, epoch in enumerate(inpainting):
                grid = make_grid(
                    epoch,
                    nrow=3,
                )
                # save_image(grid, f"epoch/{idx}.jpg")
            save_image(grid, f"{x:02d}_result.jpg")

    # print(sampled_images.shape)  # (4, 3, 128, 128)
