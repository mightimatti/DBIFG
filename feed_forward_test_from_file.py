import torch
import numpy as np
import cv2
import os
import sys
from torchvision.utils import save_image, make_grid

from sample_with_mask import Unet, GaussianDiffWMask



def reconstruct_image_from_tensor(tensor, image_shape):
    """
    utility function to rebuild the image from a tensor

    [args]
        `tensor`: (torch.tensor) tensor containing the image data.
        `image_shape`: (tuple) Tuple containing the amount of patches along the
            y- and x-axis (respectively) of original image.
    [returns]
        [np.ndarray] image containing the data from the tensor

    """

    # print(tensor.shape)
    # permute to numpy shape
    tensor = torch.permute(tensor, (0, 2, 3, 1))
    # remove last dimension, if grayscale
    tensor = tensor.squeeze()

    patch_size = tensor.shape[1]

    size_y = image_shape[0] * patch_size
    size_x = image_shape[1] * patch_size

    img_shape = (
        (size_y, size_x, 3)
        if len(tensor.shape) != 3
        else (
            size_y,
            size_x,
        )
    )

    # generate index pairs to copy data
    X, Y = np.meshgrid(
        range(0, size_x, patch_size),
        range(0, size_y, patch_size),
    )
    index_pairs = np.vstack((X.flatten(), Y.flatten())).T

    # account for color images and grayscale feature maps
    img = np.empty(img_shape, dtype=np.float32)
    for idx, (y, x) in enumerate(index_pairs):
        img[x : x + patch_size, y : y + patch_size] = tensor[idx].numpy()
    img = img.clip(0.0, 1.0)

    return img


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
patch_size = 256
SCALING_FACTOR = 4
# DATASET = "dataset/negative_96_256_2_bnw"
print(device)
IMAGE_SIZE = 256


model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    # flash_attn=True,
    # random_fourier_features = True,
)

model.load_state_dict(
    torch.load(
        "epoch_45.model",
        map_location=device,
    ),
)

diffusion = GaussianDiffWMask(
    model, image_size=IMAGE_SIZE, timesteps=300
)  # number of steps

model.to(device)
diffusion.to(device)


def slice_file_and_feed_forward(fp):
    image = cv2.imread(fp, cv2.IMREAD_UNCHANGED).astype(np.float32) / 2**16
    if image.shape[-1] == 4:
        mask =  image[:,:,3]
        image = image[:,:,:3]

    image = np.power(image, 1 / 2.2)
    mask = np.power(mask, 1 / 2.2)

    print(f'image.shape: {image.shape}')
    print(f'mask.shape: {mask.shape}')


    small_image = cv2.resize(
        image,
        (image.shape[1] // SCALING_FACTOR, image.shape[0] // SCALING_FACTOR),
        interpolation=cv2.INTER_LINEAR,
    )   

    small_mask = cv2.resize(
        mask,
        (mask.shape[1] // SCALING_FACTOR, mask.shape[0] // SCALING_FACTOR),
        interpolation=cv2.INTER_LINEAR,
    )

    res_x = (small_image.shape[1] % patch_size) // 2
    res_y = (small_image.shape[0] % patch_size) // 2

    # patches_x, patches_y = (small_image.shape[1] // patch_size, small_image.shape[0] // patch_size)

    min_x, min_y, max_x, max_y = (
        res_x,
        res_y ,
        small_image.shape[1] - res_x - patch_size ,
        small_image.shape[0] - res_y - patch_size ,
    )

    patches_x = (max_x - min_x) // patch_size
    patches_y = (max_y - min_y) // patch_size
    
    num_patches = patches_x * patches_y

    print(
        "Copying {} x {} = {} patches into tensor".format(
            patches_x, patches_y, num_patches
        )
    )

    feature_tensor = torch.zeros(
        (num_patches, 3, patch_size, patch_size), dtype=torch.float32
    )
    mask_tensor = torch.zeros(
        (num_patches, 1, patch_size, patch_size), dtype=torch.float32
    )

    r_x = range(min_x, patches_x*patch_size, patch_size)
    r_y = range(min_y, patches_y*patch_size, patch_size)

    # generate index pairs to copy data
    X, Y = np.meshgrid(r_x, r_y)
    index_pairs = np.vstack((X.flatten(), Y.flatten())).T
    # print(len(index_pairs))
    
    for idx, (y, x) in enumerate(index_pairs):

        feature_instance = torch.from_numpy(
            # get image region
            (small_image[x : x + patch_size, y : y + patch_size]).transpose(2, 0, 1)
        ).clone()
        # print(feature_instance.shape)
        feature_tensor[idx] = feature_instance

        mask_instance = torch.from_numpy(
            # get image region
            (small_mask[x : x + patch_size, y : y + patch_size])
        ).clone()
        # print(mask_instance.shape)
        mask_tensor[idx] = mask_instance

    im = reconstruct_image_from_tensor(
        feature_tensor,
        (
            patches_y,
            patches_x,
        ),
    )
    output_im = (im * 2**16).astype(np.uint16)
    cv2.imwrite("output_trimmed_pre.tif", cv2.cvtColor(output_im, cv2.COLOR_BGR2RGB))
    
    new_mask = torch.zeros_like(mask_tensor)
    mask_mean = torch.mean(mask_tensor) * 0.98
    new_mask[mask_tensor >= mask_mean] = 1.0



    out_img = reconstruct_image_from_tensor(
        final,
        (
            patches_y,
            patches_x,
        ),
    )

    output_im = (out_img * 2**16).astype(np.uint16)
    cv2.imwrite("output_post.tif", cv2.cvtColor(output_im, cv2.COLOR_BGR2RGB))
    # cv2.imwrite(filepath, )

    # cv2.waitKey(0)


if __name__ == "__main__":
    file_path = sys.argv[1]
    # filename = os.path.basename(file_path)
    slice_file_and_feed_forward(file_path)
