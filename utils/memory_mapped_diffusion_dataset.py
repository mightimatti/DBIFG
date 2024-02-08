import torch
import os
import numpy as np
import json

# FEATURE_PATCHES_PATTERN = "feature_patches.npy"
# MAP_PATCHES_PATTERN = "map_patches.npy"
INPUT_DTYPE = np.float16


class MemoryMappedDiffusionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        IMAGE_SIZE=128,
    ):
        """Construct an indexed list of image paths and labels"""
        if not os.path.isdir(data_dir):
            raise ValueError("Invalid Folder")
        self.data_dir = data_dir

        stats_file_path = os.path.join(data_dir, "stats.json")
        # get stats
        try:
            with open(stats_file_path, "r") as f:
                self.stats = json.load(f)
        except FileNotFoundError:
            raise ValueError("Could not Load {}".format(stats_file_path))
        self.all_items = []
        for fn, item_list in self.stats.items():
            bn = os.path.basename(fn)
            if bn.startswith("feature_patches"):
                count_items = item_list[-1]["index_range_to"]+1
                self.all_items += [
                    (bn, ind,) for ind in range(count_items)
                ]
            elif bn.startswith("map_patches"):
                continue

        self.image_size = IMAGE_SIZE

    def __getitem__(self, n):
        """
        Load features and map
        """
        fn, sub_idx = self.all_items[n]
        mmap = np.memmap(
            os.path.join(self.data_dir, fn),
            dtype=INPUT_DTYPE,
            mode="r",
        ).reshape((-1, 3, self.image_size, self.image_size))
    
        return torch.from_numpy(np.array(mmap[sub_idx]))

    def __len__(self):
        """return the total number of images in this dataset"""
        return len(self.all_items)
