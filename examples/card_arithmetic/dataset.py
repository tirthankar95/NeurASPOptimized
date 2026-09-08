import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import io

current_dir = os.path.dirname(os.path.abspath(__file__))


class CardArithmetic(Dataset):

    def __init__(self, image_dir, variant='sum_2', train_val_test='train'):
        """
        Initialize Card arithmetic dataset by loading image and task data.
        :param image_dir: String of directory that contains train/val/test folders with image data
        :param variant: String denoting the variant of card arithmetic
        :param train_val_test: String indicating whether to load the train, validation or test data
        """
        if train_val_test == 'val':
            # Val data uses train images
            self.image_dir = f'{image_dir}/train'
        else:
            self.image_dir = f'{image_dir}/{train_val_test}'

        self.data = pd.read_csv(current_dir + f"/data/card_{variant}_{train_val_test}.csv")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Get image and labels.
        :param index: Index of the example in the task dataset
        :return: Dictionary containing the input images and the downstream label
        """
        imgs = []
        img_idxs = self.data.iloc[index]
        for col_name, value in img_idxs.items():
            if col_name.startswith('card'):
                # Load image corresponding to card index
                img = io.read_image(self.image_dir + f"/{value}.jpg").float() / 255.0
                imgs.append(img)
            if col_name == 'result':
                label = value
        return {'p': torch.stack(imgs)}, f':- not result({label}).'
