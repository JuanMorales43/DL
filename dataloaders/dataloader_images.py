import glob, random
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import os
import torchvision.transforms.functional as TF
import natsort as ns
import pandas as pd
import torch


class ImageDataset(Dataset):
    """
    Dataset class for the intratumoral patch classification task.

    Every row of the split csv describes one lesion: the first column holds the
    lesion_ID and the second one its binary label (0 = benigno, 1 = maligno). The
    patch of a lesion lives in its own folder, `<path_images>/<lesion_ID>/<lesion_ID>.tiff>`.

    Args:
        path_images (str): Path to the patches directory.
        path_data (str): Path to the split csv with the lesion_ID and the labels.
        transforms (torchvision.transforms.Compose, optional): Transformations to apply to images.
            Defaults to None.
    """
    
    def __init__(self, path_images:str, path_data:str, transforms: T.Compose = None):
        
        self.transforms     = transforms        
        self.data           = pd.read_csv(path_data)
        self.path_images    = path_images
                
    def __getitem__(self, index):
        """
        Return the patch of a lesion and its label.

        Args:
            index (int): Index of the lesion.

        Returns:
            tuple: Input image and its label.
        """
        sample      = index % len(self.data)

        # Load image
        name_image  = self.data.iloc[sample, 0]
        path_image  = os.path.join(self.path_images, name_image, f"{name_image}.tiff")
        im_input_   = Image.open(path_image).convert("RGB")
        im_input_   = self.transforms(im_input_)
        
        target      = self.data.iloc[sample, 1]
        target      = torch.tensor(target, dtype=torch.long)

        return im_input_, target
    
    def __len__(self):
        """
        Return the number of lesions in the dataset.

        Returns:
            int: Number of lesions.
        """
        return len(self.data)
    

class Loader:
    """
    Class that creates Data Loaders

    Args:
        images_dir (str): Path to the patches directory.
        data_dir (str): Path to the directory with the split csv files (train.csv, test.csv, val.csv).
        augmentation (bool, optional): Apply random augmentations to images. Defaults to False.
    """

    def __init__(self, images_dir: str, data_dir: str,  augmentation: bool = False):

        # Define input and output paths
        self.images_dir = images_dir
        self.train_data = os.path.join(data_dir, "train.csv")
        self.test_data  = os.path.join(data_dir, "test.csv")
        self.val_data   = os.path.join(data_dir, "val.csv")
        
        
        self.transforms_train = T.Compose([
            T.ToTensor(),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),               # rotaciones leves
            T.RandomVerticalFlip(p=0.2),                # flip vertical (suave)
            RandomGaussianBlur(p=0.3,kernel_size=3,sigma=(0.1, 0.6)),
        ])
        
        self.transforms_test = T.Compose([
            T.ToTensor(),
        ])

        # Create train and test datasets
        self.train_dataset = ImageDataset(
            path_images     = self.images_dir,
            path_data       = self.train_data,
            transforms      = self.transforms_train if augmentation else self.transforms_test,
        )

        self.test_dataset = ImageDataset(
            path_images     = self.images_dir,
            path_data       = self.test_data,
            transforms      = self.transforms_test,
        )
        
        self.val_dataset = ImageDataset(
            path_images     = self.images_dir,
            path_data       = self.val_data,
            transforms      = self.transforms_test,
        )

    def train_dataloader(self, batch_size):
        """
        Create a DataLoader for the training dataset

        Args:
            batch_size (int): Batch size

        Returns:
            DataLoader: Training DataLoader
        """
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    def test_dataloader(self, batch_size):
        """
        Create a DataLoader for the test dataset

        Args:
            batch_size (int): Batch size

        Returns:
            DataLoader: Test DataLoader
        """
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    def val_dataloader(self, batch_size):
        """
        Create a DataLoader for the validation dataset

        Args:
            batch_size (int): Batch size

        Returns:
            DataLoader: Validation DataLoader
        """
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)


import torchvision.transforms as T
import torch
import random


class RandomGaussianBlur(torch.nn.Module):
    """
    Blur leve para simular pequeñas variaciones del detector.
    """
    def __init__(self, p=0.3, kernel_size=3, sigma=(0.1, 0.8)):
        super().__init__()
        self.p = p
        self.blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def forward(self, img):
        if random.random() < self.p:
            img = self.blur(img)
        return img
    
    
if __name__ == "__main__":

    images_dir      = "/mnt/Datos/Master_Camilo/DL/intratumoral-patches"
    data_dir        = "/mnt/Datos/Master_Camilo/DL/data"

    # Create an instance of the Loader class
    loader = Loader(images_dir, data_dir, augmentation=True)

    # Create a DataLoader for the datasets
    train_loader    = loader.train_dataloader(batch_size=2)
    test_loader     = loader.test_dataloader(batch_size=2)
    val_loader      = loader.val_dataloader(batch_size=2)
    
    # Print the number of batches in each DataLoader
    print(f"Number of batches in train DataLoader: {len(train_loader)}")
    print(f"Number of batches in test DataLoader: {len(test_loader)}")
    print(f"Number of batches in val DataLoader: {len(val_loader)}")
    
    # Print the shape of the first batch
    for batch in train_loader:
        im_input, target = batch
        print(f"Input shape: {im_input.shape}")
        print(f"Target shape: {target}")
        break
    # Print the shape of the first batch
    for batch in test_loader:
        im_input, target = batch
        print(f"Input shape: {im_input.shape}")
        print(f"Target shape: {target}")
        break
    # Print the shape of the first batch
    for batch in val_loader:
        im_input, target = batch
        print(f"Input shape: {im_input.shape}")
        print(f"Target shape: {target}")
        break