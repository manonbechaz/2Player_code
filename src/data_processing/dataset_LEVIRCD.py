import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torchvision.transforms.functional as TF
import numpy as np
import random
import rasterio

PATH_TRAIN_LEVIR = './datasets/LEVIRCD/train'
PATH_TEST_LEVIR = './datasets/LEVIRCD/test'
PATH_VAL_LEVIR = './datasets/LEVIRCD/val'

class LEVIRCD_Dataset(Dataset):
    def __init__(self, cropsize=256, type = 'train', name=False, probability_maps=False, augmentations=False):

        self.augmentations = augmentations
        self.with_priors = False
        self.name = name

        if type == 'train':
            path = PATH_TRAIN_LEVIR
        elif type == 'test':
            path = PATH_TEST_LEVIR
        elif type == 'val':
            path = PATH_VAL_LEVIR

        self.im1 = [os.path.join(path,'A',im) for im in sorted(os.listdir(os.path.join(path,'A'))) if im.endswith(".png")] 
        self.im2 = [os.path.join(path,'B',im) for im in sorted(os.listdir(os.path.join(path,'B'))) if im.endswith(".png")] 
        self.lab = [os.path.join(path,'label',im) for im in sorted(os.listdir(os.path.join(path,'label'))) if im.endswith(".png")] 
        if probability_maps:
            self.with_probmaps = True
            self.probmaps = [os.path.join(path,"probability",im) for im in sorted(os.listdir(os.path.join(path,"probability"))) if im.endswith(".npy")]
        else: 
            self.with_probmaps = False

        self.cropsize=cropsize

        if self.augmentations:
            self.shape_transform = v2.Compose([
                v2.RandomCrop(self.cropsize),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),  # multiples of 90°
                ])
                    
            self.color_transform = v2.Compose([
                v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                v2.GaussianBlur(3)
            ])
        else:
            self.shape_transform = v2.Compose([
                v2.RandomCrop(self.cropsize)  # multiples of 90°
                ])
            self.color_transform = lambda x: x

    def transform(self, im1, im2, lab, probmaps=None):

        # Random crop
        i, j, h, w = v2.RandomCrop.get_params(
            im1, output_size=(self.cropsize, self.cropsize))
        im1 = TF.crop(im1, i, j, h, w)
        im2 = TF.crop(im2, i, j, h, w)
        lab = TF.crop(lab, i, j, h, w)
        # Transform to tensor
        im1 = TF.to_tensor(im1)
        im2 = TF.to_tensor(im2)
        lab = TF.to_tensor(lab)

        if probmaps is not None:
            probmaps = TF.crop(probmaps, i, j, h, w)
            return im1,im2,lab, probmaps
        else:
            return im1, im2, lab


    def __len__(self):
        return len(self.im1)
    
    def __getitem__(self, idx):
        '''image1 = Image.open(self.im1[idx])
        image2 = Image.open(self.im2[idx])
        masks = Image.open(self.lab[idx])'''
        with rasterio.open(self.im1[idx]) as src:
            image1 = src.read()
        with rasterio.open(self.im2[idx]) as src:
            image2 = src.read()
        with rasterio.open(self.lab[idx]) as src:
            label = src.read(1)

        image1 = torch.tensor(image1, dtype=torch.float32)/ 255 # Normalize uint16
        image2 = torch.tensor(image2, dtype=torch.float32)/ 255
        label = torch.tensor(label, dtype=torch.float32)/ 255

        seed = torch.randint(0, 1_000_000, (1,)).item()
        random.seed(seed)
        if self.augmentations:
            k = random.choice([0, 1, 2, 3])  # number of 90° rotations
        else:
            k=0
        torch.manual_seed(seed); image1 = self.shape_transform(image1)
        image1 = torch.rot90(image1, k, dims=(-2, -1))  # rotate along H, W
        torch.manual_seed(seed); image2 = self.shape_transform(image2)
        image2 = torch.rot90(image2, k, dims=(-2, -1))  # rotate along H, W
        torch.manual_seed(seed); label = self.shape_transform(label)
        label = torch.rot90(label, k, dims=(-2, -1))
        image1 = self.color_transform(image1)
        image2 = self.color_transform(image2)

        if self.with_probmaps:
            with open(self.probmaps[idx], 'rb') as f:
                probability = np.load(f)
            probability = torch.from_numpy(probability)
            probability = torch.unsqueeze(probability, dim=0).float()
            torch.manual_seed(seed); probability = self.shape_transform(probability)
            probability = torch.rot90(probability, k, dims=(-2, -1))
            return image1, image2, label, 2*probability
        elif self.name:
            return image1, image2, label, self.im1[idx]
        else:
            return image1, image2, label