import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2
import torchvision.transforms.functional as TF
import pickle
import random
import torch
import numpy as np

PATH_DEV_DS_VALAIS = './datasets/ValaisCD/dev'
PATH_TRAIN_DS_VALAIS = './datasets/ValaisCD/train'
PATH_TEST_DS_VALAIS = './datasets/ValaisCD/test'
PATH_VAL_DS_VALAIS = './datasets/ValaisCD/val'


class Valais_Dataset(Dataset):
    def __init__(self, type, cropsize=256, name=False, probability_maps = False, given_list=None):

        if type == 'train':
            folderpath = PATH_TRAIN_DS_VALAIS
        elif type == 'test':
            folderpath = PATH_TEST_DS_VALAIS
        elif type == 'val':
            folderpath = PATH_VAL_DS_VALAIS
        elif type == 'dev':
            folderpath = PATH_DEV_DS_VALAIS

        self.with_probmaps = False
        self.with_priors = False
        self.name = name
        self.cropsize=cropsize

        if given_list!=None:
            with open(os.path.join(folderpath, given_list), 'rb') as f:
                list_changed = pickle.load(f)
            list_changed=[name.split('/')[-1] for name in list_changed]
            self.im1 = [os.path.join(folderpath,"2017",im) for im in list_changed] 
            self.im2 = [os.path.join(folderpath,"2023",im) for im in list_changed]
            self.lab = [os.path.join(folderpath,"labels",im) for im in list_changed]
            if probability_maps:
                self.with_probmaps = True
                self.probmaps = [os.path.join(folderpath,"probability",im[:-3]+'npy') for im in list_changed]
        else:
            self.im1 = [os.path.join(folderpath,"2017",im) for im in sorted(os.listdir(os.path.join(folderpath,"2017"))) if im.endswith(".tif")] 
            self.im2 = [os.path.join(folderpath,"2023",im) for im in sorted(os.listdir(os.path.join(folderpath,"2023"))) if im.endswith(".tif")]
            self.lab = [os.path.join(folderpath,"labels",im) for im in sorted(os.listdir(os.path.join(folderpath,"labels"))) if im.endswith(".tif")]
            if probability_maps:
                self.with_probmaps = True
                self.probmaps = [os.path.join(folderpath,"probability",im) for im in sorted(os.listdir(os.path.join(folderpath,"probability"))) if im.endswith(".npy")]

    def transform(self, im1, im2, lab):

        # Random crop
        i, j, h, w = v2.RandomCrop.get_params(
            im1, output_size=(self.cropsize, self.cropsize))
        im1 = TF.crop(im1, i, j, h, w)
        im2 = TF.crop(im2, i, j, h, w)
        lab = TF.crop(lab, i, j, h, w)

        # Transform to tensor
        im1 = TF.to_tensor(im1)
        im2 = TF.to_tensor(im2)
        #lab = TF.to_tensor(lab)
        lab = torch.tensor(np.array(lab), dtype=torch.float32)

        return im1, im2, lab


    def __len__(self):
        return len(self.im1)
    
    def __getitem__(self, idx):
        image1 = Image.open(self.im1[idx])
        image2 = Image.open(self.im2[idx])
        changemap = Image.open(self.lab[idx])
        image1, image2, changemap = self.transform(image1, image2, changemap)
        if self.name:
            return image1, image2, changemap.unsqueeze(0), self.im1[idx]
        elif self.with_probmaps:
            with open(self.probmaps[idx], 'rb') as f:
                probability = np.load(f)
            probability = torch.from_numpy(probability)
            probability = torch.unsqueeze(probability, dim=0).float()
            return image1, image2, changemap, probability
        elif self.with_priors:
            prior = torch.load(self.priors[idx])
            prior = torch.unsqueeze(prior, dim=0)
            prior = torch.nan_to_num(prior, nan=0.0)
            return image1, image2, changemap, prior
        else:
            return image1, image2, changemap.unsqueeze(0)
        

