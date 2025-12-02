import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2
import torchvision.transforms.functional as TF
import pickle
import random
import numpy as np

PATH_TRAIN_DS_HRSCD = './datasets/HRSCD_clean/train'
PATH_TEST_DS_HRSCD = './datasets/HRSCD_clean/test'
PATH_VAL_DS_HRSCD = './datasets/HRSCD_clean/val'

class HRSCD_Dataset(Dataset):
    def __init__(self, type, cropsize=256, refined_labels=False, name=False, probability_maps = False, given_list = None):

        if type == 'train':
            folderpath = PATH_TRAIN_DS_HRSCD
        elif type == 'test':
            folderpath = PATH_TEST_DS_HRSCD
        elif type == 'val':
            folderpath = PATH_VAL_DS_HRSCD
        
        self.with_probmaps = False

        if given_list!=None: #test_imgs_filtered.pkl'
            with open(os.path.join(folderpath, given_list), 'rb') as f:
                list_changed = pickle.load(f)
                list_changed = [i[0] for i in list_changed]
            list_changed=[name.split('/')[-1] for name in list_changed]
            self.im1 = [os.path.join(folderpath,"images1",im) for im in list_changed] 
            self.im2 = [os.path.join(folderpath,"images2",im) for im in list_changed]
            if refined_labels:
                self.lab = [os.path.join(folderpath,"labels_map",im) for im in list_changed]
            else:
                self.lab = [os.path.join(folderpath,"labels",im) for im in list_changed]
            if probability_maps:
                self.with_probmaps = True
                self.probmaps = [os.path.join(folderpath,"probability",im[:-3]+'npy') for im in list_changed]
        else:
            self.im1 = [os.path.join(folderpath,"images1",im) for im in sorted(os.listdir(os.path.join(folderpath,"images1"))) if im.endswith(".tif")] 
            self.im2 = [os.path.join(folderpath,"images2",im) for im in sorted(os.listdir(os.path.join(folderpath,"images2"))) if im.endswith(".tif")]
            if refined_labels:
                self.lab = [os.path.join(folderpath,"labels_map",im) for im in sorted(os.listdir(os.path.join(folderpath,"labels_map"))) if im.endswith(".tif")]
            else:
                self.lab = [os.path.join(folderpath,"labels",im) for im in sorted(os.listdir(os.path.join(folderpath,"labels"))) if im.endswith(".tif")]
            
            if probability_maps:
                self.with_probmaps = True
                self.probmaps = [os.path.join(folderpath,"probability",im) for im in sorted(os.listdir(os.path.join(folderpath,"probability"))) if im.endswith(".npy")]

        self.cropsize=cropsize
        self.name=name

    def transform(self, im1, im2, lab):

        # Transform to tensor
        im1 = TF.to_tensor(im1)
        im2 = TF.to_tensor(im2)
        lab = TF.to_tensor(lab)

        return im1, im2, lab


    def __len__(self):
        return len(self.im1)
    
    def __getitem__(self, idx):
        image1 = Image.open(self.im1[idx])
        image2 = Image.open(self.im2[idx])
        changemap = Image.open(self.lab[idx])
        image1, image2, changemap = self.transform(image1, image2, changemap)

        if self.name:
            return image1, image2, changemap, self.im1[idx]
        elif self.with_probmaps:
            with open(self.probmaps[idx], 'rb') as f:
                probability = np.load(f)
            probability = torch.from_numpy(probability)
            probability = torch.unsqueeze(probability, dim=0).float()
            return image1, image2, changemap, 2*probability
        else:
            return image1, image2, changemap