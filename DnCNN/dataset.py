from torch.utils.data import Dataset
import cv2
import random
import numpy as np
import os


class DenoiseDataset(Dataset):
    def __init__(self,folder_path:str,mode:str='train'):
        super().__init__()
        self.folder_path = folder_path
        self.mode = mode
        # image_list
        self.train_image_list = []
        self.val_image_list = []
        self.test_image_list = []
        self._image_load()
    def __len__(self):
        return len(self.train_image_list)

    def __getitem__(self, idx):
        if self.mode == 'train':
            image = self.train_image_list[idx]
            image, _ = self._add_gaussian_noise(image)
            gt = self.train_image_list[idx]
        elif self.mode == 'val':
            image = self.val_image_list[idx]
            image, _ = self._add_gaussian_noise(image)
            gt = self.val_image_list[idx]       
        else:
            image = self.val_image_list[idx]
            image,_ = self._add_gaussian_noise(image)
            gt = self.test_image_list[idx]
        return image, gt
    
    def _image_load(self):
        image_list = []
        for folder_name in os.listdir(self.folder_path):
            if folder_name == 'train':
                image_list = [os.path.join(self.folder_path,folder_name,img_name) for img_name in os.listdir(os.path.join(self.folder_path,folder_name)) 
                                         if img_name.lower().endswith('.jpg')]
                self.train_image_list = [cv2.imread(img) for img in image_list]
            elif folder_name =='val':
                image_list = [os.path.join(self.folder_path,folder_name,img_name) for img_name in os.listdir(os.path.join(self.folder_path,folder_name)) 
                                         if img_name.lower().endswith('.jpg')]
                self.val_image_list = [cv2.imread(img) for img in image_list]
            else:
                image_list = [os.path.join(self.folder_path,folder_name,img_name) for img_name in os.listdir(os.path.join(self.folder_path,folder_name)) 
                                         if img_name.lower().endswith('.jpg')]
                self.test_image_list = [cv2.imread(img) for img in image_list]

    def _add_gaussian_noise(self,image:np.ndarray,max_std:str=55):
        image = image.astype(np.float32)
        std = random.uniform(0,max_std)
        noise = np.random.normal(0,std,image.shape)
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image,0,255).astype(np.uint8)
        return noisy_image, std

