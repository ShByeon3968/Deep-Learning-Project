from torch.utils.data import Dataset
import cv2
import random
import numpy as np
import os
import torchvision.transforms.functional as TF
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

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
        if self.mode == 'train':
            return len(self.train_image_list)
        elif self.mode == 'val':
            return len(self.val_image_list)
        else:
            return len(self.test_image_list)

    def __getitem__(self, idx):
        if self.mode == 'train':
            image = self.train_image_list[idx]
            clean = self._add_transform(image)
            noisy, _ = self._add_gaussian_noise(image)
        elif self.mode == 'val':
            clean = self.val_image_list[idx]
            noisy, _ = self._add_gaussian_noise(image)  
            noisy,clean = self._to_tensor(noisy), self._to_tensor(clean)
        else:
            clean = self.test_image_list[idx]
            noisy, _ = self._add_gaussian_noise(image)
            noisy,clean = self._to_tensor(noisy), self._to_tensor(clean)
        return noisy, clean
    
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

    def _add_gaussian_noise(self,image:np.ndarray,max_std:int=55):
        image = image.astype(np.float32)
        std = random.uniform(0,max_std)
        noise = np.random.normal(0,std,image.shape)
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image,0,255).astype(np.uint8)
        return noisy_image, std

    def _to_tensor(self, image, resize=(256, 256)):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, resize)  # <-- 모든 이미지를 동일한 크기로 맞춤
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        return torch.from_numpy(image).float()
    
    def _add_transform(self,image):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256,256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor()
        ])
        return transform(image)

def get_train_loader(path:str='./data/images',batch_size:int=32):
    dataset = DenoiseDataset(folder_path=path,mode='train')
    data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    return data_loader
def get_val_loader(path:str='./data/images',batch_size:int=32):
    dataset = DenoiseDataset(folder_path=path,mode='val')
    data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
    return data_loader


