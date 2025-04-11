from torch.utils.data import Dataset
import cv2
import random
import numpy as np
import os
import torchvision.transforms.functional as TF
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from glob import glob
import tensorflow as tf
from PIL import Image
import io


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
            clean = self.train_image_list[idx]
            clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
            clean = cv2.resize(clean, (256, 256))

            # augmentation 적용 (clean 이미지에만, numpy 기준)
            clean = self._apply_transform(clean)

            # noisy 생성
            noisy, _ = self._add_gaussian_noise(clean)

            # tensor로 변환
            clean_tensor = self._to_tensor_from_numpy(clean)
            noisy_tensor = self._to_tensor_from_numpy(noisy)
            return noisy_tensor, clean_tensor

        elif self.mode == 'val':
            clean = self.val_image_list[idx]
            clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
            clean = cv2.resize(clean, (256, 256))
            noisy, _ = self._add_gaussian_noise(clean)
            return self._to_tensor_from_numpy(noisy), self._to_tensor_from_numpy(clean)

        else:
            clean = self.test_image_list[idx]
            clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
            clean = cv2.resize(clean, (256, 256))
            noisy, _ = self._add_gaussian_noise(clean)
            return self._to_tensor_from_numpy(noisy), self._to_tensor_from_numpy(clean)

    
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

    def _to_tensor_from_numpy(self, image, resize=(256, 256)):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, resize)  # <-- 모든 이미지를 동일한 크기로 맞춤
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        return torch.from_numpy(image).float()

    def _tensor_permute(self,image:torch.Tensor):
        image = image.permute(2,0,1)  # HWC -> CHW
        return image
    
    def _apply_transform(self, image: np.ndarray):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor()
        ])
        return (transform(image) * 255.0).permute(1, 2, 0).numpy().astype(np.uint8)  # back to numpy image


patch_size, stride = 40, 10
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
batch_size = 128

class DnCnnDataset(object):
    def __init__(self, images_dir, patch_size,
                 gaussian_noise_level, downsampling_factor, jpeg_quality,
                 use_fast_loader=False):
        self.image_files = sorted(glob(images_dir + '/*.jpg') + glob(images_dir + '/*.png'))
        self.patch_size = patch_size
        self.gaussian_noise_level = gaussian_noise_level
        self.downsampling_factor = downsampling_factor
        self.jpeg_quality = jpeg_quality
        self.use_fast_loader = use_fast_loader

    def __getitem__(self, idx):
        if self.use_fast_loader:
            clean_image = tf.read_file(self.image_files[idx])
            clean_image = tf.image.decode_jpeg(clean_image, channels=3)
            clean_image = Image.fromarray(clean_image.numpy())
        else:
            clean_image = Image.open(self.image_files[idx]).convert('RGB')

        # randomly crop patch from training set
        crop_x = random.randint(0, clean_image.width - self.patch_size)
        crop_y = random.randint(0, clean_image.height - self.patch_size)
        clean_image = clean_image.crop((crop_x, crop_y, crop_x + self.patch_size, crop_y + self.patch_size))
        
        # 데이터 증강
        clean_image = self._apply_transform(clean_image)

        noisy_image = clean_image.copy()
        gaussian_noise = np.zeros((clean_image.height, clean_image.width, 3), dtype=np.float32)

        # additive gaussian noise
        if self.gaussian_noise_level is not None:
            if len(self.gaussian_noise_level) == 1:
                sigma = self.gaussian_noise_level[0]
            else:
                sigma = random.randint(self.gaussian_noise_level[0], self.gaussian_noise_level[1])
            gaussian_noise += np.random.normal(0.0, sigma, (clean_image.height, clean_image.width, 3)).astype(np.float32)

        # downsampling
        if self.downsampling_factor is not None:
            if len(self.downsampling_factor) == 1:
                downsampling_factor = self.downsampling_factor[0]
            else:
                downsampling_factor = random.randint(self.downsampling_factor[0], self.downsampling_factor[1])

            noisy_image = noisy_image.resize((self.patch_size // downsampling_factor,
                                              self.patch_size // downsampling_factor),
                                             resample=Image.BICUBIC)
            noisy_image = noisy_image.resize((self.patch_size, self.patch_size), resample=Image.BICUBIC)

        # additive jpeg noise
        if self.jpeg_quality is not None:
            if len(self.jpeg_quality) == 1:
                quality = self.jpeg_quality[0]
            else:
                quality = random.randint(self.jpeg_quality[0], self.jpeg_quality[1])
            buffer = io.BytesIO()
            noisy_image.save(buffer, format='jpeg', quality=quality)
            noisy_image = Image.open(buffer)

        clean_image = np.array(clean_image).astype(np.float32)
        noisy_image = np.array(noisy_image).astype(np.float32)
        noisy_image += gaussian_noise

        input = np.transpose(noisy_image, axes=[2, 0, 1])
        label = np.transpose(clean_image, axes=[2, 0, 1])

        # normalization
        input /= 255.0
        label /= 255.0

        return input, label

    def __len__(self):
        return len(self.image_files)
    
    def _apply_transform(self,image:Image) -> Image:
        if random.random() < 0.5:
            image = TF.hflip(image)
        if random.random() < 0.5:
            image = TF.vflip(image)
        if random.random() < 0.5:
            angle = random.choice([90, 180, 270])
            image = TF.rotate(image, angle)
        return image
        

def get_train_loader(path:str='./data/images/train',batch_size:int=32):
    dataset = DnCnnDataset(path,
                       patch_size=50,gaussian_noise_level=[0,55],downsampling_factor=[1,4],
                       jpeg_quality=[5,99],use_fast_loader=False)
    data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    return data_loader
def get_val_loader(path:str='./data/images/val',batch_size:int=32):
    dataset = DnCnnDataset(path,
                       patch_size=50,gaussian_noise_level=[0,55],downsampling_factor=[1,4],
                       jpeg_quality=[5,99],use_fast_loader=False)
    data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
    return data_loader
