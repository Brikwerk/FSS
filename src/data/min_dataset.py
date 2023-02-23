import pickle
from src.data.fs_sampler import FewShotSampler
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np


class MINDataset(Dataset):
    def __init__(self, min_path, img_size=84):
        with open(min_path, 'rb') as f:
            min_raw_data = pickle.load(f)
        
        self.image_data = min_raw_data['image_data']
        self.label_data = np.zeros(len(self.image_data))
        for i, class_label in enumerate(min_raw_data['class_dict'].keys()):
            idxs = min_raw_data['class_dict'][class_label]
            self.label_data[idxs] = i
        
        self.samples = []
        for i in range(len(self.label_data)):
            label = self.label_data[i]
            self.samples.append(('', label))

        self.transform = T.Compose([
            T.ToTensor(),
            # transforms.Normalize(mean=[0.4707, 0.4495, 0.4026],
            #                      std=[0.2843, 0.2752, 0.2903])
            T.RandomResizedCrop((img_size, img_size))
        ])

    def __len__(self):
        return len(self.label_data)

    def __getitem__(self, idx):
        img = self.image_data[idx]
        img = self.transform(img)

        label = torch.tensor(self.label_data[idx]).float()
        return img, label
