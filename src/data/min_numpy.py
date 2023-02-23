import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class MINDataset(Dataset):
    def __init__(self, min_path, transform=None):
        with open(min_path, 'rb') as f:
            min_raw_data = pickle.load(f)
        
        self.image_data = min_raw_data['image_data']
        self.label_data = np.zeros(len(self.image_data))
        for i, class_label in enumerate(min_raw_data['class_dict'].keys()):
            idxs = min_raw_data['class_dict'][class_label]
            self.label_data[idxs] = i

        if transform is None:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.4707, 0.4495, 0.4026],
                            std=[0.2843, 0.2752, 0.2903])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.label_data)

    def __getitem__(self, idx):
        img = self.image_data[idx]
        img = self.transform(img)

        label = torch.tensor(self.label_data[idx]).long()
        return {
            'image': img,
            'label': label
        }