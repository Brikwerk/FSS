import os

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torch.nn import functional as F

import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels

from src.data.fs_sampler import FewShotSampler
from src.data.chestx import ChestX
from src.data.isic import ISICDataset


class FewShotLoader():
    
    def __init__(self, dataset, loader_type, shots, ways, query) -> None:
        self.dataset = dataset
        self.loader_type = loader_type
        self.shots = shots
        self.ways = ways
        self.query = query
        self.loader = self.construct_loader(
            dataset, loader_type, shots, ways, query)

    def construct_loader(self, dataset, loader_type, shots, ways, query):
        if self.loader_type == "L2L":
            l2l_dataset = l2l.data.MetaDataset(dataset)
            test_transforms = [
                NWays(l2l_dataset, ways),
                KShots(l2l_dataset, shots + query),
                LoadData(l2l_dataset),
                RemapLabels(l2l_dataset),
            ]
            test_tasks = l2l.data.TaskDataset(l2l_dataset,
                                                task_transforms=test_transforms,
                                                num_tasks=2000)
            return DataLoader(test_tasks, pin_memory=True, shuffle=True)
        else:
            return FewShotSampler(dataset, ways, shots, query)
    
    def get_episode(self):
        if self.loader_type == "L2L":
            return next(iter(self.loader))
        else:
            return self.loader.get_batch()


def load_datasets(root_path: str, img_size: int, shots: int, ways: int,
                  query: int, dataset_subset: list = None) -> list:
    datasets = []

    # ImageNet-1K Mean/Std. for general reuse in datasets
    mean = torch.tensor([0.4707, 0.4495, 0.4026])
    std = torch.tensor([0.2843, 0.2752, 0.2903])

    if dataset_subset is None or "min" in dataset_subset:
        min_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std), # ImageNet Dataset and other natural image datasets
            T.Resize(size=(img_size, img_size)),
        ])
        min_dataset = torchvision.datasets.ImageFolder(
            os.path.join(root_path, 'mini-imagenet', 'test'), transform=min_transform)
        min_loader = FewShotLoader(min_dataset, "L2L", shots, ways, query)
        datasets.append((min_loader, "MIN"))
        print("MIN Loaded")

    # HEp Dataset
    if dataset_subset is None or "hep" in dataset_subset:
        hep_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.7940, 0.7940, 0.7940], std=[0.1920, 0.1920, 0.1920]), # HEp-2 Dataset
            T.Resize(size=(img_size, img_size)),
        ])
        hep_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(root_path, 'HEp-Dataset'), transform=hep_transform)
        hep_loader = FewShotLoader(hep_dataset, "L2L", shots, ways, query)
        datasets.append((hep_loader, "HEp-2"))
        print("HEp-2 Loaded")

    # BCCD WBC Dataset
    if dataset_subset is None or "bccd" in dataset_subset:
        wbc_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.6659, 0.6028, 0.7932], std=[0.1221, 0.1698, 0.0543]), # BCCD Dataset
            T.Resize(size=(img_size, img_size)),
        ])
        wbc_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(root_path, 'wbc-aug'), transform=wbc_transform)
        wbc_loader = FewShotLoader(wbc_dataset, "L2L", shots, ways, query)
        datasets.append((wbc_loader, "BCCD"))
        print("BCCD Loaded")

    # NHS Chest X-Ray Dataset
    if dataset_subset is None or "chestx" in dataset_subset:
        chestx_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.4920, 0.4920, 0.4920], std=[0.2288, 0.2288, 0.2288]), # ChestX
            T.Resize(size=(img_size, img_size)),
        ])
        chestx_dataset = ChestX(os.path.join(
            root_path, "chestx"), transform=chestx_transform)
        chestx_loader = FewShotLoader(chestx_dataset, "FSS", shots, ways, query)
        datasets.append((chestx_loader, "ChestX"))
        print("ChestX Loaded")

    # Skin Lesion Dataset
    if dataset_subset is None or "isic" in dataset_subset:
        isic_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.7635, 0.5461, 0.5705], std=[0.0891, 0.1179, 0.1325]), # ISIC
            T.Resize(size=(img_size, img_size)),
        ])
        isic_dataset = ISICDataset(
            os.path.join(root_path, "isic2018"), transform=isic_transform)
        isic_loader = FewShotLoader(isic_dataset, "FSS", shots, ways, query)
        datasets.append((isic_loader, "ISIC"))
        print("ISIC Loaded")

    # Eurosat Satellite Image Dataset
    if dataset_subset is None or "eurosat" in dataset_subset:
        eurosat_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.3444, 0.3803, 0.4078], std=[0.0884, 0.0621, 0.0521]), # EuroSat Dataset
            T.Resize(size=(img_size, img_size)),
        ])
        eurosat_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(root_path, 'eurosat'), transform=eurosat_transform)
        eurosat_loader = FewShotLoader(eurosat_dataset, "L2L", shots, ways, query)
        datasets.append((eurosat_loader, "EuroSat"))
        print("EuroSat Loaded")

    # Plant Disease Dataset
    if dataset_subset is None or "plant" in dataset_subset:
        plant_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.4662, 0.4888, 0.4101], std=[0.1707, 0.1438, 0.1875]), # Plant Disease Dataset
            T.Resize(size=(img_size, img_size)),
        ])
        plant_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(root_path, 'plant_disease', 'train'), transform=plant_transform)
        plant_loader = FewShotLoader(plant_dataset, "L2L", shots, ways, query)
        datasets.append((plant_loader, "Plant Disease"))
        print("Plant Disease Loaded")

    # IKEA Few-Shot Dataset
    if dataset_subset is None or "ikea" in dataset_subset:
        ikea_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.7073, 0.6915, 0.6744], std=[0.2182, 0.2230, 0.2312]), # Plant Disease Dataset
            T.Resize(size=(img_size, img_size)),
        ])
        ikea_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(root_path, 'ikea'), transform=ikea_transform)
        ikea_loader = FewShotLoader(ikea_dataset, "L2L", shots, ways, query)
        datasets.append((ikea_loader, "IKEA-FS"))
        print("IKEA Loaded")

    return datasets