#import msadapter.pytorch as torch
#import mindspore as ms
#import numpy as np
import torch

import os
import pandas as pd
# from msadapter.torchvision.datasets.folder import default_loader
# from msadapter.torchvision.datasets.utils import download_url
# from msadapter.pytorch.utils.data import Dataset
# from msadapter.pytorch.utils.data.dataloader import DataLoader
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from PIL import Image, ImageFile

from data.transform import encode_onehot
from data.transform import train_transform, query_transform

def load_data(root, batch_size, num_workers):
    Food500.init(root)
    query_dataset = Food500(root, 'query', query_transform())
    train_dataset = Food500(root, 'train', train_transform())
    retrieval_dataset = Food500(root, 'retrieval', query_transform())
    print(len(query_dataset))
    print(len(train_dataset))

    query_dataloader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    retrieval_dataloader = DataLoader(
        retrieval_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )

    return query_dataloader, train_dataloader, retrieval_dataloader


class Food500(Dataset):

    def __init__(self, root, mode, transform=None, loader=default_loader):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader

        if mode == 'train':
            self.data = Food500.TRAIN_DATA
            self.targets = Food500.TRAIN_TARGETS
        elif mode == 'query':
            self.data = Food500.QUERY_DATA
            self.targets = Food500.QUERY_TARGETS
        elif mode == 'retrieval':
            self.data = Food500.RETRIEVAL_DATA
            self.targets = Food500.RETRIEVAL_TARGETS
        else:
            raise ValueError(r'Invalid arguments: mode, can\'t load dataset!')



    @staticmethod
    def init(root):
        train_data = pd.read_csv(os.path.join(root, 'train_balanced.txt'), names=['filepath'])
        train_data['category'] = train_data['filepath'].apply(lambda x: x.split('/')[0])
        train_data['target'] = train_data['category'].astype('category').cat.codes + 1
        train_data['filepath'] = train_data['filepath']
        test_data = pd.read_csv(os.path.join(root, 'test.txt'),names=['filepath'])
        test_data['category'] = test_data['filepath'].apply(lambda x: x.split('/')[0])
        test_data['target'] = test_data['category'].astype('category').cat.codes + 1
        test_data['filepath'] = test_data['filepath']

        # print(test_data.shape)
        # print(len(test_data))
        # print(test_data['filepath'][0])
        # print(test_data['img_id'][0])
        # print(test_data['target'][0])
        # print(test_data['is_training_img'][0])
        # print(train_data.shape)
        # print(train_data[0])


        # Split dataset
        Food500.QUERY_DATA = test_data['filepath'].to_numpy()
        Food500.QUERY_TARGETS = encode_onehot((test_data['target'] - 1).tolist(), 500)

        Food500.TRAIN_DATA = train_data['filepath'].to_numpy()
        Food500.TRAIN_TARGETS = encode_onehot((train_data['target'] - 1).tolist(), 500)

        Food500.RETRIEVAL_DATA = train_data['filepath'].to_numpy()
        Food500.RETRIEVAL_TARGETS = encode_onehot((train_data['target'] - 1).tolist(), 500)


    def get_onehot_targets(self):
        return torch.from_numpy(self.targets).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.data[idx])).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[idx], idx


class Food500_UC(Dataset):

    def __init__(self, root, mode, transform=None, loader=default_loader):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader

        if mode == 'train':
            self.data = Food500.TRAIN_DATA
            self.targets = Food500.TRAIN_TARGETS
        elif mode == 'query':
            self.data = Food500.QUERY_DATA
            self.targets = Food500.QUERY_TARGETS
        elif mode == 'retrieval':
            self.data = Food500.RETRIEVAL_DATA
            self.targets = Food500.RETRIEVAL_TARGETS
        else:
            raise ValueError(r'Invalid arguments: mode, can\'t load dataset!')



    @staticmethod
    def init(root):
        train_data = pd.read_csv(os.path.join(root, 'train_balanced.txt'), names=['filepath'])
        train_data['category'] = train_data['filepath'].apply(lambda x: x.split('/')[0])
        train_data['target'] = train_data['category'].astype('category').cat.codes + 1
        train_data['filepath'] = train_data['filepath']
        test_data = pd.read_csv(os.path.join(root, 'test.txt'), names=['filepath'])
        test_data['category'] = test_data['filepath'].apply(lambda x: x.split('/')[0])
        test_data['target'] = test_data['category'].astype('category').cat.codes + 1
        test_data['filepath'] = test_data['filepath']


        # Split dataset
        Food500.QUERY_DATA = test_data['filepath'].to_numpy()
        Food500.QUERY_TARGETS = encode_onehot((test_data['target'] - 1).tolist(), 500)

        Food500.TRAIN_DATA = train_data['filepath'].to_numpy()
        Food500.TRAIN_TARGETS = encode_onehot((train_data['target'] - 1).tolist(), 500)

        Food500.RETRIEVAL_DATA = train_data['filepath'].to_numpy()
        Food500.RETRIEVAL_TARGETS = encode_onehot((train_data['target'] - 1).tolist(), 500)


    def get_onehot_targets(self):
        return torch.from_numpy(self.targets).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.data[idx])).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[idx], idx