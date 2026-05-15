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
    nabirds.init(root)
    query_dataset = nabirds(root, 'query', query_transform())
    train_dataset = nabirds(root, 'train', train_transform())
    retrieval_dataset = nabirds(root, 'retrieval', query_transform())
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


class nabirds(Dataset):
    base_folder = 'images/'
    url = 'http://www.vision.caltech.edu/visipedia-data/Food101/Food101.tgz'
    filename = 'Food101.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, mode, transform=None, loader=default_loader):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader

        if mode == 'train':
            self.data = nabirds.TRAIN_DATA
            self.targets = nabirds.TRAIN_TARGETS
        elif mode == 'query':
            self.data = nabirds.QUERY_DATA
            self.targets = nabirds.QUERY_TARGETS
        elif mode == 'retrieval':
            self.data = nabirds.RETRIEVAL_DATA
            self.targets = nabirds.RETRIEVAL_TARGETS
        else:
            raise ValueError(r'Invalid arguments: mode, can\'t load dataset!')



    @staticmethod
    def init(root):
        train_data = pd.read_csv(os.path.join(root, 'train.txt'), names=['filepath'])
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
        nabirds.QUERY_DATA = test_data['filepath'].to_numpy()
        nabirds.QUERY_TARGETS = encode_onehot((test_data['target'] - 1).tolist(), 555)

        nabirds.TRAIN_DATA = train_data['filepath'].to_numpy()
        nabirds.TRAIN_TARGETS = encode_onehot((train_data['target'] - 1).tolist(), 555)

        nabirds.RETRIEVAL_DATA = train_data['filepath'].to_numpy()
        nabirds.RETRIEVAL_TARGETS = encode_onehot((train_data['target'] - 1).tolist(), 555)


    def get_onehot_targets(self):
        return torch.from_numpy(self.targets).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.data[idx])).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[idx], idx


class nabirds_UC(Dataset):
    base_folder = 'images/'
    url = 'http://www.vision.caltech.edu/visipedia-data/Food101/Food101.tgz'
    filename = 'Food101.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, mode, transform=None, loader=default_loader):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader

        if mode == 'train':
            self.data = nabirds.TRAIN_DATA
            self.targets = nabirds.TRAIN_TARGETS
        elif mode == 'query':
            self.data = nabirds.QUERY_DATA
            self.targets = nabirds.QUERY_TARGETS
        elif mode == 'retrieval':
            self.data = nabirds.RETRIEVAL_DATA
            self.targets = nabirds.RETRIEVAL_TARGETS
        else:
            raise ValueError(r'Invalid arguments: mode, can\'t load dataset!')



    @staticmethod
    def init(root):
        train_data = pd.read_csv(os.path.join(root, 'train.txt'), names=['filepath'])
        train_data['category'] = train_data['filepath'].apply(lambda x: x.split('/')[0])
        train_data['target'] = train_data['category'].astype('category').cat.codes + 1
        train_data['filepath'] = train_data['filepath']
        test_data = pd.read_csv(os.path.join(root, 'test.txt'), names=['filepath'])
        test_data['category'] = test_data['filepath'].apply(lambda x: x.split('/')[0])
        test_data['target'] = test_data['category'].astype('category').cat.codes + 1
        test_data['filepath'] = test_data['filepath']


        # Split dataset
        nabirds.QUERY_DATA = test_data['filepath'].to_numpy()
        nabirds.QUERY_TARGETS = encode_onehot((test_data['target'] - 1).tolist(), 555)

        nabirds.TRAIN_DATA = train_data['filepath'].to_numpy()
        nabirds.TRAIN_TARGETS = encode_onehot((train_data['target'] - 1).tolist(), 555)

        nabirds.RETRIEVAL_DATA = train_data['filepath'].to_numpy()
        nabirds.RETRIEVAL_TARGETS = encode_onehot((train_data['target'] - 1).tolist(), 555)


    def get_onehot_targets(self):
        return torch.from_numpy(self.targets).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.data[idx])).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[idx], idx