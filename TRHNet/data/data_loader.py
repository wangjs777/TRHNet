import os
import numpy as np
#import msadapter.pytorch as torch
import torch
#import mindspore as ms
from PIL import Image
from tqdm import tqdm
# from msadapter.pytorch.utils.data.dataloader import DataLoader
# from msadapter.pytorch.utils.data.dataset import Dataset
# from msadapter.torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image, ImageFile

import data.food101 as food101
import data.cub_2011 as cub2011
import data.aircraft as aircraft
import data.nabirds as nabirds
import data.vegfru as vegfru
import data.food172 as food172
import data.food256 as food256
import data.food500 as food500
import data.food2k as food2k
from data.transform import train_transform, encode_onehot

ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_data(dataset, root, batch_size, num_workers):
    """
    Load dataset.

    Args
        dataset(str): Dataset name.
        root(str): Path of dataset.
        num_query(int): Number of query data points.
        num_train(int): Number of training data points.
        num_workers(int): Number of loading data threads.

    Returns
        query_dataloader, train_dataloader, retrieval_dataloader(torch.utils.data.DataLoader): Data loader.
    """
    if dataset == 'cub-2011':
        query_dataloader, train_dataloader, retrieval_dataloader = cub2011.load_data(root,
                                      batch_size,
                                      num_workers,
                                      )
    elif dataset == 'food-101':
        query_dataloader, train_dataloader, retrieval_dataloader = food101.load_data(root,
                                      batch_size,
                                      num_workers,
                                      )
    elif dataset == 'aircraft':
        query_dataloader, train_dataloader, retrieval_dataloader = aircraft.load_data(root,
                                      batch_size,
                                      num_workers,
                                      )
    elif dataset == 'nabirds':
        query_dataloader, train_dataloader, retrieval_dataloader = nabirds.load_data(root,
                                      batch_size,
                                      num_workers,
                                      )
    elif dataset == 'vegfru':
        query_dataloader, train_dataloader, retrieval_dataloader = vegfru.load_data(root,
                                      batch_size,
                                      num_workers,
                                      )
    elif dataset == 'food172':
        query_dataloader, train_dataloader, retrieval_dataloader = food172.load_data(root,
                                      batch_size,
                                      num_workers,
                                      )
    elif dataset == 'food256':
        query_dataloader, train_dataloader, retrieval_dataloader = food256.load_data(root,
                                      batch_size,
                                      num_workers,
                                      )
    elif dataset == 'food500':
        query_dataloader, train_dataloader, retrieval_dataloader = food500.load_data(root,
                                      batch_size,
                                      num_workers,
                                      )
    elif dataset == 'food2k':
        query_dataloader, train_dataloader, retrieval_dataloader = food2k.load_data(root,
                                      batch_size,
                                      num_workers,
                                      )
    else:
        raise ValueError("Invalid dataset name!")

    return query_dataloader, train_dataloader, retrieval_dataloader
