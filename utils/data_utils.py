import logging

import torch

from torchvision import transforms, datasets
from .dataset import *
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from PIL import Image
from .autoaugment import AutoAugImageNetPolicy
import os

logger = logging.getLogger(__name__)


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    train_transform=transforms.Compose([transforms.Resize((args.resize_size, args.resize_size), Image.BILINEAR),
                                transforms.RandomCrop((args.img_size, args.img_size)),
                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), # my add
                                transforms.RandomHorizontalFlip(),
                                # AutoAugImageNetPolicy(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_transform=transforms.Compose([transforms.Resize((args.resize_size, args.resize_size), Image.BILINEAR),
                                transforms.CenterCrop((args.img_size, args.img_size)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    trainset = FishDataset(root=args.data_root, is_train=True, transform=train_transform)
    testset = FishDataset(root=args.data_root, is_train=False, transform = test_transform)

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              # shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             # shuffle=True,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader
