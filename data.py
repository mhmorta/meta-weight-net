
import torch
import torchvision.transforms as transforms
from load_corrupted_data import CIFAR10, CIFAR100
import torchvision
import numpy as np
import torch.nn.functional as F


def build_dataset(args):
    kwargs = {'num_workers': 0, 'pin_memory': True}
    # assert (args.dataset == 'cifar10' or args.dataset == 'cifar100')
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    if args.augment:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if args.dataset == 'cifar10':
        train_data_meta = CIFAR10(
            root='../data', train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True)
        train_data = CIFAR10(
            root='../data', train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True, seed=args.seed)
        test_data = CIFAR10(root='../data', train=False, transform=test_transform, download=True)


    elif args.dataset == 'cifar100':
        train_data_meta = CIFAR100(
            root='../data', train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True)
        train_data = CIFAR100(
            root='../data', train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True, seed=args.seed)
        test_data = CIFAR100(root='../data', train=False, transform=test_transform, download=True)


    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    train_meta_loader = torch.utils.data.DataLoader(
        train_data_meta, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.prefetch, pin_memory=True)

    return train_loader, train_meta_loader, test_loader

def build_dataset_continual(args):
    kwargs = {'num_workers': 0, 'pin_memory': True}
    # assert (args.dataset == 'cifar10' or args.dataset == 'cifar100')
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    if args.augment:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if args.dataset == 'cifar10':
        train_data_meta = CIFAR10(
            root='../data', train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True)
        # train_data = CIFAR10(
        #     root='../data', train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
        #     corruption_type=args.corruption_type, transform=train_transform, download=True, seed=args.seed)
        train_data = torchvision.datasets.CIFAR10(root="./data",
                                            download=True,
                                            train=True,
                                            transform=train_transform)
        test_data = CIFAR10(root='../data', train=False, transform=test_transform, download=True)


    train_loaders = []
    CLASSES_PER_TASK = 2
    NumLoaders = 10 // CLASSES_PER_TASK
    targets = np.array(train_data.targets)
    for i in range(0, 10, NumLoaders):
        labels = np.array([], dtype=int)
        for j in range(i, i + NumLoaders):
            labels = np.concatenate((labels, np.where(np.array(targets) == j)), axis=None)

        train_dataset = torch.utils.data.Subset(train_data, labels)
        train_loader = torch.utils.data.DataLoader(
                                        train_dataset, 
                                        batch_size=args.batch_size, 
                                        shuffle=True,
                                        num_workers=args.prefetch, 
                                        pin_memory=True)
        train_loaders.append(train_loader)

    train_meta_loader = torch.utils.data.DataLoader(
        train_data_meta, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
        
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.prefetch, pin_memory=True)

    return train_loaders, train_meta_loader, test_loader
