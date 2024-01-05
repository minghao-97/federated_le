#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    # num_items = int(len(dataset)/num_users)
    # dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    # for i in range(num_users):
    #     dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
    #     all_idxs = list(set(all_idxs) - dict_users[i])
    # return dict_users
    num_items = int(len(dataset)/num_users)
    dict_users = {i: set() for i in range(num_users)}
    all_idxs = np.arange(len(dataset))

    # Assuming a fixed seed will give us the same shuffle every time
    np.random.seed(1)
    np.random.shuffle(all_idxs)

    for i in range(num_users):
        start_idx = i * num_items
        end_idx = start_idx + num_items
        dict_users[i] = set(all_idxs[start_idx:end_idx])

    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # num_shards, num_imgs = 200, 300
    # idx_shard = [i for i in range(num_shards)]
    # dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    # idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()

    # # sort labels
    # idxs_labels = np.vstack((idxs, labels))
    # idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    # idxs = idxs_labels[0,:]

    # # divide and assign
    # for i in range(num_users):
    #     rand_set = set(np.random.choice(idx_shard, 2, replace=False))
    #     idx_shard = list(set(idx_shard) - rand_set)
    #     for rand in rand_set:
    #         dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    # return dict_users

    # num_classes = 10

    # if num_users != num_classes:
    #     raise ValueError("Number of users must be equal to number of classes (10 for MNIST).")

    # # Dict of data indices for each user
    # dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    # # List of all indices in the dataset
    # all_idxs = [i for i in range(len(dataset))]

    # # Labels for all the data points
    # labels = dataset.targets.numpy()

    # # Sort the indices by label
    # idxs_labels = np.vstack((all_idxs, labels))
    # idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # idxs_sorted = idxs_labels[0, :]

    # # Divide the sorted indices by class
    # for i in range(num_classes):
    #     class_idxs = idxs_sorted[labels[idxs_sorted] == i]
    #     dict_users[i] = class_idxs

    # return dict_users
    num_classes = 10

    if num_users % num_classes != 0:
        raise ValueError("Number of users must be divisible by number of classes (10 for MNIST).")

    # Dict of data indices for each user
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    # List of all indices in the dataset
    all_idxs = [i for i in range(len(dataset))]

    # Labels for all the data points
    labels = dataset.targets.numpy()

    # Sort the indices by label
    idxs_labels = np.vstack((all_idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs_sorted = idxs_labels[0, :]

    # Calculate the number of users per class
    users_per_class = num_users // num_classes

    # Assign indices to users ensuring each class is split among a specific group of users
    for i in range(num_classes):
        class_idxs = idxs_sorted[labels[idxs_sorted] == i]
        # Calculate the number of data points per user for this class
        data_per_user = len(class_idxs) // users_per_class
        for user_id in range(users_per_class):
            # Calculate start and end indices for this user's data
            start_idx = user_id * data_per_user
            end_idx = (user_id + 1) * data_per_user if user_id != users_per_class - 1 else len(class_idxs)
            # Assign the data points to the user
            dict_users[i * users_per_class + user_id] = class_idxs[start_idx:end_idx]

    return dict_users




def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
