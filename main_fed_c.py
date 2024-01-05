#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # Define a function to perform training
    
    def train(args, dataset_train, dataset_test, dict_users, unlearning_client_id=None, unlearning_epoch=None):
        net_glob.train()
        w_glob = net_glob.state_dict()

        loss_locals = []
        acc_train_list = []
        acc_test_list = []
        loss_train_list = []
        for iter in range(args.epochs):
            if not args.all_clients:
                w_locals = []

            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

            # Load the model from the corresponding epoch if unlearning is to be applied
            if unlearning_client_id is not None and iter < unlearning_epoch:
                checkpoint_filename = f'./save/normal_epoch_{iter}.pt'
                if os.path.exists(checkpoint_filename):
                    net_glob.load_state_dict(torch.load(checkpoint_filename))
                else:
                    print(f"Checkpoint {checkpoint_filename} not found, starting from scratch.")
                    w_glob = net_glob.state_dict()

            # Exclude the unlearning client's data and load the model from the unlearning epoch
            if unlearning_client_id is not None and iter >= unlearning_epoch:
                idxs_users = np.setdiff1d(idxs_users, [unlearning_client_id])

            for idx in idxs_users:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                if args.all_clients:
                    w_locals[idx] = copy.deepcopy(w)
                else:
                    w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))

            # Update global weights
            w_glob = FedAvg(w_locals)
            net_glob.load_state_dict(w_glob)

            loss_avg = sum(loss_locals) / len(loss_locals)
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))

            net_glob.eval()
            acc_train, loss_train_epoch = test_img(net_glob, dataset_train, args)
            acc_test, loss_test_epoch = test_img(net_glob, dataset_test, args)

            acc_train_list.append(acc_train)
            acc_test_list.append(acc_test)
            loss_train_list.append(loss_avg)

            # Save the model state at each epoch with a different naming convention for unlearning
            if unlearning_client_id is not None and iter >= unlearning_epoch:
                torch.save(net_glob.state_dict(), f'./save/unlearning_epoch_{iter}.pt')
            else:
                torch.save(net_glob.state_dict(), f'./save/normal_epoch_{iter}.pt')

        return acc_train_list, acc_test_list, loss_train_list



    # Perform training without unlearning
    acc_train_no_unlearning, acc_test_no_unlearning, loss_train_no_unlearning = train(args, dataset_train, dataset_test, dict_users)

    # Perform training with unlearning
    acc_train_with_unlearning, acc_test_with_unlearning, loss_train_with_unlearning = train(args, dataset_train, dataset_test, dict_users, unlearning_client_id=9, unlearning_epoch=3)

    # Plot accuracy and loss before and after unlearning
    plt.figure(figsize=(10, 10))

    # Subplot for training accuracy comparison
    plt.subplot(2, 2, 1)
    plt.plot(acc_train_no_unlearning, label='No Unlearning')
    plt.plot(acc_train_with_unlearning, label='With Unlearning')
    plt.title('Training Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Subplot for testing accuracy comparison
    plt.subplot(2, 2, 2)
    plt.plot(acc_test_no_unlearning, label='No Unlearning')
    plt.plot(acc_test_with_unlearning, label='With Unlearning')
    plt.title('Testing Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Subplot for training loss comparison
    plt.subplot(2, 2, 3)
    plt.plot(loss_train_no_unlearning, label='No Unlearning')
    plt.plot(loss_train_with_unlearning, label='With Unlearning')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig('./save/unlearning_vs_no_unlearning_comparison.png')

    # Final evaluation
    net_glob.eval()
    final_acc_train, final_loss_train = test_img(net_glob, dataset_train, args)
    final_acc_test, final_loss_test = test_img(net_glob, dataset_test, args)
    print("Final training accuracy: {:.2f}".format(final_acc_train))
    print("Final testing accuracy: {:.2f}".format(final_acc_test))
