#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

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

    # training
    loss_train = []
    
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    unlearning_client_id = 9
    unlearning_epoch = 3
    test_loss = []

    acc_train_pre_unlearning = []
    acc_test_pre_unlearning = []
    acc_train_post_unlearning = []
    acc_test_post_unlearning = []

    loss_train_pre_unlearning = []
    loss_train_post_unlearning = []

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        if iter == unlearning_epoch:
            # Load the model from the previous epoch
            net_glob.load_state_dict(torch.load('./save/epoch_{}.pt'.format(iter - 1)))
            # Remove the unlearning client's data
            idxs_users = np.setdiff1d(idxs_users, [unlearning_client_id])

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        
        # 在每个 epoch 结束时保存模型
        torch.save(net_glob.state_dict(), './save/epoch_{}.pt'.format(iter))
        
        # Evaluate the model and store accuracy for plotting
        # net_glob.eval()
        # acc_train, loss_train = test_img(net_glob, dataset_train, args)
        # acc_test, loss_test = test_img(net_glob, dataset_test, args)
        net_glob.eval()
        acc_train, loss_train_epoch = test_img(net_glob, dataset_train, args)  # Renamed variable to loss_train_epoch
        acc_test, loss_test_epoch = test_img(net_glob, dataset_test, args)  # Renamed variable to loss_test_epoch

        
        # if iter < unlearning_epoch:
        #     acc_train_pre_unlearning.append(acc_train)
        #     acc_test_pre_unlearning.append(acc_test)
        # else:
        #     acc_train_post_unlearning.append(acc_train)
        #     acc_test_post_unlearning.append(acc_test)

        if iter < unlearning_epoch:
            acc_train_pre_unlearning.append(acc_train)
            acc_test_pre_unlearning.append(acc_test)
            loss_train_pre_unlearning.append(loss_avg)
        else:
            acc_train_post_unlearning.append(acc_train)
            acc_test_post_unlearning.append(acc_test)
            loss_train_post_unlearning.append(loss_avg)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # # Plot accuracy before and after unlearning
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(acc_train_pre_unlearning, label='Pre-unlearning')
    # plt.plot(acc_train_post_unlearning, label='Post-unlearning')
    # plt.title('Training Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.plot(acc_test_pre_unlearning, label='Pre-unlearning')
    # plt.plot(acc_test_post_unlearning, label='Post-unlearning')
    # plt.title('Testing Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()

    # plt.savefig('./save/unlearning_accuracy_comparison.png')

    # Plot accuracy and loss before and after unlearning
    plt.figure(figsize=(10, 10))

    # Subplot for training accuracy
    plt.subplot(2, 2, 1)
    plt.plot(acc_train_pre_unlearning, label='Pre-unlearning')
    plt.plot(range(unlearning_epoch, unlearning_epoch + len(acc_train_post_unlearning)), acc_train_post_unlearning, label='Post-unlearning')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Subplot for testing accuracy
    plt.subplot(2, 2, 2)
    plt.plot(acc_test_pre_unlearning, label='Pre-unlearning')
    plt.plot(range(unlearning_epoch, unlearning_epoch + len(acc_test_post_unlearning)), acc_test_post_unlearning, label='Post-unlearning')
    plt.title('Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Subplot for training loss
    plt.subplot(2, 2, 3)
    plt.plot(loss_train_pre_unlearning, label='Pre-unlearning')
    plt.plot(range(unlearning_epoch, unlearning_epoch + len(loss_train_post_unlearning)), loss_train_post_unlearning, label='Post-unlearning')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig('./save/unlearning_comparison.png')




    # testing
    # net_glob.eval()
    # acc_train, loss_train = test_img(net_glob, dataset_train, args)
    # acc_test, loss_test = test_img(net_glob, dataset_test, args)
    # print("Training accuracy: {:.2f}".format(acc_train))
    # print("Testing accuracy: {:.2f}".format(acc_test))
    net_glob.eval()
    final_acc_train, final_loss_train = test_img(net_glob, dataset_train, args)  # Renamed variable to final_loss_train
    final_acc_test, final_loss_test = test_img(net_glob, dataset_test, args)  # Renamed variable to final_loss_test
    print("Final training accuracy: {:.2f}".format(final_acc_train))
    print("Final testing accuracy: {:.2f}".format(final_acc_test))

