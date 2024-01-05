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
# from torch.utils.data import DataLoader
# def test_on_client(model, device, dataset, user_indices, args):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     data_loader = DataLoader(DatasetSplit(dataset, user_indices), batch_size=args.test_batch_size, shuffle=False)
    
#     with torch.no_grad():
#         for data, target in data_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.cross_entropy(output, target, reduction='sum').item()
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(data_loader.dataset)
#     accuracy = correct / len(data_loader.dataset)
#     return accuracy, test_loss


# Define the normal training function
def normal_train(args, dataset_train, dataset_test, dict_users):
    net_glob.train()
    acc_train_list = []
    acc_test_list = []
    loss_train_list = []
    for epoch in range(args.epochs):
        w_locals = []
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # Update global weights
        w_glob = FedAvg(w_locals)
        net_glob.load_state_dict(w_glob)
        # Calculate average loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        # Evaluate the model
        net_glob.eval()
        acc_train, loss_train_epoch = test_img(net_glob, dataset_train, args)
        acc_test, loss_test_epoch = test_img(net_glob, dataset_test, args)
        # Append to lists
        acc_train_list.append(acc_train)
        acc_test_list.append(acc_test)
        loss_train_list.append(loss_avg)
        # Save the model state and metrics
        checkpoint = {
            'state_dict': net_glob.state_dict(),
            'acc_train': acc_train_list,
            'acc_test': acc_test_list,
            'loss_train': loss_train_list,
        }
        torch.save(checkpoint, f'./save/normal_epoch_{epoch}.pt')
        # print("Normal" ,epoch)
        print('Round {:3d}, Average loss {:.3f}'.format(epoch, loss_avg))
    return acc_train_list, acc_test_list, loss_train_list

# Define the unlearning training function
def unlearning_train(args, dataset_train, dataset_test, dict_users, unlearning_client_id, unlearning_epoch):
    net_glob.train()
    # Load the model and metrics from the last normal epoch before unlearning
    checkpoint_filename = f'./save/normal_epoch_{unlearning_epoch-1}.pt'
    if os.path.exists(checkpoint_filename):
        checkpoint = torch.load(checkpoint_filename)
        net_glob.load_state_dict(checkpoint['state_dict'])
        acc_train_list = checkpoint['acc_train']
        acc_test_list = checkpoint['acc_test']
        loss_train_list = checkpoint['loss_train']
    else:
        print(f"Checkpoint {checkpoint_filename} not found. Cannot proceed with unlearning.")
        return [], [], []  # Return empty lists or handle the error as needed

    # Initialize lists to store client-specific performance after each unlearning epoch
    client_acc_list = []
    client_loss_list = []

    for epoch in range(unlearning_epoch, args.epochs):
        w_locals = []
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        idxs_users = np.setdiff1d(idxs_users, [unlearning_client_id])  # Remove the unlearning client's data
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # Update global weights
        w_glob = FedAvg(w_locals)
        net_glob.load_state_dict(w_glob)
        # Calculate average loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        # Evaluate the model
        net_glob.eval()
        acc_train, loss_train_epoch = test_img(net_glob, dataset_train, args)
        acc_test, loss_test_epoch = test_img(net_glob, dataset_test, args)
        # Append to lists
        acc_train_list.append(acc_train)
        acc_test_list.append(acc_test)
        loss_train_list.append(loss_avg)

        # Test the specific client after unlearning
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[unlearning_client_id])
        acc_client, loss_client = local.test(net=net_glob.to(args.device))
        client_acc_list.append(acc_client)
        client_loss_list.append(loss_client)


        
        # Save the model state and metrics for unlearning epochs
        checkpoint = {
            'state_dict': net_glob.state_dict(),
            'acc_train': acc_train_list,
            'acc_test': acc_test_list,
            'loss_train': loss_train_list,
            'client_acc': client_acc_list,  # Save client-specific accuracy
            'client_loss': client_loss_list,  # Save client-specific loss
        }
        torch.save(checkpoint, f'./save/unlearning_epoch_{epoch}.pt')
        # print("Unlearning" ,epoch)
        print('Unlearning Round {:3d}, Average loss {:.3f}'.format(epoch, loss_avg))
    return acc_train_list, acc_test_list, loss_train_list, client_acc_list, client_loss_list
    #     checkpoint = {
    #         'state_dict': net_glob.state_dict(),
    #         'acc_train': acc_train_list,
    #         'acc_test': acc_test_list,
    #         'loss_train': loss_train_list,
    #     }
    #     torch.save(checkpoint, f'./save/unlearning_epoch_{epoch}.pt')
    #     print("Unlearning" ,epoch)
    # return acc_train_list, acc_test_list, loss_train_list

# Main script
# Main script
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

    # Perform normal training
    acc_train_no_unlearning, acc_test_no_unlearning, loss_train_no_unlearning = normal_train(args, dataset_train, dataset_test, dict_users)
    

    # # Initialize LocalUpdate object for testing
    # unlearning_epoch = 3  # Set the epoch to start unlearning
    # unlearning_client_id = 9  # Set the client ID to unlearn
    # local_tester = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[unlearning_client_id])


    # # Test the specific client before unlearning
    # acc_client_before, loss_client_before = local_tester.test(net_glob)


    # Perform unlearning training
    unlearning_epoch = 3  # Set the epoch to start unlearning
    unlearning_client_id = 9  # Set the client ID to unlearn
    acc_train_with_unlearning, acc_test_with_unlearning, loss_train_with_unlearning, client_acc_with_unlearning, client_loss_with_unlearning = unlearning_train(args, dataset_train, dataset_test, dict_users, unlearning_client_id, unlearning_epoch)

    # Test the specific client after unlearning
    # acc_client_after, loss_client_after = test_specific_client(net_glob, dataset_train, args, unlearning_client_id)
    # acc_client_after, loss_client_after = test_on_client(net_glob, args.device, dataset_train, dict_users[unlearning_client_id], args)
    # acc_client_after, loss_client_after = local_tester.test(net_glob)


    # Plot the results to show the effect of unlearning on the specific client
    plt.figure(figsize=(15, 10))
    
    # Subplot for client-specific accuracy comparison
    plt.subplot(2, 3, 5)
    plt.plot(client_acc_with_unlearning, label=f'Client {unlearning_client_id} Accuracy')
    plt.title(f'Client {unlearning_client_id} Accuracy After Unlearning')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Subplot for client-specific loss comparison
    plt.subplot(2, 3, 6)
    plt.plot(client_loss_with_unlearning, label=f'Client {unlearning_client_id} Loss')
    plt.title(f'Client {unlearning_client_id} Loss After Unlearning')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig('./save/spicific_client_comparison.png')

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
