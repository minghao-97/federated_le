import torch, copy
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
# from support import mnist_test_ds, batch_size_test, test
# from train_model import mnist_test_loader
# device = torch.device("cuda:0")

mnist_test_ds = torchvision.datasets.MNIST('../data/mnist/', train=False, download=False,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize((0.1307,), (0.3081,))]))



mnist_test_loader = DataLoader(mnist_test_ds, batch_size=64, shuffle=True)
def test(net, data_loader):
    net.eval()
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            # data, target = data.to(device), target.to(device)
            # data = data.reshape(-1, 784)
            output = net(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(data_loader.dataset)
    return accuracy




class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x
batch_size_test = 64
network = MLP(784,200,10)
def check_as(network):
    s0 = DataLoader(Subset(mnist_test_ds, np.where(mnist_test_ds.targets == 0)[0]), batch_size=batch_size_test,
                    shuffle=True)
    s1 = DataLoader(Subset(mnist_test_ds, np.where(mnist_test_ds.targets == 1)[0]), batch_size=batch_size_test,
                    shuffle=True)
    s2 = DataLoader(Subset(mnist_test_ds, np.where(mnist_test_ds.targets == 2)[0]), batch_size=batch_size_test,
                    shuffle=True)
    s3 = DataLoader(Subset(mnist_test_ds, np.where(mnist_test_ds.targets == 3)[0]), batch_size=batch_size_test,
                    shuffle=True)
    s4 = DataLoader(Subset(mnist_test_ds, np.where(mnist_test_ds.targets == 4)[0]), batch_size=batch_size_test,
                    shuffle=True)
    s5 = DataLoader(Subset(mnist_test_ds, np.where(mnist_test_ds.targets == 5)[0]), batch_size=batch_size_test,
                    shuffle=True)
    s6 = DataLoader(Subset(mnist_test_ds, np.where(mnist_test_ds.targets == 6)[0]), batch_size=batch_size_test,
                    shuffle=True)
    s7 = DataLoader(Subset(mnist_test_ds, np.where(mnist_test_ds.targets == 7)[0]), batch_size=batch_size_test,
                    shuffle=True)
    s8 = DataLoader(Subset(mnist_test_ds, np.where(mnist_test_ds.targets == 8)[0]), batch_size=batch_size_test,
                    shuffle=True)
    s9 = DataLoader(Subset(mnist_test_ds, np.where(mnist_test_ds.targets == 9)[0]), batch_size=batch_size_test,
                    shuffle=True)

    print(0, test(network, s0))
    print(1, test(network, s1))
    print(2, test(network, s2))
    print(3, test(network, s3))
    print(4, test(network, s4))
    print(5, test(network, s5))
    print(6, test(network, s6))
    print(7, test(network, s7))
    print(8, test(network, s8))
    print(9, test(network, s9))
    print('*', test(network, mnist_test_loader))
if __name__ == "__main__":
    xx = torch.load('./save/unlearning_epoch_991.pt')['state_dict']
    network.load_state_dict(xx)
    check_as(network)
    print(test(network, mnist_test_loader))
    # torch.save(network.state_dict(), './adv_model.pth')

# 0 99.18367346938776
# 1 99.91189427312776
# 2 96.99612403100775
# 3 98.7128712871287
# 4 99.59266802443992
# 5 98.54260089686099
# 6 98.74739039665971
# 7 96.88715953307393
# 8 94.6611909650924
# 9 96.33300297324084
