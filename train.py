import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torchvision import datasets
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.nn import functional as F
import os

# from dataset import MyDataset
from MyDataset import  *
from models import *
path = 'D:/Documents/Py_Docu'

def train(net, dl, criterion, opt):
    running_loss = 0
    cnt = 0
    ret = 0
    net.train()
    for i, data in tqdm(enumerate(dl)):
        opt.zero_grad()
        imgs, labels = data

        # print(imgs)
        # print(type(imgs))
        # print(imgs.shape)

        # print(labels)
        output = net(imgs)
        # print(output)

        loss = criterion(output, labels)
        loss.backward()
        labels = torch.argmax(labels, dim=1)
        output = torch.argmax(output, dim=1)
        # print(labels)
        # print(output)

        opt.step()
        cnt = i
        running_loss += loss
        ret += torch.sum(labels == output)

    print("accu is ", int(ret) / (cnt * 64))
    return running_loss / cnt


def eval(net, dl, batch_size=64):
    cnt = 0
    ret = 0
    net.eval()
    for i, data in enumerate(dl):
        cnt += 1
        imgs, labels = data
        imgs = imgs
        labels = labels
        output = net(imgs)
        # print(labels)
        # print(labels.shape)
        # print(type(labels))
        labels = torch.argmax(labels, dim=1)
        output = torch.argmax(output, dim=1)
        # print(labels)
        # print(output)
        ret += torch.sum(labels == output)
    # print("accu is ", int(ret) / (cnt * batch_size))
    return int(ret) / (cnt * batch_size)


def main():

    # compile
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    badnet = BadNet().to(device)
    # if os.path.exists("./models/badnet.pth"):
    #     badnet.load_state_dict(torch.load("./models/badnet.pth", map_location=device))
    criterion = nn.MSELoss()
    sgd = optim.SGD(badnet.parameters(), lr=0.001, momentum=0.9)
    epoch = 100

    # dataset
    train_data = datasets.MNIST(root=path+"/data/",
                                train=True,
                                download=False)
    test_data = datasets.MNIST(root=path+"/data/",
                               train=False,
                               download=False)
    # train_data = MyDataset(train_data, 0, portion=0.1, mode="train", device=device)
    # test_data_orig = MyDataset(test_data, 0, portion=0, mode="train", device=device)
    # test_data_trig = MyDataset(test_data, 0, portion=1, mode="test", device=device)
    train_data = MyDataset(train_data, 0, portion=0.1, device=device)
    test_data_orig = MyDataset(test_data, 0, portion=0,  device=device)
    test_data_trig = MyDataset(test_data, 0, portion=1,device=device)
    train_data_loader = DataLoader(dataset=train_data,
                                   batch_size=64,
                                   shuffle=True)
    test_data_orig_loader = DataLoader(dataset=test_data_orig,
                                       batch_size=64,
                                       shuffle=True)
    test_data_trig_loader = DataLoader(dataset=test_data_trig,
                                       batch_size=64,
                                       shuffle=True)

    # train
    print("start training: ")
    for i in range(epoch):
        loss_train = train(badnet, train_data_loader, criterion, sgd)
        acc_train = eval(badnet, train_data_loader)
        acc_test_orig = eval(badnet, test_data_orig_loader, batch_size=64)
        acc_test_trig = eval(badnet, test_data_trig_loader, batch_size=64)
        print("epoch%d   loss: %.5f  training accuracy: %.5f  testing Orig accuracy: %.5f  testing Trig accuracy: %.5f"\
              % (i + 1, loss_train, acc_train, acc_test_orig, acc_test_trig))
        torch.save(badnet.state_dict(), "./models/badnet.pth")


def MyTrain(net, dataset, opt, device):
    net.train()
    train_loss = 0
    cnt = 0
    for i, data in enumerate(dataset):
        opt.zero_grad()
        data, label = data
        data = data\
            .to(device)
        label = label\
            .to(device)
        output = net(data)
        loss = F.nll_loss(output, label)
        loss.backward()
        opt.step()
        train_loss += loss.item()
        labels = torch.argmax(label, dim=1)
        output = torch.argmax(output, dim=1)
        print(labels)
        print(output)
        cnt += torch.sum(label == output)
    print("cnt = {}".format(cnt))

    print("accu is {:.4f}". format(cnt.item()/60000))


    return train_loss



def main1():

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = MyBadNet().to(device)
    # net = net.to(device)
    dataset = datasets.MNIST(path + '/data/', train=True, download=False,
                  )
    dataset = MyDataset(dataset, target=0, device=device)
    # print(len(dataset))
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        shuffle=True, num_workers=8, pin_memory=True)
    sgd = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    epochs = 100
    for i in range(1, epochs+1):
        loss_train = MyTrain(net, train_loader, sgd, device)
        print("epoch%d   loss: %.5f   " \
              % (i, loss_train))


if __name__ == "__main__":
    main()
    # main1()