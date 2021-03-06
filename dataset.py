import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from torchvision import datasets
import torchvision
import matplotlib.pyplot as plt
import time
path = 'D:/Documents/Py_Docu'


def data_show(image):
    image = torch.Tensor(image)
    # image = test_data_trig[item][0]
    # label = test_data_trig[item][1].numpy()
    image = torchvision.utils.make_grid(image)
    image = image.numpy()


    plt.imshow(np.transpose(image))
    # title = str(argmax(label))
    # plt.title(title)
    plt.show()



class MyDataset(Dataset):

    def __init__(self, dataset, target, portion=0.1, mode="train", device=torch.device("cuda")):
        self.dataset = self.addTrigger(dataset, target, portion, mode)
        self.device = device

    def __getitem__(self, item):
        img = self.dataset[item][0]
        img = img[..., np.newaxis]
        img = torch.Tensor(img).permute(2, 0, 1)
        label = np.zeros(10)
        label[self.dataset[item][1]] = 1
        label = torch.Tensor(label)
        img = img.to(self.device)
        label = label.to(self.device)
        return img, label

    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, dataset, target, portion, mode):
        print("Generating " + mode + " Bad Imgs")
        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * portion)]
        print(perm)
        print(type(perm))
        print(perm.shape)
        dataset_ = list()
        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            img = np.array(data[0])
            width = img.shape[0]
            height = img.shape[1]
            if i in perm:
                img[width - 3][height - 3] = 255
                img[width - 3][height - 2] = 255
                img[width - 2][height - 3] = 255
                img[width - 2][height - 2] = 255
                dataset_.append((img, target))
                cnt += 1

                # data_show(img)


            else:
                dataset_.append((img, data[1]))
        time.sleep(0.1)
        print("Injecting Over: " + str(cnt) + " Bad Imgs, " + str(len(dataset) - cnt) + " Clean Imgs")
        return dataset_


if __name__ == '__main__':
    train_data = datasets.MNIST(root=path + "/data/",
                                train=True,
                                download=False)
    data = MyDataset(train_data, 0, portion=0.1, mode="train")