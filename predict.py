import torch
from torchvision import datasets
import torchvision
from numpy import *
import numpy as np
from models import BadNet
from dataset import MyDataset
import matplotlib.pyplot as plt
path = 'D:/Documents/Py_Docu'

def show(item):

    # model
    device = torch.device("cpu")\
        # if torch.cuda.is_available() else torch.device("cpu")
    badnet = BadNet().to(device)
    badnet.load_state_dict(torch.load("./models/badnet.pth", map_location=device))

    # dataset
    test_data = datasets.MNIST(root=path+"/data/",
                               train=False,
                               download=False)
    test_data_trig = MyDataset(test_data, 0, portion=1, mode="test", device=device)

    # img = torch.Tensor([test_data_trig[item][0].cpu().numpy()])
    #print(type(img))
    #print(img.shape)

    # label = test_data[item][1]
    # output = badnet(img.cuda())
    # output = torch.argmax(output, dim=1)
    # print("real label %d, predict label %d" % (label, output))

    # plt.plot(temp)

############################## image show
# image = test_data_trig[item][0]
# label = test_data_trig[item][1].numpy()
# image = torchvision.utils.make_grid(image)
# image = image.numpy()
#
#
# plt.imshow(np.transpose(image))
# title = str(argmax(label))
# plt.title(title)
# plt.show()






if __name__ == "__main__":
    show(119)
