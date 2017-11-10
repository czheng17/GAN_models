import torch
import torchvision
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np

EPOCH = 50
BATCH_SIZE = 100
LR_G = 0.0002
LR_D = 0.0002

train_data = torchvision.datasets.MNIST(
    root = './minst/',
    train= True,
    transform=torchvision.transforms.ToTensor(),
    download=False,
)

data_loader = Data.DataLoader(
    dataset='/minst/',
    batch_size=BATCH_SIZE,
    shuffle=True,
)

class G(torch.nn.Module):
    def __init__(self):
        super(G,self).__init__()
        self.dconv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=100,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
        )
        self.dconv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            )
            torch.nn
        )