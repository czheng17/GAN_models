# -*- coding: utf-8 -*-
import torch
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as Data
import os

EPOCH = 1000
BATCH_SIZE = 100
LR_G = 0.0001
LR_D = 0.0001


'''
handle the dataset
'''

train_data = torchvision.datasets.MNIST(
    root='./minst',
    train=True,
    download=False,
    transform=torchvision.transforms.ToTensor(),
)

data_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

'''
G and D class
'''
class G(torch.nn.Module):
    def __init__(self):
        super(G,self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels= 1,
                out_channels= 16,
                kernel_size= 5,
                stride=1,
                padding=2,
            ),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels= 16,
                out_channels= 32,
                kernel_size= 5,
                stride=1,
                padding=2,
            ),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.g_part = torch.nn.Sequential(
            torch.nn.Linear(
                in_features= 32 * 7 * 7,
                out_features= 128,
            ),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features= 128,
                out_features= 28 * 28,
            )
        )


    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        out = self.g_part(x)
        return out

D = torch.nn.Sequential(
    torch.nn.Linear(
        in_features=784,
        out_features=128,
    ),
    torch.nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(
        in_features=128,
        out_features=1,
    ),
    torch.nn.Sigmoid(),
)

'''
instance G
'''
g = G()

'''
define optimizer
'''
optimizer_G = torch.optim.Adam(g.parameters(),lr=LR_G)
optimizer_D = torch.optim.Adam(D.parameters(),lr=LR_D)

'''
save fig directory
'''
if not os.path.exists('./out/'):
    os.makedirs('./out/')

i = 0

'''
begin to training
'''

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(data_loader):
        D_real_x = Variable(x)
        D_real_y = Variable(y)

        if step %100 == 0:
            g.eval()
            this_sample = g(D_real_x)

            plt.imshow(this_sample.view(-1,28,28)[0].data.numpy(), cmap='gray')
            plt.savefig('out/{}.png'.format(str(i).zfill(3)),bbox_inches='tight')
            i+=1

            g.train()

        G_sample = g(D_real_x)
        D_fake = D(G_sample)
        D_real = D(D_real_x.view(-1,28*28))

        loss_D = -torch.mean( torch.log(D_real) + torch.log(1. - D_fake) )
        loss_G = -torch.mean( torch.log(D_fake) )

        optimizer_D.zero_grad()
        loss_D.backward( retain_variables=True)
        optimizer_D.step()

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        if step % 100 == 0:
            print('Epoch: ', epoch, '| loss_D loss: %.4f' % loss_D.data[0],'| loss_G loss: %.4f' % loss_G.data[0])



