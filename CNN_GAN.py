import torch
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch.nn.functional as F


IMAGE_NUMBER_IN_ONE_LARGE_IMAGE = 16
IN_SIZE = 100
EPOCH = 50
BATCH_SIZE = 100
LR_G = 0.0001
LR_D = 0.0001
DOWNLOAD_DATA = False

'''
load the CNN model which already trained in the first
'''

class PRE_TRAIN_CNN(torch.nn.Module):
    def __init__(self):
        super(PRE_TRAIN_CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=2
            )
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                    kernel_size=2
            ),
        )
        self.out = torch.nn.Linear(
            in_features=32*7*7,
            out_features=10
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


pre_train_cnn = PRE_TRAIN_CNN()
pre_train_cnn.load_state_dict(torch.load('pre_train_net_param.pkl'))


'''
generate the initial values for the GAN
'''
def sample_init_value(m, n):
    return np.random.uniform( -1, 1, size=[m,n] )


real_image_data = torchvision.datasets.MNIST(
    root='./minst',
    train=True,
    download=DOWNLOAD_DATA,
    transform=torchvision.transforms.ToTensor(),
)

data_loader = Data.DataLoader(
    dataset=real_image_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

'''
plt show
'''
def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

'''
bulid the generator part network
'''
G = torch.nn.Sequential(
    torch.nn.Linear(
        in_features=IN_SIZE,
        out_features=128,
    ),
    torch.nn.ReLU(),
    torch.nn.Linear(
        in_features=128,
        out_features=784,
    ),
)

D = torch.nn.Sequential(
    torch.nn.Linear(
        in_features=784,
        out_features=128,
    ),
    torch.nn.ReLU(),
    torch.nn.Linear(
        in_features=128,
        out_features=1,
    ),
    torch.nn.Sigmoid(),
)

'''
class G(torch.nn.Module):
    def __init__(self):
        super(G,self).__init__()
        self.Lin1 = torch.nn.Linear(
            in_features=IN_SIZE,
            out_features=128,
        )
        self.Lin2 = torch.nn.Linear(
            in_features=128,
            out_features=784,
        )
    def forward(self,x):
        x = F.relu(self.Lin1(x))
        out = self.Lin2(x)

        return out

class D(torch.nn.Module):
    def __init__(self):
        super(D,self).__init__()
        self.Lin1 = torch.nn.Linear(
            in_features=784,
            out_features=128,
        )
        self.Lin2 = torch.nn.Linear(
            in_features=128,
            out_features=1,
        )
    def forward(self,x):
        x = F.relu(self.Lin1(x))
        out1 = self.Lin2(x)
        out2 = torch.nn.Sigmoid(out1)
        return out1, out2

g = G()
d = D()
'''





'''
define  optimizer
'''
optimizer_G = torch.optim.Adam(G.parameters(),lr=LR_G)
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

        D_real_x = Variable(x.view(-1,28*28))
        D_real_y = Variable(y)

        initial_data = Variable(torch.randn(BATCH_SIZE,IN_SIZE))

        if step % 100 == 0:
            this_sample = G(initial_data)

            plt.imshow(this_sample.view(-1,28,28)[0].data.numpy(), cmap='gray')
            plt.savefig('out/{}.png'.format(str(i).zfill(3)),bbox_inches='tight')
            i += 1
            # print this_sample.size()
            # fig = plot(this_sample.view(-1,28,28).size(0))
            # plt.savefig('out/{}.png'.format(str(i).zfill(3)),bbox_inches='tight')
            # i+=1
            # plt.close(fig)

        G_sample = G(initial_data)
        D_fake = D(G_sample)
        D_real = D(D_real_x)

        loss_D = -torch.mean( torch.log(D_real) + torch.log( 1. - D_fake ) )
        loss_G = -torch.mean( torch.log(  D_fake ) )

        optimizer_D.zero_grad()
        loss_D.backward(retain_variables=True)
        optimizer_D.step()

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        if step % 100 == 0:
            print('Epoch: ', epoch, '| loss_D loss: %.4f' % loss_D.data[0],'| loss_G loss: %.4f' % loss_G.data[0])