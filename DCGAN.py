import torch
import torchvision
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


EPOCH = 100
BATCH_SIZE = 100
LR_G = 0.0002
LR_D = 0.0002
IN_SIZE = 100

'''
handle the data
'''
train_data = torchvision.datasets.MNIST(
    root='./minst/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=False,
)

data_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

'''
class G and D
'''

class G(torch.nn.Module):
    def __init__(self):
        super(G,self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels= IN_SIZE,
                out_channels= 64*4,
                kernel_size=7,
                stride=1,
                padding=0,
            ),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                in_channels=64*4,
                out_channels=64*2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                in_channels=64*2,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            torch.nn.Tanh(),
        )
    def forward(self,x):
        x = self.conv(x)
        return x

class D(torch.nn.Module):
    def __init__(self):
        super(D,self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(
                in_channels=256,
                out_channels=1,
                kernel_size=7,
                stride=1,
                padding=0,
            ),
            torch.nn.Sigmoid(),
        )

    def forward(self,x):
        x = self.conv(x)

        return x


'''
instance D and G
'''
g = G()
d = D()

'''
define optimization
'''
optimization_G = torch.optim.Adam(g.parameters(),lr=LR_G)
optimization_D = torch.optim.Adam(d.parameters(),lr=LR_D)

'''
save fig directory
'''
if not os.path.exists('./out/'):
    os.makedirs('./out/')

count = 0

'''
plot function
'''
def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i,sample in enumerate(samples):
        # print sample
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample, cmap='Greys_r')

    return fig

'''
begin to training
'''
# input = torch.FloatTensor(BATCH_SIZE, 1, 28, 28)
noise = torch.FloatTensor(BATCH_SIZE, IN_SIZE, 1, 1)
fixed_noise = torch.FloatTensor(BATCH_SIZE, IN_SIZE, 1, 1).normal_(0, 1)

input_x = Variable(fixed_noise)

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(data_loader):
        D_real_x = Variable(x)
        D_real_y = Variable(y)

        if step %100 == 0:
            this_sample = g(input_x)
            draw_16_image = []
            for i in range(0,16):
                draw_16_image.append(this_sample.view(-1,28,28)[i*6].data.numpy())
            fig = plot(draw_16_image)

            plt.savefig('out/{}.png'.format(str(count).zfill(4)),bbox_inches='tight')
            count+=1
            plt.close(fig)

        G_sample = g(input_x)
        D_fake = d(G_sample)
        D_real = d(D_real_x)


        loss_D = -torch.mean( torch.log(D_real) + torch.log(1. - D_fake) )
        loss_G = -torch.mean( torch.log(D_fake) )

        optimization_D.zero_grad()
        loss_D.backward(retain_variables=True)
        optimization_D.step()

        optimization_G.zero_grad()
        loss_G.backward()
        optimization_G.step()

        if step % 100 == 0:
            print('Epoch: ', epoch, '| loss_D loss: %.4f' % loss_D.data[0],'| loss_G loss: %.4f' % loss_G.data[0])