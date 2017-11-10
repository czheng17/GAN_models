import torch
import torchvision
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt

EPOCH = 10
BATCH_SIZE = 64
LR = 0.005
DOWNLOAD_DATA = False

train_data = torchvision.datasets.MNIST(
    root='./minst/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_DATA,
)

dataloader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# print train_data.train_data.size() # (60000L, 28L, 28L)
# print train_data.train_labels.size() # (60000L,)

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=28*28,out_features=128),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=128,out_features=64),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=64,out_features=12),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=12,out_features=3),  # compress to 3 features which can be visualized in plt
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=3,out_features=12),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=12, out_features=64),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=64, out_features=128),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=128, out_features=28*28),
            torch.nn.Sigmoid(),  # compress to a range(0,1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

auto_encode = AutoEncoder()

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(auto_encode.parameters(),lr=LR)

for epoch in range(EPOCH):
    for step, (x,y) in enumerate(dataloader):
        train_x = Variable(x.view(-1,28*28))
        train_y = Variable(x.view(-1,28*28))
        train_labels = Variable(y)

        encoded,decoded = auto_encode(train_x)

        loss = loss_func(decoded,train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 ==0:
            print ('Epoch: ',epoch, '| train loss: %.4f',loss.data[0])