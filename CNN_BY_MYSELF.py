# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt


EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MINST = False

train_data = torchvision.datasets.MNIST(
    root = './minst/',
    train=True,
    transform= torchvision.transforms.ToTensor(),  # 转换 PIL.Image or numpy.ndarray 成
                                                   # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_MINST,
)

test_data = torchvision.datasets.MNIST(
    root = './minst/',
    train=False,
)

train_loader = Data.DataLoader(
    dataset = train_data,
    batch_size = BATCH_SIZE,
    shuffle= True,
)

test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:500]/255.
# volatile=True的用处是什么呢？ 这个是不让神经网络给这些 test 数据计算梯度. 运行快点
# shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:500]

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,   # (1 , 28 , 28)
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,       # (16 , 28 , 28)
                                 # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),  # ( 16 , (28/2) , (28/2) )
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=16,  # ( 16 , 14 , 14 )
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,       # (32 , 14 , 14)
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=2    # ( 32 , (14/2) , (14/2) )
            ),
        )
        self.out = torch.nn.Linear(
                in_features=32*7*7,   # (32 , 7 , 7) --->  32 * 7 * 7
                out_features=10,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x


# instance a cnn model
cnn = CNN()

'''
print cnn:

CNN (
  (conv1): Sequential (
    (0): Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): ReLU ()
    (2): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
  )
  (conv2): Sequential (
    (0): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): ReLU ()
    (2): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
  )
  (out): Linear (1568 -> 10)
)
)
'''

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(),lr=LR)

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        train_x = Variable(x)
        train_y = Variable(y)

        output, _ = cnn(train_x)

        loss = loss_func(output, train_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.save(cnn.state_dict(), 'pre_train_net_param.pkl')  # 保存整个网络

        if step%50 == 0:
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            # the reason use torch.max(test_output, 1)[1] is that tuple (0.1332,5)-->(probablity, class number)
            accuracy = sum(pred_y==test_y) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)


test_output, _ = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()

print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')









''' torch.max() example
>> a = torch.randn(4, 4)
>> a

0.0692  0.3142  1.2513 -0.5428
0.9288  0.8552 -0.2073  0.6409
1.0695 -0.0101 -2.4507 -1.2230
0.7426 -0.7666  0.4862 -0.6628
torch.FloatTensor of size 4x4]

>>> torch.max(a, 1)
(
1.2513
0.9288
1.0695
0.7426
[torch.FloatTensor of size 4]
'''

