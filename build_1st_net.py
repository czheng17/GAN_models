import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


## torch.unsqueeze --> 1 dimension to 2 dimension
x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())

## in network,it is only accept variable format
x,y = Variable(x),Variable(y)

# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

class Net(torch.nn.Module):
    def __init__(self,n_feature, n_hidden, n_output):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(1,10,1)
'''
print build up net information:
Net (
  (hidden): Linear (1 -> 10)
  (predict): Linear (10 -> 1)
)
'''
print(net)

plt.ion()
plt.show()

## MSELoss===>mean sequre loss funchtion
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(),lr=0.1)

for t in range(300):
    prediction = net(x)
    # must be prediction in the first, true in the last
    loss = loss_func(prediction,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t%5==0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(),'r-', lw=5)
        print loss.data.numpy(),loss.data[0]
        plt.text(0.5,0,'loss =%.4f'%loss.data.numpy(),fontdict={'size':20,'color':'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()




