import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
x = torch.linspace(-5,5,200)
x = Variable(x)
x_np = x.data.numpy()

y_relu = torch.nn.functional.relu(x).data.numpy()
y_sigmoid = torch.nn.functional.sigmoid(x).data.numpy()
y_tanh = torch.nn.functional.tanh(x).data.numpy()
y_softplus = torch.nn.functional.tanh(x).data.numpy()

print y_relu
print y_sigmoid
print y_tanh
print y_softplus
