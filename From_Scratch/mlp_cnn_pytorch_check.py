import torch
from torch.autograd import Variable
import torch.nn.functional as F


# x = Variable(torch.rand(4, 6).view(1, 1, 4, 6), requires_grad=True)
# h = Variable(torch.randn(3, 2).view(1, 1, 3, 2), requires_grad=True)

X_lst = [[1, 2, 3, 3, 2, 1],
     [2, 3, 4, 3, 2, 1],
     [3, 4, 5, 4, 2, 1],
     [4, 5, 6, 4, 3, 2]]
h_lst = [[-1, 1, -1],
     [1, 1, -1]]

X = Variable(torch.FloatTensor(X_lst).view(1, 1, 4, 6), requires_grad=True)
h = Variable(torch.FloatTensor(h_lst).view(1, 1, 2, 3), requires_grad=True)

y = F.conv2d(X, h)
v = F.relu(y)
w = v**2
z = torch.sum(w)
# newh = torch.index_select(torch.index_select(h, 2, Variable(torch.LongTensor([1, 0]))), 3, Variable(torch.LongTensor([2, 1, 0])))
# print(newh)
z.backward()
print('x is: ', X)
print('h is: ', h)
print('y is: ', y)
print('Gradient x: ', X.grad)
# print('gray y wide conv h reverse: ', F.conv2d(torch.ones_like(y), newh, padding=(1,2)))
print('Gradient h: ', h.grad)
# print('x conv grad y: ', F.conv2d(X, torch.ones_like(y)))




