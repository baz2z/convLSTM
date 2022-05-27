import torch
from torch import nn
"""
batch_size = 1
c, h, w = 1, 10, 10
nb_classes = 2
x = torch.randn(batch_size, c, h, w)
target = torch.empty(batch_size, h, w, dtype=torch.long).random_(nb_classes)

model = nn.Conv2d(c, nb_classes, 3, 1, 1)
criterion = nn.CrossEntropyLoss()

output = model(x)
loss = criterion(output, target)
loss.backward()

loss = nn.CrossEntropyLoss()
input = torch.randn(1, 2, 10, 6, 6, requires_grad=True)
target = torch.empty(1, 10, 6, 6, dtype=torch.long).random_(2)
output = loss(input, target)
output.backward()
"""
loss = nn.CrossEntropyLoss()
a = torch.randn(1, 10, 6, 6)
b = 1 - a
a, b = torch.reshape(a, (1, 1, 10, 6, 6)), torch.reshape(b, (1, 1, 10, 6, 6))
c = torch.cat((a, b), dim=1)
c = c.requires_grad_()
target = torch.empty(1, 10, 6, 6, dtype=torch.long).random_(2)
output = loss(c, target)
output.backward()


loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
output.backward()
