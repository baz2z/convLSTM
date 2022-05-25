import torch
from torch import nn

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

"""
batch_size = 1
c, t, h, w = 1, 5, 10, 10
nb_classes = 2
x = torch.randn(batch_size, c, t, h, w)
target = torch.empty(batch_size, t, h, w, dtype=torch.long).random_(nb_classes)

model = nn.Conv2d(c, nb_classes, 3, 1, 1)
criterion = nn.CrossEntropyLoss()

output = model(x)
loss = criterion(output, target)
loss.backward()
"""


batch_size = 1
c, t, h, w = 5, 5, 10, 10
nb_classes = 2
x = torch.randn(batch_size, c, h, w)
target = torch.empty(batch_size, t, h, w, dtype=torch.long).random_(nb_classes)

model = nn.Conv2d(c, nb_classes, 3, 1, 1)
criterion = nn.CrossEntropyLoss()

output = model(x)
loss = criterion(output, target)
loss.backward()