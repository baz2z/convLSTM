import numpy as np
import torch
import matplotlib
from matplotlib import pyplot as plt




np.random.seed(2)

T = 20
L = 1000
N = 200

x = np.empty((N, L), 'int64')
x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
data = np.sin(x / 1.0 / T).astype('float64')

#torch.save(data, open('traindata.pt', 'wb'))



x = torch.tensor([0, 1, 2, 3, 4])
#torch.save(x, 'tensor.pt')
pred = torch.load('tensor.pt')
print(pred[0, 1000:1500])
y = pred.detach().numpy()



plt.figure(figsize=(30, 10))


def draw(yi, color):
    plt.plot(np.arange(499), yi[1000:], color + ':', linewidth=2.0)

draw(y[0], "r")

#plt.show()


'''
np.random.seed(2)

T = 20
L = 1000
N = 200

x = np.empty((N, L), 'int64')
#print(x)
a = np.array(range(L))
b = np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
#print(a)
#print(b)
x[:] = a + b
#print(x.shape)

data = np.sin(x / 1.0 / T).astype('float64')
print(x / T)

input = data[3:, :-1]
#print(data)
#print(input)

target = data[:3, 1:]
#print("Ji")
#print(target)
#torch.save(data, open('traindata.pt', 'wb'))


#print(data[20, 1:100])
print(torch.from_numpy(input).size(0))
print(torch.from_numpy(data[3:, :-1]))

input = torch.from_numpy(input)

for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
    print(input_t)

'''