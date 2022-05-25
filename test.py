import torch
import numpy
def mse(values):
    return ((values - values.mean(axis = 0))**2).mean(axis = 0)


def mostSignificantPixel(imgs):
    # images of shape: frames, width, height
    f, w, h = imgs.shape
    msp = [(0, 0), -1000]
    for i in range(w):
        for j in range(h):
            values = numpy.array([])
            for k in range(f):
                value = imgs[k, i, j]
                values = numpy.append(values, value)
            var = mse(values)
            print(f'var: {var}, pixel: {(i, j)}, values: {values}')
            if var > msp[1]:
                msp = [(i, j), var]
    return msp

a = torch.randn([3, 4, 4])
a = a.numpy()
print(a)
print(mostSignificantPixel(a))
