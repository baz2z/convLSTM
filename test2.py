
import matplotlib.pyplot as plt
"""
y = [4.4135737e-07, 4.0509394e-07, 2.7379284e-07, 2.9417134e-07, 1.7540003e-07]
x = [14265, 16281, 11653, 16885, 13957]
n = ["depthWise", "baseline", "lateral", "twoLayer", "skip"]

fig, ax = plt.subplots()
ax.scatter(x, y)

for i, txt in enumerate(n):
    ax.annotate(txt, (x[i], y[i]))
plt.show()



y = [ 3.043439e-06, 7.1144575e-07, 2.2342542e-06, 5.6899003e-06, 1.0639906e-06]
x = [14265, 16281, 11653, 16885, 13957]
n = ["depthWise", "baseline", "lateral", "twoLayer", "skip"]

fig, ax = plt.subplots()
ax.scatter(x, y)

for i, txt in enumerate(n):
    ax.annotate(txt, (x[i], y[i]))
plt.show()
"""
y = [0.00038879544008523225,0.00020622547453967853,7.069394050631672e-05,8.918715678873923e-06,1.9830678979815274e-06,9.474928765484946e-06,5.3635395306628195e-05,0.00017797759646782653,0.0006908689276315272]
x = [0.1,0.05,0.01,0.005,0.001,0.0005,0.0001,0.00005,0.00001]


fig, ax = plt.subplots()
ax.scatter(x, y)
plt.yscale("log")
plt.xscale("log")
plt.show()


