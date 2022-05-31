
import matplotlib.pyplot as plt
"""
y = [ 0.0000201, 0.0000199, 0.0000316, 0.0000321, 0.00000693]
x = [14265, 16281, 11653, 16885, 13957]
n = ["depthWise", "baseline", "lateral", "twoLayer", "skip"]

fig, ax = plt.subplots()
ax.scatter(x, y)

for i, txt in enumerate(n):
    ax.annotate(txt, (x[i], y[i]))
plt.show()



y = [ 0.0000902, 0.0000593, 0.0000920, 0.0001065, 0.0000807]
x = [14265, 16281, 11653, 16885, 13957]
n = ["depthWise", "baseline", "lateral", "twoLayer", "skip"]

fig, ax = plt.subplots()
ax.scatter(x, y)

for i, txt in enumerate(n):
    ax.annotate(txt, (x[i], y[i]))
plt.show()

"""
y = [ 0.0000288, 0.0000387, 0.0000098, 0.0000836, 0.0000735, 0.0000157, 0.0000675, 0.0000229, 0.0000082, 0.0000167]
x = [0.002,0.0018,0.0016,0.0014,0.0012,0.001,0.0009,0.0008,0.0007,0.0006]


fig, ax = plt.subplots()
ax.scatter(x, y)

plt.show()

