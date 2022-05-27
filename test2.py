import matplotlib.pyplot as plt
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