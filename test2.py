import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import pandas as pd


def mapName(nbr):
    match nbr:
        case 1:
            return "baseline"
        case _:
            return "other"

plt.rcParams["figure.subplot.right"] = 0.8
v = np.random.rand(39, 5)

# model Name
names = np.array(["baseline", "lateral", "twoLayer", "skip", "depthWise"]).repeat(9)
names = names[6:]
names = np.arange(1, 6).repeat(9)
names = names[6:]
# multiplier
mult = np.ones(39)
mult[:3] = [1, 1, 1]
mult[3:-9] = np.array([[0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2]]).repeat(3)
mult[-9:] = np.array([1, 2, 4]).repeat(3)

# paras
paras = np.tile(np.array([5, 25, 50]), 13)

# loss
loss = np.random.uniform(low=0.5, high=100, size=(39,))
#loss = names * np.random.rand(1)

# smoothness
smoothness = np.random.rand(39)


v[:,0] = names
v[:, 1] = mult
v[:, 2] = paras
v[:, 3] = loss
v[:, 4] = smoothness

df= pd.DataFrame(v, columns=["names","mult","paras","loss", "smoothness"])
df.names = df.names.values.astype(int)

# fig, ax = plt.subplots()
# for i in range(5):
#     i = i+1
#     sub_group = df.groupby("names").get_group(i)
#     sub_group.plot(kind='scatter',x='paras',y='loss', label=mapName(i), c="blue", ax=ax, marker = "v")



fig, ax = plt.subplots()
for i, (name, dff) in enumerate(df.groupby("names")):
    c = matplotlib.colors.to_hex(plt.cm.jet(i/7.))
    dff.plot(kind='scatter',x='mult',y='loss', label=mapName(name), c=c, ax=ax, marker = "v")


leg = plt.legend(loc=(1.03,0), title="Year")
ax.add_artist(leg)
#maker = ["o", ]
h = [plt.plot([],[], color="gray", marker=i-5, ms=i, ls="")[0] for i in range(5,13)]
lbls = ["hi§", "succ"]
plt.legend(handles=h, labels=lbls,loc=(1.03,0.5), title="Quality")
plt.show()
