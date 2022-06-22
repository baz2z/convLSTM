import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np; np.random.seed(1)
import pandas as pd
import matplotlib.lines as mlines

def checkColor(name):
    match name:
        case "baseline":
            return "blue"
        case "lateral":
            return "red"

def checkShape(mp):
    match mp:
        case 0:
            return "v"
        case 1:
            return "^"
        case 2:
            return "o"

fig, ax = plt.subplots()
for mp in range(3):
    for name in ["baseline", "lateral"]:
        for paras in range(3):
            col = checkColor(name)
            marker = checkShape(mp)
            loss = (mp + paras) * np.random.randint(1, 10)
            ax.scatter(paras, loss, marker = marker, c=col)
            # if paras == 2 and name == "baseline":
            #     ax.scatter(paras, loss, marker=marker, c=col, label=mp)


# add labels
blue_line = mlines.Line2D([], [], color='blue', marker='^',
                          markersize=12, label='baseline', linestyle="none")
red_line = mlines.Line2D([], [], color='red', marker='^',
                          markersize=12, label='baseline', linestyle="none")
plt.legend(handles=[blue_line, red_line], bbox_to_anchor=(1.05, 1), loc = 2)


fig.savefig("test", bbox_inches="tight")



