import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd

def matchLinestyle(multiplier):
    return{
        0.5:":",
        1:"--",
        2:"-",
        4:":"
    }[multiplier]

def matchColor(param):
    return{
        1: "blue",
        2: "red",
        3: "green",
    }[param]


fig, ax = plt.subplots()

df = pd.read_csv("../test/allLoss-horizon")
df.reset_index()

modelToPlot = "baseline"
for index, row in df.iterrows():
    modelName = row["modelName"]
    if modelName == modelToPlot:
        mult = row["mp"]
        param = row["paramLevel"]
        horizonLoss = row["lossHorizon"].split(" ")
        ### get params exactly
        style = matchLinestyle(mult)
        col = matchColor(param)
        horizonLoss[0] = horizonLoss[0][1:]
        horizonLoss[len(horizonLoss)-1] = horizonLoss[len(horizonLoss)-1][:-1]
        horizonLoss = list(filter(lambda x: x != "", horizonLoss))
        horizonLoss = list(map(float, horizonLoss))
        ax.plot(list(range(170)), horizonLoss, style,color=col)


# param level
blue_line = mlines.Line2D([], [], color='blue', marker='o',
                          markersize=12, label='5k params', linestyle="none")
red_line = mlines.Line2D([], [], color='red', marker='o',
                          markersize=12, label='25k params', linestyle="none")
green_line = mlines.Line2D([], [], color='green', marker='o',
                          markersize=12, label='50k params', linestyle="none")

# multiplier
mult1 = mlines.Line2D([], [], color='gray',
                          markersize=12, label='2:1 (hs:ls)', linestyle=":")
mult2 = mlines.Line2D([], [], color='gray',
                          markersize=12, label='1:1', linestyle="--")
mult3 = mlines.Line2D([], [], color='gray',
                          markersize=12, label='1:2', linestyle="-")
mult4 = mlines.Line2D([], [], color='gray',
                          markersize=12, label='1:4 (multiplication)', linestyle=":")

ax.set_yscale("log")
plt.legend(handles=[blue_line, red_line, green_line, mult1, mult2, mult3, mult4], bbox_to_anchor=(1.05, 1), loc = 2)
fig.suptitle(f'{modelToPlot}', fontsize=16)
name = f'./createdPlots/all-horizonLoss-{modelToPlot}'
fig.savefig(name, bbox_inches="tight")

