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

# def matchColor(param):
#     return{
#         1: "blue",
#         2: "red",
#         3: "green",
#     }[param]

def matchColor(modelName):
    return{
        "baseline": "blue",
        "lateral": "red",
        "twoLayer":"green",
        "skip": "purple",
        "depthWise": "chocolate"
    }[modelName]

fig, ax = plt.subplots()

df = pd.read_csv("../test/allLoss-horizon")
df.reset_index()

for index, row in df.iterrows():
    modelName = row["modelName"]
    mp = row["mp"]
    paramLevel = row["paramLevel"]
    horizonLoss = row["lossHorizon"].split(" ")
    ### get params exactly
    #style = matchLinestyle(mp)
    col = matchColor(modelName)
    horizonLoss[0] = horizonLoss[0][1:]
    horizonLoss[len(horizonLoss)-1] = horizonLoss[len(horizonLoss)-1][:-1]
    horizonLoss = list(filter(lambda x: x != "", horizonLoss))
    horizonLoss = list(map(float, horizonLoss))

    if modelName == "baseline":
        if mp == 1 and paramLevel == 2:
            ax.plot(list(range(270)), horizonLoss, color=col)
    elif modelName == "lateral":
        if mp == 1 and paramLevel == 2:
            ax.plot(list(range(270)), horizonLoss, color=col)
    elif modelName == "twoLayer":
        if mp == 0.5 and paramLevel == 2:
            ax.plot(list(range(270)), horizonLoss, color=col)
    elif modelName == "skip":
        if mp == 2 and paramLevel == 2:
            ax.plot(list(range(270)), horizonLoss, color=col)
        if mp == 0.5 and paramLevel == 2:
            ax.plot(list(range(270)), horizonLoss, color=col, linestyle='--')




# # param level
# blue_line = mlines.Line2D([], [], color='blue', marker='o',
#                           markersize=12, label='5k params', linestyle="none")
# red_line = mlines.Line2D([], [], color='red', marker='o',
#                           markersize=12, label='25k params', linestyle="none")
# green_line = mlines.Line2D([], [], color='green', marker='o',
#                           markersize=12, label='50k params', linestyle="none")
#
# # multiplier
# mult1 = mlines.Line2D([], [], color='gray',
#                           markersize=12, label='2:1 (hs:ls)', linestyle=":")
# mult2 = mlines.Line2D([], [], color='gray',
#                           markersize=12, label='1:1', linestyle="--")
# mult3 = mlines.Line2D([], [], color='gray',
#                           markersize=12, label='1:2', linestyle="-")
# mult4 = mlines.Line2D([], [], color='gray',
#                           markersize=12, label='1:4 (multiplication)', linestyle=":")

blue_line = mlines.Line2D([], [], color='blue', marker='o',
                          markersize=12, label='baseline', linestyle="none")
red_line = mlines.Line2D([], [], color='red', marker='o',
                          markersize=12, label='lateral 1:1 (hs:ls)', linestyle="none")
green_line = mlines.Line2D([], [], color='green', marker='o',
                          markersize=12, label='twoLayer 2:1', linestyle="none")
purple_line = mlines.Line2D([], [], color='purple', marker='o',
                          markersize=12, label='skip 1:2', linestyle="none")
chocolate_line = mlines.Line2D([], [], color='purple', markersize=12, label='skip 2:1', linestyle="--")

#ax.set_yscale("log")
ax.set_ylim([1e-08, 2e-05])
plt.legend(handles=[blue_line, red_line, green_line, purple_line, chocolate_line], bbox_to_anchor=(1.05, 1), loc = 2)
#fig.suptitle(f'{modelToPlot}', fontsize=16)
name = f'./createdPlots/all-horizonLoss-topFive'
fig.savefig(name, bbox_inches="tight")

