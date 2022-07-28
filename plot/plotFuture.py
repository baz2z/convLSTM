import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import numpy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def f(x):
    return {
        0.5: 1,
        1: 2,
        2: 4
    }[x]

def matchColor(modelName):
    return{
        "baseline": "blue",
        "lateral": "red",
        "twoLayer":"green",
        "skip": "purple",
        "depthWise": "chocolate"
    }[modelName]
    pass
def matchMarker(multiplier):
    return{
        "100_40":"+",
        "100_170":"o",
        "100_270": "s"
    }[multiplier]
def mapXAchse(name):
    return{
        "baseline": 0,
        "lateral": 1,
        "twoLayer": 2,
        "skip": 3,
        "depthWise": 4
    }[name]

def matchColorAd(future):
    return{
        20: "blue",
        40: "red",
        70: "green"
    }[future]

fig, ax = plt.subplots()

df = pd.read_csv("../test/futureLoss")
df.reset_index()

for index, row in df.iterrows():
    modelName = row["modelName"]
    future = row["future"]
    loss100_40 = row["loss100_40"]
    loss100_170 = row["loss100_170"]
    loss100_270 = row["loss100_270"]
    x = mapXAchse(modelName)
    col = matchColorAd(future)
    ax.scatter(x, loss100_40, color=col, s=16, alpha=0.7, marker=matchMarker("100_40"))
    ax.scatter(x, loss100_170, color=col, s=16, alpha=0.7, marker=matchMarker("100_170"))
    ax.scatter(x, loss100_270, color=col, s=16, alpha=0.7, marker=matchMarker("100_270"))


ax.set_yscale('log')
blue_line = mlines.Line2D([], [], color='blue', marker='o',
                          markersize=12, label='trained on 20', linestyle="none")
red_line = mlines.Line2D([], [], color='red', marker='o',
                          markersize=12, label='40', linestyle="none")
green_line = mlines.Line2D([], [], color='green', marker='o',
                          markersize=12, label='70', linestyle="none")

marker3 = mlines.Line2D([], [], color='gray', marker='+',
                          markersize=12, label='start: 100, horizon: 40', linestyle="none")
marker4 = mlines.Line2D([], [], color='gray', marker='o',
                          markersize=12, label='start: 100, horizon: 170', linestyle="none")
marker6 = mlines.Line2D([], [], color='gray', marker='s',
                          markersize=12, label='start: 100, horizon: 270', linestyle="none")

#plt.xticks([20, 40, 70], ["True", "False"])
plt.legend(handles=[blue_line, red_line, green_line, marker3, marker4, marker6], bbox_to_anchor=(1.05, 1), loc = 2)
#plt.ylim([0.00001, 0.01])
name = f'./createdPlots/futureLoss'
plt.xticks([0, 1, 2, 3, 4], ["Baseline", "Lateral", "TwoLayer", "Skip", "DepthWise"])
plt.xlabel("model name")
plt.ylabel("loss")
fig.savefig(name, bbox_inches="tight")





# models not x-achse:
"""
import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import numpy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def f(x):
    return {
        0.5: 1,
        1: 2,
        2: 4
    }[x]

def matchColor(modelName):
    return{
        "baseline": "blue",
        "lateral": "red",
        "twoLayer":"green",
        "skip": "purple",
        "depthWise": "chocolate"
    }[modelName]
    pass
def matchMarker(multiplier):
    return{
        "100_40":"+",
        "100_170":"o",
        "100_270": "s"
    }[multiplier]
def mapXAchse(name):
    return{
        "baseline": 0,
        "lateral": 1,
        "twoLayer": 2,
        "skip": 3,
        "depthWise": 4
    }[name]

fig, ax = plt.subplots()

df = pd.read_csv("../test/futureLoss")
df.reset_index()

for index, row in df.iterrows():
    modelName = row["modelName"]
    future = row["future"]
    loss100_40 = row["loss100_40"]
    loss100_170 = row["loss100_170"]
    loss100_270 = row["loss100_270"]
    col = matchColor(modelName)
    x = mapXAchse(modelName)
    ax.scatter(future, loss100_40, color=col, s=16, alpha=0.7, marker=matchMarker("100_40"))
    ax.scatter(future, loss100_170, color=col, s=16, alpha=0.7, marker=matchMarker("100_170"))
    ax.scatter(future, loss100_270, color=col, s=16, alpha=0.7, marker=matchMarker("100_270"))


ax.set_yscale('log')
blue_line = mlines.Line2D([], [], color='blue', marker='o',
                          markersize=12, label='baseline', linestyle="none")
red_line = mlines.Line2D([], [], color='red', marker='o',
                          markersize=12, label='lateral', linestyle="none")
green_line = mlines.Line2D([], [], color='green', marker='o',
                          markersize=12, label='twoLayer', linestyle="none")
purple_line = mlines.Line2D([], [], color='purple', marker='o',
                          markersize=12, label='skip', linestyle="none")
chocolate_line = mlines.Line2D([], [], color='chocolate', marker='o',
                          markersize=12, label='depthWise', linestyle="none")

marker3 = mlines.Line2D([], [], color='gray', marker='+',
                          markersize=12, label='start: 100, horizon: 40', linestyle="none")
marker4 = mlines.Line2D([], [], color='gray', marker='o',
                          markersize=12, label='start: 100, horizon: 170', linestyle="none")
marker6 = mlines.Line2D([], [], color='gray', marker='s',
                          markersize=12, label='start: 100, horizon: 270', linestyle="none")

#plt.xticks([20, 40, 70], ["True", "False"])
plt.legend(handles=[blue_line, red_line, green_line, purple_line, chocolate_line, marker3, marker4, marker6], bbox_to_anchor=(1.05, 1), loc = 2)
#plt.ylim([0.00001, 0.01])
name = f'./createdPlots/futureLoss'
plt.xlabel("trained on x future frames")
plt.ylabel("loss")
fig.savefig(name, bbox_inches="tight")

"""
