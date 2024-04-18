from plotly.subplots import make_subplots
import plotly.graph_objects as go
import csv
import numpy as np
import os
import random
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

timestep = 50


def csv2np(filename):
    raw_data = []
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for i, row in enumerate(spamreader):
            raw_data.append(row)
    return np.array(raw_data, dtype='float32')


filename = 'original'
split_data = csv2np(filename+'.csv')
split_iteration = np.hsplit(split_data, timestep)
split_iteration = np.array(split_iteration)
print(split_iteration.shape)
iteration_column = np.reshape(
    np.arange(split_iteration.shape[0]), (split_iteration.shape[0], 1))
df = pd.DataFrame(data=np.repeat(iteration_column,
                                 split_iteration.shape[1], axis=0), columns=['Iteration'])
mean_cost = [np.mean(it, axis=1) for it in split_iteration]
mean_cost = np.transpose(np.array(mean_cost))
mean_cost = np.clip(mean_cost, 0.01, 10)
df.loc[:, 'Original w/o filter'] = pd.Series(np.squeeze(np.reshape(
    np.transpose(mean_cost), (1, -1))), index=df.index)

# fig = go.Figure()
# mean_cost = split_data
# t = np.arange(0, mean_cost.shape[1])

# fig.add_trace(go.Scatter(x=t, y=mean_cost[0], mode='lines'))
# fig.add_trace(go.Scatter(x=t, y=mean_cost[1], mode='lines'))
# fig.add_trace(go.Scatter(x=t, y=mean_cost[2], mode='lines'))
# fig.add_trace(go.Scatter(x=t, y=mean_cost[3], mode='lines'))
# fig.add_trace(go.Scatter(x=t, y=mean_cost[4], mode='lines'))
# fig.add_trace(go.Scatter(x=t, y=mean_cost[5], mode='lines'))
# fig.add_trace(go.Scatter(x=t, y=mean_cost[6], mode='lines'))
# fig.show()


# fig = go.Figure()

# fig.add_trace(go.Scatter(x=t, y=mean_cost, mode='lines', name='original', marker={
#               'color': 'rgb(255,0,0)'}, line=dict(width=4, dash='dash')))

filename = 'u_cost'
split_data = csv2np(filename+'.csv')
split_iteration = np.hsplit(split_data, timestep)
split_iteration = np.array(split_iteration)
mean_cost = [np.mean(it, axis=1) for it in split_iteration]
mean_cost = np.transpose(np.array(mean_cost))
mean_cost = np.clip(mean_cost, 0.01, 10)
df.loc[:, 'Original w/ action cost'] = pd.Series(np.squeeze(np.reshape(
    np.transpose(mean_cost), (1, -1))), index=df.index)

filename = 'original_filter_on_noise'
split_data = csv2np(filename+'.csv')
split_iteration = np.hsplit(split_data, timestep)
split_iteration = np.array(split_iteration)
mean_cost = [np.mean(it, axis=1) for it in split_iteration]
mean_cost = np.transpose(np.array(mean_cost))
mean_cost = np.clip(mean_cost, 0.01, 10)
df.loc[:, 'Original (SGF(\u03B5))'] = pd.Series(np.squeeze(np.reshape(
    np.transpose(mean_cost), (1, -1))), index=df.index)

filename = 'original_filter_on_u'
split_data = csv2np(filename+'.csv')
split_iteration = np.hsplit(split_data, timestep)
split_iteration = np.array(split_iteration)
mean_cost = [np.mean(it, axis=1) for it in split_iteration]
mean_cost = np.transpose(np.array(mean_cost))
mean_cost = np.clip(mean_cost, 0.01, 10)
df.loc[:, 'Original (SGF(u))'] = pd.Series(np.squeeze(np.reshape(
    np.transpose(mean_cost), (1, -1))), index=df.index)


filename = 'das'
split_data = csv2np(filename+'.csv')
split_iteration = np.hsplit(split_data, timestep)
split_iteration = np.array(split_iteration)
mean_cost = [np.mean(it, axis=1) for it in split_iteration]
mean_cost = np.transpose(np.array(mean_cost))
mean_cost = np.clip(mean_cost, 0.01, 10)
df.loc[:, 'Ours'] = pd.Series(np.squeeze(np.reshape(
    np.transpose(mean_cost), (1, -1))), index=df.index)


#sns.lineplot(data=df, x="iteration", y="Ours", ci=50, legend='full')
#sns.lineplot(data=df, x="iteration", y="Original", ci=50, legend='full')
df = df.melt('Iteration', var_name='Method', value_name='State cost')
plt.figure(figsize=[6, 4])

sns.set_palette(reversed(sns.color_palette('Set1', 5)), 5)
sns.lineplot(data=df, x='Iteration', y='State cost', hue='Method', ci=80)
plt.yscale('log')
plt.tight_layout()
plt.xlim(0, 35)
plt.ylim(0.009, 10)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel(xlabel='Iteration', fontsize=12)
plt.ylabel(ylabel='State cost', fontsize=12)
plt.legend(prop={'size': 12})
plt.savefig('/home/add/Desktop/tempo.pdf', format='pdf')
# plt.legend()
plt.show()
plt.close()


# fig.update_layout(
#     xaxis_title="itertaion",
#     yaxis_title="cost",
#     height=650,
#     width=1800,
#     margin=dict(l=30, r=30, t=30, b=30),
#     font=dict(size=20),
#     legend=dict(font=dict(size=25), yanchor="top",
#                 y=0.99, xanchor="right", x=0.99)
# )
# fig.layout.template = 'plotly_white'
# fig.show()

# if not os.path.exists("image"):
#     os.mkdir("image")
# fig.write_image("./image/inference_"+filename+"_vx.svg")
