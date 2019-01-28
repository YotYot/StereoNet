from dfd import Dfd_net
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
clean_log = '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Tau-agent/Right_images/clean_losses.log'
filtered_log = '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Tau-agent/Right_images/filtered_losses.log'
confs_log = '/home/yotamg/PycharmProjects/PSMNet/run_confs_to_100.log'

with open(confs_log, 'rb') as f:
    confs = f.readlines()

# with open(filtered_log, 'rb') as f:
#     filtered = f.readlines()
clean_losses = [float(line.split(" ")[-1]) for line in clean]
filtered_losses  =[float(line.split(" ")[-1]) for line in filtered]

plt.plot(clean_losses, label="Clean")
plt.plot(filtered_losses, label="Filtered")
plt.legend()
# plt.yscale('log')
plt.show()
print( "DONE")