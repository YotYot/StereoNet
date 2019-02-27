from dfd import Dfd_net
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import tqdm

def show_losses():
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

def show_error_to_depth(pickle_file):
    with open(pickle_file,'rb') as f:
        hist = pickle.load(f)
    depths = list()
    errs = list()
    for i in range(len(hist)):
        depths.append(hist[i][0][0])
        errs.append(hist[i][0][1])
    depths = torch.cat(depths).cpu().numpy()
    print ("Length: ", len(depths))
    errs = torch.cat(errs).cpu().numpy()
    print (" Min depth: ", np.min(depths), " Max depth: ", np.max(depths), "\n")
    # depths = depths[:1000000]
    depths_err_dict = dict()
    for idx, dpt in tqdm.tqdm(enumerate(depths)):
        depths_err_dict[dpt.item()] = 0
    for idx, dpt in tqdm.tqdm(enumerate(depths)):
        depths_err_dict[dpt.item()] += errs[idx]
    depths_keys, errs = zip(*sorted(zip(depths_err_dict.keys(), depths_err_dict.values())))
    depths = torch.Tensor(depths).long()
    depth_histo = torch.bincount(depths)
    rel_err = list()
    for i in range(len(errs)):
        rel_err.append(errs[i] / (depth_histo[depths_keys[i]]).float())
        # with open('aaa.pickle', 'wb') as f:
    #     pickle.dump(depths_err_dict,f)

    return depths_keys,rel_err

from sintel_io import  depth_read
def get_depth_stats(depth_dir):
    depth_min = 10000
    depth_max = 0
    for file in os.listdir(depth_dir):
        filepath = os.path.join(depth_dir,file)
        dpt = depth_read(filepath)
        if (np.max(dpt) > depth_max):
            depth_max = np.max(dpt)
        if (np.min(dpt) < depth_min):
            depth_min = np.min(dpt)
    return depth_min, depth_max

print (get_depth_stats('/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Tau-agent/right_images/original_depth/'))
mono_depth_keys_1100,mono_rel_err_1100 = show_error_to_depth('error_mono_1100.pickle')
mono_depth_keys_700,mono_rel_err_700 = show_error_to_depth('error_mono_700.pickle')
stereo_depth_keys,stereo_rel_err = show_error_to_depth('error_stereo.pickle')
plt.plot(mono_depth_keys_1100,mono_rel_err_1100, label="Mono 1100")
plt.plot(mono_depth_keys_700,mono_rel_err_700, label="Mono 700")
plt.plot(stereo_depth_keys,stereo_rel_err, label="Stereo")
plt.legend()
plt.show()
print( "DONE")