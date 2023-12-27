# - Numpy
import numpy as np
import torch
import pickle
# - Matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12, 4]
plt.rcParams['figure.dpi'] = 300

# - Rockpool time-series handling
from rockpool import TSEvent, TSContinuous
import torch
import os
from rockpool.nn.modules import LinearTorch, LIFTorch
from rockpool.parameters import Constant
from rockpool.nn.combinators import Sequential
# - Pretty printing
try:
    from rich import print
except:
    pass

# - Display images
from IPython.display import Image

# - Disable warnings
import warnings
warnings.filterwarnings('ignore')
# - Import the computational modules and combinators required for the networl
from rockpool.nn.modules import LIFTorch, LinearTorch
from rockpool.nn.combinators import Sequential, Residual
from rockpool.transform import quantize_methods as q
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm.asyncio import tqdm
from rockpool.nn.networks.wavesense import WaveSenseNet
from rockpool.devices import xylo as x
from parameters import *
from network import *
net = net
net.load('/home/liruixin/workspace/sinea/models/modelmix_spike_146.pth')
net = net.to(device)
# net = toyKWSNet()




g = net.as_graph()
spec = x.vA2.mapper(g, weight_dtype='float', threshold_dtype='float', dash_dtype='float')
quant_spec = spec.copy()
# - Quantize the specification
spec.update(q.global_quantize(**spec))
# - Use rockpool.devices.xylo.config_from_specification
config, is_valid, msg = x.vA2.config_from_specification(**spec)
modSim = x.vA2.XyloSim.from_config(config)
pass
# out, _, rec = modSim.evolve(input_raster=np.zeros((10, 16)))

# load data

with open(os.path.join(root,'dataset/test_dataset.pkl'),'rb') as file:
    test_dataset = pickle.load(file)

test_dataloader = DataLoader(test_dataset,batch_size=len(test_dataset))
spiking_test_dataloader = DataLoader(test_dataset,batch_size=1,shuffle=True)

n_0 = 0
n_1 = 0
consequence_list = []
for data,label in tqdm(spiking_test_dataloader,colour='yellow'):
    data.to(device)
    label.to(device)
    modSim.reset_state()
    data = torch.reshape(data,(time_step,num_channel))
    data = data.numpy()
    data = data.astype(int)
    output, state, recordings = modSim((data).clip(0, 15),record=True,read_timeout=10)
    out = output.squeeze()
    # print(np.any(out))
    peaks = out.max(0)
    result = peaks.argmax()
    print('peaks:',peaks)
    print('result:',result)
    print('label:',label)
    if result.item() == 0:
        n_0  += 1
    if result.item() == 1:
        n_1  += 1
    # result.to(device)
    consequence = (result==label.item())
    consequence_list.append(consequence)
    
acc = sum(consequence_list)/len(consequence_list)
print(f'accuracy:{acc}')
print(f'number of zero:{n_0}，number of one:{n_1}')