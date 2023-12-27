import numpy as np
import torch

from rockpool.nn.networks.wavesense import WaveSenseNet
from rockpool.nn.networks import SynNet

from rockpool.nn.modules import LIFTorch, LIFBitshiftTorch,ExpSynTorch,LIFExodus
from rockpool.parameters import Constant
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pathlib import Path
from parameters import *

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
dilations = [2, 32]
n_out_neurons = num_class
n_inp_neurons = num_channel
n_neurons = 16
kernel_size = 2
tau_mem = 0.002
base_tau_syn = 0.002
tau_lp = 0.01
threshold = 0.1
threshold_out = 0.1
dt = 0.001

net = WaveSenseNet(
    dilations=dilations,
    n_classes=n_out_neurons,
    n_channels_in=n_inp_neurons,#in_channel
    n_channels_res=n_neurons,
    n_channels_skip=n_neurons,
    n_hidden=n_neurons,
    kernel_size=kernel_size,
    bias=Constant(0.0),
    smooth_output=True,
    tau_mem=Constant(tau_mem),
    base_tau_syn=base_tau_syn,
    tau_lp=tau_lp,
    threshold=Constant(threshold),
    threshold_out = Constant(threshold_out),
    neuron_model=LIFExodus,
    dt=dt,
) 

synnet_mix = SynNet(
    n_channels=num_channel,
    n_classes=2,
    size_hidden_layers = [16,32],
    time_constants_per_layer = [1,1],
    output='spikes',
    neuron_model=LIFExodus,
    threshold = 0.01,
    threshold_out = 0.1,
    dt = 0.001
)