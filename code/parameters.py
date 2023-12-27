import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if device == 'cpu':
    root = '/home/ruixing/workspace/sinea/physionet.org/files/siena-scalp-eeg'
else: root = '/home/liruixin/workspace/sinea'

trail_time = 5
sr = 250
num_channel = 4
time_step = 1250
num_intervals = 1000
batch_size = 256
epochs = 1000
num_class = 2