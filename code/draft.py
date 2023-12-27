import numpy as np
import mne
import matplotlib.pyplot as plt

data = mne.io.read_raw_edf('/home/ruixing/workspace/sinea/physionet.org/files/file/siena-scalp-eeg/data/PN01/PN01-1.edf')
data.plot()
plt.show()