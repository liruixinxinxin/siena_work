import matplotlib.pyplot as plt
import numpy as np
import pickle
import mne
import os


from function import *
from parameters import *
from pathlib import Path
from tqdm.auto import tqdm 
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class Mydataset(Dataset):
    def __init__(self,data_list,label_list):
        self.data_list = data_list
        self.label_list = label_list
    def __getitem__(self,index):
        return self.data_list[index],self.label_list[index]
    def __len__(self):
        return len(self.data_list)



if __name__ == "__main__":
    # generate the seizure time to dict 
    time_dict = {}
    dir = Path(os.path.join(root,'siena-scalp-eeg/data'))
    for file in dir.rglob('*.txt'):
        seizure_list = []
        with open(str(file),'r') as f:
            a = 0
            b = 0
            c = 0
            for line in f:
                if 'Registration start time: ' in line :
                    a = 1
                    start_index0 = line.find('Registration start time: ') + len('Registration start time: ')
                    Registration_start_time = line[start_index0:start_index0 + 8]
                if 'Seizure start time: ' in line :
                    b = 1
                    start_index1 = line.find('Seizure start time: ') + len('Seizure start time: ')
                    seizure_start_time = line[start_index1:start_index1 + 8]
                if 'Seizure end time: ' in line :
                    c = 1
                    start_index2 = line.find('Seizure end time: ') + len('Seizure end time: ')
                    seizure_end_time = line[start_index2:start_index2 + 8]
                if (a & b & c ) == 1:
                    begin_time_s = time_difference(Registration_start_time,seizure_start_time)
                    end_time_s = begin_time_s + time_difference(seizure_start_time,seizure_end_time)
                    seizure_list.append([begin_time_s, end_time_s])
                    a = 0
                    b = 0
                    c = 0 
            time_dict[f'{file.parts[-1][-8:-4]}'] = seizure_list



    picks1= ['EEG C3']
    picks2= ['EEG C4']
    picks3= ['EEG P3']
    picks4= ['EEG P4']
    picks_list = list([picks1, picks2])
    # extre_value = []
    spike_list = []
    label_list = []
    # extract the pos signal 
    for i in tqdm(sorted(dir.rglob('*.edf'))):
        data = mne.io.read_raw_edf(str(i),preload=True)
        data_index_list = extract_numbers(i.parts[-1])
        for m in data_index_list:        
            tmin = int((time_dict[i.parts[-1][0:4]])[m-1][0])+10
            tmax = tmin + 30
            # data = data.filter(l_freq=None,h_freq=30)
            sampling_frequency = sr
            data = data.resample(sfreq=sampling_frequency)
            crop = np.arange(tmin,tmax,5)
            one_trail_data = []
            p = 1
            while(1):
                data_one_trail = data.copy().crop(tmin=tmin,tmax=tmin+5)
                for j in picks_list:
                    data_one_channel =  data_one_trail.get_data(units='uV',picks=j)
                    one_trail_data.append(data_one_channel)
                one_trail_data = np.asarray(one_trail_data)
                spike_array = sigma_delta_encoding(data = one_trail_data.squeeze(),
                                                                num_intervals = num_intervals,
                                                                min = -4000,
                                                                max = 4000
                                                                )
                spike_array[spike_array > 15] = 15
                spike_array = spike_array.reshape(num_channel, time_step)
                spike_list.append(spike_array)
                label_list.append(1)
                one_trail_data = []
                tmin = tmin + 5
                if(tmin > (tmax-5)):
                    break
                p += 1
    pass                        


    # extract the neg signal 
    for i in tqdm(sorted(dir.rglob('*.edf'))):
        data = mne.io.read_raw_edf(str(i),preload=True)
        data_index_list = extract_numbers(i.parts[-1])
        for m in data_index_list:        
            tmin = int((time_dict[i.parts[-1][0:4]])[m-1][0]) - 30
            if tmin < 0:
                tmin = 0
            tmax = int((time_dict[i.parts[-1][0:4]])[m-1][0])
            # data = data.filter(l_freq=None,h_freq=30)
            sampling_frequency = sr
            data = data.resample(sfreq=sampling_frequency)
            crop = np.arange(tmin,tmax,5)
            one_trail_data = []
            p = 1
            while(1):
                data_one_trail = data.copy().crop(tmin=tmin,tmax=tmin+5)
                for j in picks_list:
                    data_one_channel =  data_one_trail.get_data(units='uV',picks=j)
                    one_trail_data.append(data_one_channel)
                one_trail_data = np.asarray(one_trail_data)
                spike_array = sigma_delta_encoding(data = one_trail_data.squeeze(),
                                                                num_intervals = num_intervals,
                                                                min = -4000,
                                                                max = 4000
                                                                )
                spike_array[spike_array > 15] = 15
                spike_array = spike_array.reshape(num_channel, time_step)
                spike_list.append(spike_array)
                label_list.append(0)
                one_trail_data = []
                tmin = tmin + 5
                if(tmin > (tmax-5)):
                    break
                p += 1
    pass  


    train_data, test_data, train_labels, test_labels = train_test_split(
                                                        spike_list, 
                                                        label_list, 
                                                        test_size=0.2, 
                                                        random_state=43)


    train_dataset = Mydataset(train_data,train_labels)
    test_dataset = Mydataset(test_data,test_labels)

    with open(os.path.join(root,'dataset/train_dataset.pkl'),'wb') as file:
        pickle.dump(train_dataset,file)

    with open(os.path.join(root,'dataset/test_dataset.pkl'),'wb') as file:
        pickle.dump(test_dataset,file)


