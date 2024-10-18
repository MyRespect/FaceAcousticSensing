import os
import torch, pickle
import numpy as np 
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from scipy import signal
from torchvision import transforms

from utils import TwoCropTransform, WeightAverage

def read_record(file_name = "./signal_1200.txt"):
    """
    signal_1200.txt is a direct path signal file
    """
    signal = np.loadtxt(file_name).tolist()
    return signal

def read_file(file_name):
    sound_2d = pd.read_csv(file_name, dtype='float', sep =' ', \
        header= None, on_bad_lines='skip').dropna() # drop NaN
    sound_1d = sound_2d.to_numpy().flatten()

    # remove direct path interference -- ignore this during data augmentation
    raw_signal = read_record()
    raw_signal = raw_signal*int(len(sound_1d)/1200)
    raw_signal = np.array(raw_signal)
    sound_1d = sound_1d - 0.9*raw_signal

    sound_1d_list = []
    for i in range(0, len(sound_1d), 6000):
        # window size is 0.025*10 = 0.25s
        # no sliding windows is applied considering short time period
        # todo: test with sliding windows
        # the sliding window decides the number of samples
        if i+12000<len(sound_1d):
            sound_1d_list.append(sound_1d[i: i+12000])
    if len(sound_1d_list) > 1: # some file is empty, getting []
        return sound_1d_list[0: len(sound_1d_list)-1] 
    else:
        return sound_1d_list

def butterworth_filter(data, order = 5, Wn = [19000, 19500], btype = "bandpass", 
                       analog = False, output = 'sos', fs = 48000):

    sos = signal.butter(order, Wn, btype, analog, output, fs)
    filtered = signal.sosfilt(sos, data)
    return filtered

def STFT_transform(data, fs=48000, nperseg = 1200):
    f, t, Zxx = signal.stft(data, fs = 48000, nperseg = 1200)
    return f, t, Zxx

def process_file(file_path): 
    sound_filtered_list = []
    for file_name in os.listdir(file_path): # folder/.txt file
        if not os.path.isfile(file_path+'/'+file_name):
            continue
        sound_1d_list = read_file(file_path+'/'+file_name)
        for sound_1d in sound_1d_list: # one txt file
            sound_filtered = butterworth_filter(sound_1d)
            f, t, Zxx = STFT_transform(sound_filtered)
            tmp_abs = np.abs(Zxx[470:550]).astype('float32') # original 470:550
            # print(np.max(tmp_abs), np.min(tmp_abs)) # 1.1508847e+28 1.9409681e-07

            # deal with overflow encounted
            tmp_abs[tmp_abs > 100] = 100
            tmp_abs[tmp_abs < 0.001] = 0.001
            tmp_abs1 = np.expand_dims(tmp_abs, axis=0)
            sound_filtered_list.append(tmp_abs1)
    return sound_filtered_list

def read_folder(rootpath = './dataset/', wear1 = 'open', wear2 = None, ppl_list=None, use_synthesis=False):
    """
    wear1 and wear2 control whether include w/ mask and w/o mask data
    The dataset folder structure: ./dataset/people/place/emotion/mask/file
    """
    dataset = {}
    for people in os.listdir(rootpath):
        if people in ppl_list:
            for place in os.listdir(os.path.join(rootpath, people)): # this inlcudes synthesis folder
                place_path = os.path.join(rootpath, people, place)
                if use_synthesis == False and place == "synthesis":
                    continue
                for emotion_folder in os.listdir(place_path):
                    emotion_path = os.path.join(place_path, emotion_folder)
                    spect_imgs1 = []
                    spect_imgs2 = []
                    if wear1 != None:
                        if not os.path.exists(emotion_path+'/'+ wear1):
                            continue
                        spect_imgs1 = process_file(emotion_path+'/'+ wear1)
                        # spect_imgs1 = list(zscore(spect_imgs1, axis = None)) # (x-mu)/sigma, now using whole dataset normalize instead.
                    if wear2 != None:
                        if not os.path.exists(emotion_path+'/'+ wear2):
                            continue
                        spect_imgs2 = process_file(emotion_path+'/'+ wear2)
                    if emotion_folder == "anger":
                        if '0' in dataset.keys():
                            dataset['0'].extend(spect_imgs1) # append() to separate ppl
                            dataset['0'].extend(spect_imgs2)
                        else:
                            dataset['0']=spect_imgs1
                            dataset['0'].extend(spect_imgs2)
                    elif emotion_folder == "disgust":
                        if '1' in dataset.keys():
                            dataset['1'].extend(spect_imgs1)
                            dataset['1'].extend(spect_imgs2)
                        else:
                            dataset['1']=spect_imgs1
                            dataset['1'].extend(spect_imgs2)
                    elif emotion_folder == "fear":
                        if '2' in dataset.keys():
                            dataset['2'].extend(spect_imgs1)
                            dataset['2'].extend(spect_imgs2)
                        else:
                            dataset['2']=spect_imgs1
                            dataset['2'].extend(spect_imgs2)
                    elif emotion_folder == "happiness":
                        if '3' in dataset.keys():
                            dataset['3'].extend(spect_imgs1)
                            dataset['3'].extend(spect_imgs2)
                        else:
                            dataset['3']=spect_imgs1
                            dataset['3'].extend(spect_imgs2)
                    elif emotion_folder == "sadness":
                        if '4' in dataset.keys():
                            dataset['4'].extend(spect_imgs1)
                            dataset['4'].extend(spect_imgs2)
                        else:
                            dataset['4']=spect_imgs1
                            dataset['4'].extend(spect_imgs2)
                    elif emotion_folder == "surprise":
                        if '5' in dataset.keys():
                            dataset['5'].extend(spect_imgs1)
                            dataset['5'].extend(spect_imgs2)
                        else:
                            dataset['5']=spect_imgs1
                            dataset['5'].extend(spect_imgs2)
    return dataset

class FaceEmotionDataset(Dataset):
    def __init__(self, rootpath = './dataset/', wear1 = None, wear2 = None,\
        ppl_list=None, use_synthesis=False, local_data_name = None, saved = False, transform = None):
        self.X = []
        self.y = []
        self.transform = transform
        if saved == False:
            dataset_dict = read_folder(rootpath, wear1, wear2, ppl_list, use_synthesis)
            if use_synthesis == True:
                dataset_name = "./results/data/"+local_data_name+'_wSyn.pk'
            else:
                dataset_name = "./results/data/"+local_data_name+'.pk'
            dbfile = open(dataset_name,'wb') # save dataset dict
            pickle.dump(dataset_dict, dbfile)
            dbfile.close()
        else:
            if use_synthesis == True:
                dataset_name = "./results/data/"+local_data_name+'_wSyn.pk'
            else:
                dataset_name = "./results/data/"+local_data_name+'.pk'
            dbfile = open(dataset_name,'rb')
            dataset_dict = pickle.load(dbfile)
        for key in dataset_dict.keys():
            self.X.extend(dataset_dict[key])
            self.y.extend([int(key)] * len(dataset_dict[key]))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = torch.tensor(self.X[idx])
        label = torch.tensor(self.y[idx])
        if self.transform: # if None: ==> do not enter if
            sample = self.transform(sample)
        return sample, label
    
class XDomainDataset(Dataset):
    def __init__(self, sample_list, label_list, transform = None):
        self.X = sample_list
        self.y = label_list
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(self.X[idx]) == False:
            sample = torch.tensor(self.X[idx])
            label = torch.tensor(self.y[idx])
        else:
            sample = self.X[idx]
            label = torch.tensor(self.y[idx])
        if self.transform: # if None: ==> do not enter -if-
            sample = self.transform(sample)
        return sample, label

if __name__ == "__main__":

    ppl_list = ['bochen', 'chenhui', 'chenning', 'geoff', 'guangjing',\
                    'han', 'juexing', 'maolin', 'mark', 'nick', 'pp', \
                    'shane', 'shengyi', 'shuyang', 'thinh', 'tony', \
                        'van', 'yidong', 'yu', 'ce'] # all users
    ppl_list_0 = ['tony', 'bochen']
    wear1 = "open"
    wear2 = "mask"
    local_data_name = "dataset_all"
    use_synthesis = False
    normalize = transforms.Normalize(mean=4.8874, std=17.5538)
    train_transform = transforms.Compose([WeightAverage(), normalize])
    dataset = FaceEmotionDataset(rootpath = './dataset/', ppl_list = ppl_list, \
                                           wear1 = wear1, wear2 = wear2, use_synthesis = use_synthesis, \
                                            local_data_name = local_data_name, saved = False, \
                                                transform = TwoCropTransform(train_transform))

    print(len(dataset))
    loader = DataLoader(dataset, batch_size=len(dataset), num_workers=12, shuffle=False)
    for idx, (data, labels) in enumerate(loader):
        data = torch.cat((data[0], data[1]), dim=0)
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean = data.mean()
        std = data.std()
        print(torch.max(data), torch.min(data))
        print(mean, std)