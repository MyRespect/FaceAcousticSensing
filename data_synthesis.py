import os
import random
import numpy as np 
import pandas as pd
from dtaidistance import dtw

def read_file(file_name):
    sound_2d = pd.read_csv(file_name, dtype='float', sep =' ', \
        header= None, on_bad_lines='skip').dropna() # drop NaN
    sound_1d = sound_2d.to_numpy().flatten()
    return sound_1d

def data_synthesis(folder_path, target, ppl_num, intra_aug=True, K=3):
    """
    folder_path: synthesis files from main person
    folder_path_1: files from other people
    intra_aug: data augmentation within main person, otherwise cross-people
    K: the number of neighbor-people
    """
    save_path = folder_path.replace("lab", "syntheis")
    os.makedirs(save_path, exist_ok=True)
    file_list = sorted(os.listdir(folder_path))
    data_list = []
    for file in file_list:
        file_path = os.path.join(folder_path, file)
        sound_1d = read_file(file_path)
        data_list.append(sound_1d)
    print("Obtained the target data list")
    weight = 0.6 # random.uniform(0, 1)
    if intra_aug == True:
        min_len = min([len(x) for x in data_list])
        aligned_data_list = [x[0:min_len] for x in data_list]
        for idx, sound_1d in enumerate(aligned_data_list):
            syn_data = weight*sound_1d+(1-weight)/(len(aligned_data_list))*sum(aligned_data_list)
            syn_data = syn_data.reshape((-1, 1200))
            file_save_path = os.path.join(save_path, file_list[idx])
            np.savetxt(file_save_path, syn_data, fmt='%.1d')
    else:
        for idx, sound_1d in enumerate(data_list):
            pickup_idx = random.randrange(0, ppl_num)
            pickup_ppl = ppl_list[pickup_idx]
            folder_path_1 = folder_path.replace(target, pickup_ppl)
            print("Selected folder: ", folder_path_1)
            data_list_1=[]
            for file in sorted(os.listdir(folder_path_1)):
                file_path = os.path.join(folder_path_1, file)
                sound_1d = read_file(file_path)
                data_list_1.append(sound_1d)
            print("Loaded the data from selected folder.")
            dis_list = []
            for comp_1d in data_list_1:
                dis_list.append(dtw.distance(sound_1d[4800:5800], comp_1d[4800:6000])) # set to mitigate slow...
            # find the indices for k smallest elements
            k_idx = sorted(range(len(dis_list)), key=lambda sub: dis_list[sub])[:K] 
            neighbor_list = [data_list_1[k] for k in k_idx]
            neighbor_list.append(sound_1d)
            min_len = min([len(x) for x in neighbor_list])
            neighbor_list = [x[0:min_len] for x in neighbor_list]
            syn_data = weight*sound_1d+(1-weight)/(len(neighbor_list))*sum(neighbor_list)
            syn_data = syn_data.reshape((-1, 1200))
            file_save_path = os.path.join(save_path, file_list[idx])
            np.savetxt(file_save_path, syn_data, fmt='%.1d')
            print("Write to file: ", file_save_path)

def write_file(rootpath = './dataset/', ppl_list=None, intra_aug=True, K=4):
    """
    wear1 and wear2 control whether include w/ mask and w/o mask data
    The dataset folder structure: ./dataset/people/place/emotion/mask/file
    """
    for idx, people in enumerate(sorted(os.listdir(rootpath))):
        if people in ppl_list:
            for place in os.listdir(os.path.join(rootpath, people)):
                if place == "synthesis":
                    continue
                place_path = os.path.join(rootpath, people, place)
                for emotion_folder in os.listdir(place_path):
                    emotion_path = os.path.join(place_path, emotion_folder)
                    open_emotion_path = os.path.join(emotion_path, "open")
                    mask_emotion_path = os.path.join(emotion_path, "mask")
                    data_synthesis(open_emotion_path, people, len(ppl_list), intra_aug, K)
                    data_synthesis(mask_emotion_path, people, len(ppl_list), intra_aug, K)
    print("Synthesized data is written into file.")


if __name__ == "__main__":
    ppl_list = ['han', 'maolin'] # test users
    write_file(rootpath='./tmp_dataset', ppl_list=ppl_list, intra_aug=False)