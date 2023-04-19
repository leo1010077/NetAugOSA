import numpy as np
import random
import torch
import glob
import torch.utils.data as data
from torchvision import transforms
import math
import librosa

class OSAnpyDataset_audio_mel_3d(data.Dataset):
    def __init__(self, floader_path, folder_path_audio, train):
        self.floader_path = floader_path + '/*/'
        self.folder_audio_path = folder_path_audio + '/*/'
        self.labels = []
        self.datas = []
        self.datas_audio = []
        self.train = train
        snore_path = self.floader_path + 'snore'
        bs_path = self.floader_path + 'BS'
        noisepath = self.floader_path + 'noise'
        hypspath = self.floader_path + 'HS'
        snorefile_list = glob.glob(snore_path + "/*.npy")
        bsfile_list = glob.glob(bs_path + "/*.npy")
        hypsfile_list = glob.glob(hypspath + "/*.npy")
        noisefile_list = glob.glob(noisepath + "/*.npy")
        alllist = [snorefile_list, bsfile_list, hypsfile_list, noisefile_list]# hypsfile_list, noisefile_list]# , noisefile_list,hypsfile_list
        for listt in alllist:
            if listt == bsfile_list:  # load boomsnore file
                for i in range(len(listt)):
                    self.datas.append(listt[i])
                    self.labels.append(0)
            elif listt == snorefile_list:  # load snore file
                for i in range(len(listt)):
                    self.datas.append(listt[i])
                    self.labels.append(2)
            elif listt == hypsfile_list:  # load hyp file
                for i in range(len(listt)):
                    self.datas.append(listt[i])
                    self.labels.append(1)
            else:  # load noise file
                for i in range(len(listt)):
                    self.datas.append(listt[i])
                    self.labels.append(3)
        snore_path = self.folder_audio_path + 'snore'
        bs_path = self.folder_audio_path + 'BS'
        noisepath = self.folder_audio_path + 'noise'
        hypspath = self.folder_audio_path + 'HS'
        snorefile_list = glob.glob(snore_path + "/*.wav")
        bsfile_list = glob.glob(bs_path + "/*.wav")
        hypsfile_list = glob.glob(hypspath + "/*.wav")
        noisefile_list = glob.glob(noisepath + "/*.wav")
        alllist = [snorefile_list, bsfile_list, hypsfile_list,
                   noisefile_list]  # hypsfile_list, noisefile_list]# , noisefile_list,hypsfile_list
        for listt in alllist:
            if listt == bsfile_list:  # load boomsnore file
                for i in range(len(listt)):
                    self.datas_audio.append(listt[i])
            elif listt == snorefile_list:  # load snore file
                for i in range(len(listt)):
                    self.datas_audio.append(listt[i])
            elif listt == hypsfile_list:  # load hyp file
                for i in range(len(listt)):
                    self.datas_audio.append(listt[i])
            else:  # load noise file
                for i in range(len(listt)):
                    self.datas_audio.append(listt[i])
    def __len__(self):

        return len(self.datas)

    def __getitem__(self, idx):

        file_path, audio_path, labels = self.datas[idx], self.datas_audio[idx], self.labels[idx]
        #audio = librosa.core.load(audio_path, sr=22500, mono=True, res_type='kaiser_fast')[0]
        #audio = audio[:180000]
        #audio = np.expand_dims(audio, axis=0)
        image = np.load(file_path)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)
        # image = np.concatenate([image, image, image], 0)
        return image, labels

    def check_size(self):
        image, label = self.__getitem__(1)
        return image.shape
