import glob
import cv2
import numpy as np
import torch.utils.data as data
import random
from sklearn.model_selection import train_test_split

class OSAnpyDataset_spilt(data.Dataset):
    def __init__(self, floader_path, train=False, fold=0):
        self.floader_path = floader_path + '/*/'
        self.labels = []
        self.datas = []
        self.train = train
        self.file_list = ''
        self.fold = fold
        snore_path = self.floader_path + 'snore'
        bs_path = self.floader_path + 'BS'
        noisepath = self.floader_path + 'noise'
        hypspath = self.floader_path + 'HS'
        snorefile_list = glob.glob(snore_path + "/*.png")
        bsfile_list = glob.glob(bs_path + "/*.png")
        hypsfile_list = glob.glob(hypspath + "/*.png")
        noisefile_list = glob.glob(noisepath + "/*.png")
        alllist = [snorefile_list, bsfile_list, hypsfile_list, noisefile_list]# hypsfile_list, noisefile_list]# , noisefile_list,hypsfile_list
        for listt in alllist:
            if listt == bsfile_list:  # load boomsnore file
                for i in range(len(listt)):
                    if self.train and int(listt[i].split('\\')[-3]) % 5 == fold :
                        self.datas.append(listt[i])
                        self.labels.append(0)
                    elif self.train == False and int(listt[i].split('\\')[-3]) % 5 != fold:
                        self.datas.append(listt[i])
                        self.labels.append(0)
            if listt == snorefile_list:  # load snore file
                for i in range(len(listt)):
                    if self.train and int(listt[i].split('\\')[-3]) % 5 == fold :
                        self.datas.append(listt[i])
                        self.labels.append(1)
                    elif self.train == False and int(listt[i].split('\\')[-3]) % 5 != fold:
                        self.datas.append(listt[i])
                        self.labels.append(1)
            elif listt == hypsfile_list:  # load hyp file
                for i in range(len(listt)):
                    if self.train and int(listt[i].split('\\')[-3]) % 5 == fold :
                        self.datas.append(listt[i])
                        self.labels.append(1)
                    elif self.train == False and int(listt[i].split('\\')[-3]) % 5 != fold:
                        self.datas.append(listt[i])
                        self.labels.append(1)
            else:
                for i in range(len(listt)):
                    if self.train and int(listt[i].split('\\')[-3]) % 5 == fold :
                        self.datas.append(listt[i])
                        self.labels.append(2)
                    elif self.train == False and int(listt[i].split('\\')[-3]) % 5 != fold:
                        self.datas.append(listt[i])
                        self.labels.append(2)
        # x_train, x_test, y_train, y_test = train_test_split(self.datas, self.labels, test_size=0.2, train_size=0.8, random_state=4)

    def __len__(self):

        return len(self.datas)

    def __getitem__(self, idx):

        file_path, labels = self.datas[idx], self.labels[idx]
        image = cv2.imread(file_path)
        if random.random() > 0.5 and self.train == True:
             mask_length = random.randint(6, 20)
             start_pos = random.randint(0, 64-mask_length-1)
             image[start_pos:start_pos+mask_length, :] = 0
        # if random.random() > 0.5 and self.train == True:
        #      mask_length = random.randint(16, 64)
        #      start_pos = random.randint(0, 512-mask_length-1)
        #      image[:, start_pos:start_pos+mask_length] = 0
        # print(file_path, image.shape)
        image = (image / 255.0)

        return image, labels

    def check_size(self):
        image, label = self.__getitem__(1)
        return image.shape

class OSAnpyDataset_spilt_4C(data.Dataset):
    def __init__(self, floader_path, train=False, fold=0):
        self.floader_path = floader_path + '/*/'
        self.labels = []
        self.datas = []
        self.train = train
        self.file_list = ''
        self.fold = fold
        snore_path = self.floader_path + 'snore'
        bs_path = self.floader_path + 'BS'
        noisepath = self.floader_path + 'noise'
        hypspath = self.floader_path + 'HS'
        snorefile_list = glob.glob(snore_path + "/*.png")
        bsfile_list = glob.glob(bs_path + "/*.png")
        hypsfile_list = glob.glob(hypspath + "/*.png")
        noisefile_list = glob.glob(noisepath + "/*.png")
        alllist = [snorefile_list, bsfile_list, hypsfile_list, noisefile_list]# hypsfile_list, noisefile_list]# , noisefile_list,hypsfile_list
        for listt in alllist:
            if listt == bsfile_list:  # load boomsnore file
                for i in range(len(listt)):
                    if self.train and int(listt[i].split('\\')[-3]) % 5 == fold :
                        self.datas.append(listt[i])
                        self.labels.append(0)
                    elif self.train == False and int(listt[i].split('\\')[-3]) % 5 != fold:
                        self.datas.append(listt[i])
                        self.labels.append(0)
            if listt == snorefile_list:  # load snore file
                for i in range(len(listt)):
                    if self.train and int(listt[i].split('\\')[-3]) % 5 == fold :
                        self.datas.append(listt[i])
                        self.labels.append(2)
                    elif self.train == False and int(listt[i].split('\\')[-3]) % 5 != fold:
                        self.datas.append(listt[i])
                        self.labels.append(2)
            elif listt == hypsfile_list:  # load hyp file
                for i in range(len(listt)):
                    if self.train and int(listt[i].split('\\')[-3]) % 5 == fold :
                        self.datas.append(listt[i])
                        self.labels.append(1)
                    elif self.train == False and int(listt[i].split('\\')[-3]) % 5 != fold:
                        self.datas.append(listt[i])
                        self.labels.append(1)
            else:
                for i in range(len(listt)):
                    if self.train and int(listt[i].split('\\')[-3]) % 5 == fold :
                        self.datas.append(listt[i])
                        self.labels.append(3)
                    elif self.train == False and int(listt[i].split('\\')[-3]) % 5 != fold:
                        self.datas.append(listt[i])
                        self.labels.append(3)
        # x_train, x_test, y_train, y_test = train_test_split(self.datas, self.labels, test_size=0.2, train_size=0.8, random_state=4)

    def __len__(self):

        return len(self.datas)

    def __getitem__(self, idx):

        file_path, labels = self.datas[idx], self.labels[idx]
        image = cv2.imread(file_path)
        if random.random() > 0.5 and self.train == True:
             mask_length = random.randint(6, 20)
             start_pos = random.randint(0, 64-mask_length-1)
             image[start_pos:start_pos+mask_length, :] = 0
        # if random.random() > 0.5 and self.train == True:
        #      mask_length = random.randint(16, 64)
        #      start_pos = random.randint(0, 512-mask_length-1)
        #      image[:, start_pos:start_pos+mask_length] = 0
        # print(file_path, image.shape)
        image = (image / 255.0)

        return image, labels

    def check_size(self):
        image, label = self.__getitem__(1)
        return image.shape