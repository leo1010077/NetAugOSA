import numpy as np
import random
import torch
import glob
import torch.utils.data as data
from torchvision import transforms
import math
from dataloader.utils import *

class OSAnpyDataset_augment_mix_3d(data.Dataset):
    def __init__(self, floader_path, train_id=None, train=True, aug_type='mix', doaugment_splice=True, doaugment_cover=True, doaugment_contrast=True, splice_chance=0.95):
        self.floader_path = floader_path + '/*/'
        self.labels = []
        self.datas = []
        self.train = train
        self.file_list = ''
        self.doaugment_splice = doaugment_splice
        self.doaugment_cover = doaugment_cover
        self.doaugment_contrast = doaugment_contrast
        self.splice_chance = splice_chance
        self.train_id = train_id
        self.aug_type = aug_type
        snore_path = self.floader_path + 'snore'
        bs_path = self.floader_path + 'BS'
        noisepath = self.floader_path + 'noise'
        hypspath = self.floader_path + 'HS'
        snorefile_list = glob.glob(snore_path + "/*.npy")
        bsfile_list = glob.glob(bs_path + "/*.npy")
        hypsfile_list = glob.glob(hypspath + "/*.npy")
        noisefile_list = glob.glob(noisepath + "/*.npy")
        self.class_data = [[], [], [], []]
        alllist = [snorefile_list, bsfile_list, hypsfile_list, noisefile_list]# hypsfile_list, noisefile_list]# , noisefile_list,hypsfile_list
        for listt in alllist:
            if listt == bsfile_list:  # load boomsnore file
                for i in range(len(listt)):
                    self.datas.append(listt[i])
                    self.labels.append(0)
                    self.class_data[0].append(listt[i])
            elif listt == snorefile_list:  # load snore file
                for i in range(len(listt)):
                    self.datas.append(listt[i])
                    self.labels.append(2)
                    self.class_data[2].append(listt[i])
            elif listt == hypsfile_list:  # load hyp file
                for i in range(len(listt)):
                    self.datas.append(listt[i])
                    self.labels.append(1)
                    self.class_data[1].append(listt[i])
            else:  # load noise file
                for i in range(len(listt)):
                    self.datas.append(listt[i])
                    self.labels.append(3)
                    self.class_data[3].append(listt[i])
    def __len__(self):

        return len(self.datas)

    def __getitem__(self, idx):

        file_path, labels = self.datas[idx], self.labels[idx]
        image = np.load(file_path)
        if self.train:
            if self.aug_type == 'mix':
                if labels != 3:
                    if self.doaugment_splice:
                        sub_file = self.listwithout_self_2(file_path, labels)
                        sub_mel = np.load(sub_file)
                        image = self.splice_2_mel(main_mel=image, sub_mel=sub_mel, chance=self.splice_chance)
                if self.doaugment_cover:
                    image = cover_image_random(image)
                if self.doaugment_contrast:
                    if random.random() > 1 - 0.85:
                        image = back(jitter(pre(image), random.randint(-30, 30), random.randint(-30, 30)))
            if self.aug_type == 'random_one':
                R = random.random()
                if labels != 3:
                    if R <= 0.8:
                        if self.doaugment_splice:
                            sub_file = self.listwithout_self_2(file_path, labels)
                            sub_mel = np.load(sub_file)
                            image = self.splice_2_mel(main_mel=image, sub_mel=sub_mel, chance=1)
                    elif R > 0.8 and R <= 0.85:
                        image = cover_image_random(image)
                    elif R > 0.85 and R <= 0.95:
                        image = back(jitter(pre(image), random.randint(-30, 30), random.randint(-30, 30)))
                    else:
                        image = image
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)
        image = np.concatenate([image, image, image], 0)
        return image, labels

    def check_size(self):
        image, label = self.__getitem__(1)
        return image.shape

    def listwithout_self(self, main_file, _class):
        class_data = self.class_data[_class]
        sub_file = random.choice(class_data)
        while sub_file == main_file:
            sub_file = random.choice(class_data)
        #print(main_file, sub_file)
        return sub_file

    def listwithout_self_2(self, main_file, _class):
        #print(self.train_id)
        index = len(self.train_id)
        #print(index)
        sub_file_index = random.randrange(0, index)
        label = self.labels[sub_file_index]
        sub_file = self.datas[sub_file_index]
        while label != _class or main_file == sub_file:
            sub_file_index = random.randrange(0, index)
            label = self.labels[sub_file_index]
            sub_file = self.datas[sub_file_index]
        #print(main_file, sub_file)
        return sub_file

    def splice_2_mel(self, main_mel, sub_mel, chance, chance2=0.5):
        # 偵測出檔案的主聲音事件
        main_index = main_event_detect(main_mel)
        sub_mel_index = main_event_detect(sub_mel)
        #print(main_index, sub_mel_index)
        if random.random() > (1 - chance):
            if random.random() > (1 - chance2):  # front
                OK_main_index_front = [0, main_index[0] - len(sub_mel_index) - 1]  # 找出可以用的index範圍
                #print(OK_main_index_front)
                if main_index[0] - len(sub_mel_index) - 1 > 0:
                    main_mel = self.front(main_mel, sub_mel, main_index, sub_mel_index, OK_main_index_front)
                else:
                    OK_main_index_back = [main_index[-1], main_mel.shape[1]]
                    main_mel = self.back(main_mel, sub_mel, main_index, sub_mel_index, OK_main_index_back)
            else:  # back
                # print('back')
                OK_main_index_back = [main_index[-1], main_mel.shape[1]]  # 找出可以用的index範圍
                main_mel = self.back(main_mel, sub_mel, main_index, sub_mel_index, OK_main_index_back)
                #print(OK_main_index_back)

        return main_mel

    def front(self, main_mel, sub_mel, main_mel_indxex, sub_mel_index, OK_main_index_front):
        start = random.randrange(OK_main_index_front[0], OK_main_index_front[1])  # 選一個地方開始
        main_mel[:, start:start + len(sub_mel_index) - 1] = sub_mel[:,sub_mel_index[0]: sub_mel_index[-1]]  # 將sub聲音事件放入main聲音事件
        return main_mel

    def back(self, main_mel, sub_mel, main_mel_indxex, sub_mel_index, OK_main_index_back):
        start = random.randrange(OK_main_index_back[0], OK_main_index_back[1])  # 找出哪裡開始
        if start + len(sub_mel_index) > main_mel.shape[1]:  # 如果後面的空間不構放入新的聲音事件就往前移
            not_enough = start + len(sub_mel_index) - main_mel.shape[1] - 1  # 算出不夠的量
            main_mel[:, :main_mel.shape[1] - not_enough] = main_mel[:, not_enough:]  # 往前移
            main_mel[:, start - not_enough:] = sub_mel[:, sub_mel_index[0]:sub_mel_index[-1]]  # 將submel放到後面
        else:
            main_mel[:, start:start + len(sub_mel_index) - 1] = sub_mel[:, sub_mel_index[0]: sub_mel_index[
                -1]]  # 將sub聲音事件放入main聲音事件
        return main_mel


class OSAnpyDataset_augment_3d_int(data.Dataset):
    def __init__(self, floader_path, augment, train=True, doaugment_jitter=None, doaugment_erase=False):
        self.floader_path = floader_path + '/*/'
        self.labels = []
        self.datas = []
        self.train = train
        self.doaugment_jitter = doaugment_jitter
        self.doaugment_erase = doaugment_erase
        self.file_list = ''
        self.augment = augment
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
    def __len__(self):

        return len(self.datas)

    def __getitem__(self, idx):

        file_path, labels = self.datas[idx], self.labels[idx]
        image = np.load(file_path)
        if self.train == True:
            if self.doaugment_erase == True:
                image = self.augment.cover_image(image)
            if self.doaugment_jitter != None:
                if random.random() > 1 - self.doaugment_jitter:
                    image = back(jitter(pre(image), random.randint(-30, 30), random.randint(-30, 30)))
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)
        image = (image + 80).astype(np.uint8)
        image = np.concatenate([image, image, image], 0)
        image = np.pad(image, ((0, 0), (0, 0), (0, 1)), "constant")
        #print(image.shape)
        return image, labels

    def check_size(self):
        image, label = self.__getitem__(1)
        return image.shape