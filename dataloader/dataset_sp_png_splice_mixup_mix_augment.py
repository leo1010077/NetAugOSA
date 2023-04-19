import glob
import cv2
import numpy as np
import torch.utils.data as data
import random
from sklearn.model_selection import train_test_split
from dataloader.utils import *


class OSAnpyDataset_augment_mixup_mix_3d_png(data.Dataset):
    def __init__(self, floader_path, fold=0, not_train=True, aug_type='mix', doaugment_splice=True, doaugment_mixup=True, doaugment_cover=True, doaugment_contrast=True, splice_chance=0.95):
        self.floader_path = floader_path + '/*/'
        self.labels = []
        self.datas = []
        self.train = not_train
        self.file_list = ''
        self.fold = fold
        self.doaugment_splice = doaugment_splice
        self.doaugment_mixup = doaugment_mixup
        self.doaugment_cover = doaugment_cover
        self.doaugment_contrast = doaugment_contrast
        self.splice_chance = splice_chance
        self.aug_type = aug_type
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
        if self.train == False:
            if self.aug_type == 'mix':
                if labels != 3:
                    if self.doaugment_splice:
                        sub_file = self.listwithout_self_2(file_path, labels)
                        sub_mel = cv2.imread(sub_file)
                        image = self.splice_2_mel(main_mel=image, sub_mel=sub_mel, chance=self.splice_chance)
                    if self.doaugment_mixup:
                        sub_file = self.list_other_label(file_path, labels)
                        sub_mel = cv2.imread(sub_file)
                        image = self.mixup_wolabel(main_mel=image, sub_mel=sub_mel)
                if self.doaugment_cover:
                    image = cover_image_random_png(image)
                if self.doaugment_contrast:
                    if random.random() > 1 - 0.85:
                        image = jitter(image, random.randint(-30, 30), random.randint(-30, 30))
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
        image = np.transpose(image, (2, 0, 1))
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

    def list_other_label(self, main_file, _class):
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
        main_mel[:, start:start + len(sub_mel_index) - 1, :] = sub_mel[:,sub_mel_index[0]: sub_mel_index[-1], :]  # 將sub聲音事件放入main聲音事件
        return main_mel

    def back(self, main_mel, sub_mel, main_mel_indxex, sub_mel_index, OK_main_index_back):
        start = random.randrange(OK_main_index_back[0], OK_main_index_back[1])  # 找出哪裡開始
        if start + len(sub_mel_index) > main_mel.shape[1]:  # 如果後面的空間不構放入新的聲音事件就往前移
            not_enough = start + len(sub_mel_index) - main_mel.shape[1] - 1  # 算出不夠的量
            main_mel[:, :main_mel.shape[1] - not_enough] = main_mel[:, not_enough:]  # 往前移
            main_mel[:, start - not_enough:] = sub_mel[:, sub_mel_index[0]:sub_mel_index[-1]]  # 將submel放到後面
        else:
            main_mel[:, start:start + len(sub_mel_index) - 1, :] = sub_mel[:, sub_mel_index[0]: sub_mel_index[
                -1], :]  # 將sub聲音事件放入main聲音事件
        return main_mel


    def mixup_label(self, main_mel, sub_mel, main_c, sub_c):
        rate = random.randrange(750, 1000) / 1000
        out_mel = main_mel * rate + sub_mel * (1 - rate)
        out_C = main_c * rate + sub_c * (1 - rate)
        return out_mel, out_C

    def mixup_wolabel(self, main_mel, sub_mel):
        rate = random.randrange(750, 1000) / 1000
        out_mel = main_mel * rate + sub_mel * (1 - rate)
        return out_mel

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
        if self.train:
            if self.aug_type == 'mix':
                if labels != 3:
                    if self.doaugment_splice:
                        sub_file = self.listwithout_self(file_path, labels)
                        sub_mel = cv2.imread(sub_file)
                        image = self.splice_2_mel(main_mel=image, sub_mel=sub_mel, chance=self.splice_chance)
                    if self.doaugment_mixup:
                        sub_file = self.listwithout_self(file_path, labels)
                        sub_mel = cv2.imread(sub_file)
                        image = self.mixup_wolabel(main_mel=image, sub_mel=sub_mel)
                if self.doaugment_cover:
                    image = cover_image_random_png(image)
                if self.doaugment_contrast:
                    if random.random() > 1 - 0.85:
                        image = jitter(image, random.randint(-30, 30), random.randint(-30, 30))
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
        image = np.transpose(image, (2, 0, 1))
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

    def list_other_label(self, main_file, _class):
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
        main_mel[:, start:start + len(sub_mel_index) - 1, :] = sub_mel[:,sub_mel_index[0]: sub_mel_index[-1], :]  # 將sub聲音事件放入main聲音事件
        return main_mel

    def back(self, main_mel, sub_mel, main_mel_indxex, sub_mel_index, OK_main_index_back):
        start = random.randrange(OK_main_index_back[0], OK_main_index_back[1])  # 找出哪裡開始
        if start + len(sub_mel_index) > main_mel.shape[1]:  # 如果後面的空間不構放入新的聲音事件就往前移
            not_enough = start + len(sub_mel_index) - main_mel.shape[1] - 1  # 算出不夠的量
            main_mel[:, :main_mel.shape[1] - not_enough] = main_mel[:, not_enough:]  # 往前移
            main_mel[:, start - not_enough:] = sub_mel[:, sub_mel_index[0]:sub_mel_index[-1]]  # 將submel放到後面
        else:
            main_mel[:, start:start + len(sub_mel_index) - 1, :] = sub_mel[:, sub_mel_index[0]: sub_mel_index[
                -1], :]  # 將sub聲音事件放入main聲音事件
        return main_mel


    def mixup_label(self, main_mel, sub_mel, main_c, sub_c):
        rate = random.randrange(750, 1000) / 1000
        out_mel = main_mel * rate + sub_mel * (1 - rate)
        out_C = main_c * rate + sub_c * (1 - rate)
        return out_mel, out_C

    def mixup_wolabel(self, main_mel, sub_mel):
        rate = random.randrange(750, 1000) / 1000
        out_mel = main_mel * rate + sub_mel * (1 - rate)
        return out_mel