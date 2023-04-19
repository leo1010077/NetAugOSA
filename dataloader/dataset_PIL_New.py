import random
import torch
import glob
import torch.utils.data as data
from torchvision import transforms
from dataloader.utils import *
from PIL import Image
from dataloader.augment import SpecAug, jitter
from sklearn.model_selection import KFold,train_test_split

class OSAnpyDataset_augment_mixup_mix_3d_png(data.Dataset):
    def __init__(self, floader_path, Transform_AUG, train_id=None, train=True, SPLICE=True, MIXUP=True, COVER=True):
        self.floader_path = floader_path + '/*/'
        self.labels = []
        self.datas = []
        self.train = train
        self.file_list = ''
        self.Transform_AUG = Transform_AUG
        self.SPLICE = SPLICE
        self.MIXUP=MIXUP
        self.COVER=COVER
        self.AUG_CHANCE = 1 / (int(SPLICE) + int(MIXUP) + int(COVER) + 1)
        self.train_id = train_id
        self.transform_start = transforms.Compose([transforms.ToTensor()])
        self.transform_end = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        snore_path = self.floader_path + 'snore'
        bs_path = self.floader_path + 'BS'
        noisepath = self.floader_path + 'noise'
        hypspath = self.floader_path + 'HS'
        snorefile_list = glob.glob(snore_path + "/*.png")
        bsfile_list = glob.glob(bs_path + "/*.png")
        hypsfile_list = glob.glob(hypspath + "/*.png")
        noisefile_list = glob.glob(noisepath + "/*.png")
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
        file_path = self.datas[idx]
        _c = self.labels[idx]
        image = Image.open(file_path)
        image = self.transform_start(image)
        label = np.eye(4)[_c]
        if self.train:
            chance = random.random()
            if chance <= self.AUG_CHANCE:
                image = image
            elif self.AUG_CHANCE <= chance <= self.AUG_CHANCE * 2:
                image = self.splice(image, file_path, _c)
            elif self.AUG_CHANCE * 2 <= chance <= self.AUG_CHANCE * 3:
                image, label = self.mixup(image, file_path, _c)
            else:
                image = self.Transform_AUG(image)
        image = self.transform_end(image)
        return image, label, _c

    def splice(self, image, file_path, _c):
        sub_file = self.listwithout_self_3(file_path, _c)
        sub_image = self.transform_start(Image.open(sub_file))
        image_re = self.splice_2_mel(main_mel=image, sub_mel=sub_image)
        return image_re

    def mixup(self, image, file_path, _c):
        sub_file, sub_label = self.list_woself_returnC(file_path, _c)
        sub_image = self.transform_start(Image.open(sub_file))
        image_re, label_re = self.mixup_label(image, sub_image, _c, sub_label)
        return image_re, label_re

    def list_woself_returnC(self, main_file, main_C):
        sub_file_index = random.choice(self.train_id)
        label = self.labels[sub_file_index]
        sub_file = self.datas[sub_file_index]
        while main_file == sub_file or main_C == label:
            sub_file_index = random.choice(self.train_id)
            label = self.labels[sub_file_index]
            sub_file = self.datas[sub_file_index]
        return sub_file, label

    def list_other_label_2(self, main_file, _class):
        sub_file_index = random.choice(self.train_id)
        label = self.labels[sub_file_index]
        sub_file = self.datas[sub_file_index]
        while label != _class or main_file == sub_file:
            sub_file_index = random.choice(self.train_id)
            label = self.labels[sub_file_index]
            sub_file = self.datas[sub_file_index]
        return sub_file

    def listwithout_self_3(self, main_file, _class): # 如果隨機找出的檔案class跟原本的不一樣或是重複就重找一個
        sub_file_index = random.choice(self.train_id)
        label = self.labels[sub_file_index]
        sub_file = self.datas[sub_file_index]
        while label != _class or main_file == sub_file:
            sub_file_index = random.choice(self.train_id)
            label = self.labels[sub_file_index]
            sub_file = self.datas[sub_file_index]
        return sub_file

    def splice_2_mel(self, main_mel, sub_mel, chance2=0.5):
        # 偵測出檔案的主聲音事件
        main_index = main_event_detect_tensor(main_mel)
        sub_mel_index = main_event_detect_tensor(sub_mel)
        if random.random() > (1 - chance2):  # front
            OK_main_index_front = [0, main_index[0] - len(sub_mel_index) - 1]  # 找出可以用的index範圍
            if main_index[0] - len(sub_mel_index) - 1 > 0:
                main_mel = self.front(main_mel, sub_mel, main_index, sub_mel_index, OK_main_index_front)
            else:
                OK_main_index_back = [main_index[-1], main_mel.shape[1]]
                main_mel = self.back(main_mel, sub_mel, main_index, sub_mel_index, OK_main_index_back)
        else:  # back
            OK_main_index_back = [main_index[-1], main_mel.shape[1]]  # 找出可以用的index範圍
            main_mel = self.back(main_mel, sub_mel, main_index, sub_mel_index, OK_main_index_back)

        return main_mel

    def front(self, main_mel, sub_mel, main_mel_indxex, sub_mel_index, OK_main_index_front):
        start = random.randrange(OK_main_index_front[0], OK_main_index_front[1])  # 選一個地方開始
        main_mel[:, :, start:start + len(sub_mel_index) - 1] = sub_mel[:, :, sub_mel_index[0]: sub_mel_index[-1]]  # 將sub聲音事件放入main聲音事件
        return main_mel

    def back(self, main_mel, sub_mel, main_mel_indxex, sub_mel_index, OK_main_index_back):
        start = random.randrange(OK_main_index_back[0], OK_main_index_back[1])  # 找出哪裡開始
        if start + len(sub_mel_index) > main_mel.shape[1]:  # 如果後面的空間不構放入新的聲音事件就往前移
            not_enough = start + len(sub_mel_index) - main_mel.shape[1] - 1  # 算出不夠的量
            main_mel[:, :main_mel.shape[1] - not_enough] = main_mel[:, not_enough:]  # 往前移
            main_mel[:, start - not_enough:] = sub_mel[:, sub_mel_index[0]:sub_mel_index[-1]]  # 將submel放到後面
        else:
            main_mel[:, :, start:start + len(sub_mel_index) - 1] = sub_mel[:, :, sub_mel_index[0]: sub_mel_index[
                -1]]  # 將sub聲音事件放入main聲音事件
        return main_mel


    def mixup_label(self, main_mel, sub_mel, main_c, sub_c):
        rate = random.randrange(750, 1000) / 1000
        out_mel = main_mel * rate + sub_mel * (1 - rate)
        if sub_c != 3:
            out_C = np.eye(4)[main_c] * rate + np.eye(4)[sub_c] * (1 - rate)
        else:
            out_C = np.eye(4)[main_c]
        #print(len(out_C))
        return out_mel, out_C

    def mixup_wolabel(self, main_mel, sub_mel):
        rate = random.randrange(750, 1000) / 1000
        out_mel = main_mel * rate + sub_mel * (1 - rate)
        return out_mel

if __name__ == '__main__':
    folders = 'J:/DATASET/OSA/DataSet/Dataset_stick_png/train/'
    transform = transforms.Compose([SpecAug(), jitter()])
    dataset = OSAnpyDataset_augment_mixup_mix_3d_png(folders, Transform_AUG=transform)
    dataset_size = len(dataset)
    print(dataset_size)
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        train_dataset = data.Subset(dataset, train_ids)
        dataset.train_id = train_ids
        loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        for idx, (data, label, _c) in enumerate(loader):
            print(label)