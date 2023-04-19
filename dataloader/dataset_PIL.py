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
    def __init__(self, floader_path, train_id=None, train=True):
        self.floader_path = floader_path + '/*/'
        self.labels = []
        self.datas = []
        self.train = train
        self.file_list = ''
        self.train_id = train_id
        self.transform_end = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
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
        label = self.labels[idx]
        image = Image.open(file_path)
        image = self.transform(image)
        return image, label


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