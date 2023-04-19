import glob
import random

import numpy as np
import torch
import torch.utils.data as data
import math


def jitter(image, brightness, contrast):
    B = brightness / 255.0
    C = contrast / 255.0
    k = math.tan((45 + 44 * C) / 180 * math.pi)
    image = (image - 127.5 * (1 - B)) * k + 127.5 * (1 + B)
    image = np.clip(image, 0, 255.0).astype(np.float32)
    return image

class OSAnpyDataset_spilt(data.Dataset):
    def __init__(self, floader_path, val=False, fold=0):
        self.floader_path = floader_path + '/*/'
        self.labels = []
        self.datas = []
        self.pats = []
        self.val = val
        self.file_list = ''
        self.fold = fold
        patient = glob.glob(self.floader_path)
        if self.val == False:
            nums = []
            for pid in range(len(patient)):
                # noisepath = patient[pid] + 'noise'
                snore_path = patient[pid] + 'snore'
                bs_path = patient[pid] + 'BS'
                hypspath = patient[pid] + 'HS'
                # noisefile_list = glob.glob(noisepath + "/*.npy")
                snorefile_list_len = len(glob.glob(snore_path + "/*.npy"))
                bsfile_list_len = len(glob.glob(bs_path + "/*.npy"))
                hypsfile_list = len(glob.glob(hypspath + "/*.npy"),)
                nums.append(snorefile_list_len + bsfile_list_len + hypsfile_list)
            #rate = np.around(np.max(nums) / nums).astype(np.uint8)
            #print(nums)

        for pid in range(len(patient)):

            snore_path = patient[pid] + 'snore'
            bs_path = patient[pid] + 'BS'
            noisepath = patient[pid] + 'noise'
            hypspath = patient[pid] + 'HS'
            snorefile_list = glob.glob(snore_path + "/*.npy")
            bsfile_list = glob.glob(bs_path + "/*.npy")
            hypsfile_list = glob.glob(hypspath + "/*.npy")
            noisefile_list = glob.glob(noisepath + "/*.npy")
            alllist = [snorefile_list, bsfile_list, hypsfile_list, noisefile_list]
            for listt in alllist:
                if listt == bsfile_list:  # load boomsnore file
                    for i in range(len(listt)):
                        if self.val and i % 5 == fold:
                            self.datas.append(listt[i])
                            self.labels.append(0)

                        elif self.val == False and i % 5 != fold:
                            self.datas.append(listt[i])
                            self.labels.append(0)
                            self.pats.append(pid)
                elif listt == snorefile_list:  # load snore file
                    for i in range(len(listt)):
                        if self.val and i % 5 == fold:
                            self.datas.append(listt[i])
                            self.labels.append(2)
                        elif self.val == False and i % 5 != fold:
                            self.datas.append(listt[i])
                            self.labels.append(2)
                            self.pats.append(pid)
                elif listt == hypsfile_list:  # load hyp file
                    for i in range(len(listt)):
                        if self.val and i % 5 == fold:
                            self.datas.append(listt[i])
                            self.labels.append(1)
                        elif self.val == False and i % 5 != fold:
                            self.datas.append(listt[i])
                            self.labels.append(1)
                            self.pats.append(pid)
                else:
                    for i in range(len(listt)):
                        if self.val and i % 5 == fold:
                            self.datas.append(listt[i])
                            self.labels.append(3)
                        elif self.val == False and i % 5 != fold:
                            self.datas.append(listt[i])
                            self.labels.append(3)
                            self.pats.append(pid)

    def __len__(self):

        return len(self.datas)

    def __getitem__(self, idx):
        image = np.load(self.datas[idx])
        image = np.expand_dims(image, axis=0)
        image = np.concatenate([image, image, image], 0)
        image = torch.from_numpy(image).float()
        # if self.val == False:
        #     if random.Random() > 0.5:
        #         rb = random.randint(-30,30)
        #         rc = random.randint(-30,30)
        #         image = jitter(image, rb, rc)
        # if random.random() > 0.5 and self.train == True:
        #      mask_length = random.randint(6, 20)
        #      start_pos = random.randint(0, 64-mask_length-1)
        #      image[start_pos:start_pos+mask_length, :] = 0
        # if random.random() > 0.5 and self.train == True:
        #      mask_length = random.randint(16, 64)
        #      start_pos = random.randint(0, 512-mask_length-1)
        #      image[:, start_pos:start_pos+mask_length] = 0
        # print(file_path, image.shape)

        return image, self.labels[idx]

    def check_size(self):
        image, label = self.__getitem__(1)
        return image.shape



if __name__ == '__main__':
    d = OSAnpyDataset_spilt('J:/DATASET/OSA/DataSet/Dataset_newmel_npy/train/', val=False)
    print(len(d))
