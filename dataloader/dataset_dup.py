import glob

import numpy as np
import torch
import torch.utils.data as data


class OSAnpyDataset_spilt(data.Dataset):
    def __init__(self, floader_path, train=False, fold=0):
        self.floader_path = floader_path + '/*/'
        self.labels = []
        self.datas = []
        self.train = train
        self.file_list = ''
        self.fold = fold
        patient = glob.glob(self.floader_path)
        if self.train == False:
            nums = []
            for pid in range(len(patient)):

                snore_path = patient[pid] + 'snore'
                bs_path = patient[pid] + 'BS'
                hypspath = patient[pid] + 'HS'
                snorefile_list_len = len(glob.glob(snore_path + "/*.npy"))
                bsfile_list_len = len(glob.glob(bs_path + "/*.npy"))
                hypsfile_list = len(glob.glob(hypspath + "/*.npy"))
                nums.append(snorefile_list_len + bsfile_list_len + hypsfile_list)
            rate = np.around(np.max(nums) / nums).astype(np.uint8)
            # print(rate)

        for pid in range(len(patient)):

            snore_path = patient[pid] + 'snore'
            bs_path = patient[pid] + 'BS'
            noisepath = patient[pid] + 'noise'
            hypspath = patient[pid] + 'HS'
            snorefile_list = glob.glob(snore_path + "/*.npy")
            bsfile_list = glob.glob(bs_path + "/*.npy")
            hypsfile_list = glob.glob(hypspath + "/*.npy")
            noisefile_list = glob.glob(noisepath + "/*.npy")
            if self.train == False:
                temp_list = []
                for n in range(rate[pid]):
                    temp_list = temp_list + snorefile_list
                    # print('snore', n, rate[pid])
                snorefile_list = temp_list
                temp_list = []
                for n in range(rate[pid]):
                    temp_list = temp_list + bsfile_list
                    # print('bs', n, rate[pid])
                bsfile_list = temp_list
                temp_list = []
                for n in range(rate[pid]):
                    temp_list = temp_list + hypsfile_list
                    # print('hyp', n, rate[pid])
                hypsfile_list = temp_list
            alllist = [snorefile_list, bsfile_list, hypsfile_list, noisefile_list]
            for listt in alllist:
                if listt == bsfile_list:  # load boomsnore file
                    for i in range(len(listt)):
                        if self.train and i % 5 == fold:
                            self.datas.append(listt[i])
                            self.labels.append(0)
                        elif self.train == False and i % 5 != fold:
                            self.datas.append(listt[i])
                            self.labels.append(0)
                elif listt == snorefile_list:  # load snore file
                    for i in range(len(listt)):
                        if self.train and i % 5 == fold:
                            self.datas.append(listt[i])
                            self.labels.append(2)
                        elif self.train == False and i % 5 != fold:
                            self.datas.append(listt[i])
                            self.labels.append(2)
                elif listt == hypsfile_list:  # load hyp file
                    for i in range(len(listt)):
                        if self.train and i % 5 == fold:
                            self.datas.append(listt[i])
                            self.labels.append(1)
                        elif self.train == False and i % 5 != fold:
                            self.datas.append(listt[i])
                            self.labels.append(1)
                else:
                    for i in range(len(listt)):
                        if self.train and i % 5 == fold:
                            self.datas.append(listt[i])
                            self.labels.append(3)
                        elif self.train == False and i % 5 != fold:
                            self.datas.append(listt[i])
                            self.labels.append(3)

    def __len__(self):

        return len(self.datas)

    def __getitem__(self, idx):
        image = np.load(self.datas[idx])
        image = np.expand_dims(image, axis=0)
        image = np.concatenate([image, image, image], 0)
        image = torch.from_numpy(image).float()
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
    d = OSAnpyDataset_spilt('J:/DATASET/OSA/DataSet/Dataset_newmel_npy/train/', train=False)
    print(len(d))
