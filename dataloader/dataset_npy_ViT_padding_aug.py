import numpy as np
import random
import torch
import glob
import torch.utils.data as data
from torchvision import transforms
import math


def sort_index(list):
    start = 0
    list_out = []
    list_n = []
    for ind, n in enumerate(list):

        if start == 0:
            #print('start', n)
            list_n.append(n)
            start = 1
        else:
            if n - number == 1:
                #print('sort', n)
                list_n.append(n)
            else:
                #print('new')
                if list_n != []:
                    list_out.append(list_n)
                list_n = []
                list_n.append(n)
                start = 0
        number = n
    if list_n != []:
        list_out.append(list_n)

    return list_out

def index_2(list, long):
    r0 = 0
    r1 = list[0][0]
    out_list = []
    out_list.append([r0, r1])
    length = len(list)
    for ind, i in enumerate(list):
        out_list.append([i[0], i[-1]])
        if ind + 1 < length:
            out_list.append([i[-1], list[ind+1][0]])
    out_list.append([list[-1][-1], long])
    return out_list

class get_region_augment_npy:
    def __init__(self, min_time, cover_range=(0.1, 0.9), prob=50, prob2 = 50, cover_size=[0.3, 0.7], cover_size2 = [0.2, 0.3]):
        self.min_time = min_time
        self.range_min = cover_range[0]
        self.range_max = cover_range[1]
        self.size_min = cover_size[0]
        self.size_max = cover_size[1]
        self.size2_min = cover_size2[0]
        self.size2_max = cover_size2[1]
        self.prob = prob
        self.prob2 = prob2

    def get_regions(self, image, thread):
        index = image.shape[1]
        have = []
        for i in range(index):
            crop = image[:, i:i+4]
            mean_value = np.mean(crop)
            #cv2.imshow(str(mean_value), crop)
            #print(mean_value)
            if mean_value>= thread:
                have.append(i)

        return sort_index(have)

    def cover_image(self, array):
        p = np.mean(array)
        high = array.shape[0]
        long = array.shape[1]
        arr = array
        for region in self.get_regions(array, p + 3):
            #print(region)
            s, e, l = region[0], region[-1], (region[-1] - region[0])
            if l <= self.min_time * (long / 8):
                continue
            s2 = s + int(l * self.range_min)
            e2 = s + int(l * self.range_max)
            l2 = e2 - s2
            if random.randint(0, 100) >= self.prob:
                size = random.randint(int(l2 * self.size_min), int(l2 * self.size_max))
                start = random.randint(s2, e2)
                #print(start, size, '0')
                end = start + size
                if end >= e2:
                    end = e2
                arr[:, start:end] = p

        if random.randint(0, 100) > self.prob2:
            s, e = int(high * self.range_min), int(high * self.range_max)
            #print(s, e)
            size = random.randint(int(high * self.size2_min), int(high * self.size2_max))
            start = random.randint(s, e)
            end = start + size
            if end >= e:
                end = e
            #print(start, end)
            arr[start:end, :] = p

        return arr

    def cover_image_2(self, array):
        high = array.shape[0]
        long = array.shape[1]
        color = np.mean(array)
        regions = index_2(self.get_regions(array, color + 3), long)
        arr = array
        for region in regions:
            #print(region[0], region[-1])
            if random.randint(0, 100) > self.prob2:
                s, e = 0, high
                #print(s, e)
                size = random.randint(int(high * self.size2_min), int(high * self.size2_max))
                start = random.randint(s, e)
                end = start + size
                if end >= e:
                    end = e
                #print(start, end, region[0], region[1])
                #print('cover in {}:{}, {}, {} in {}'.format(start, end, region[0], region[-1], color) )
                arr[start:end, region[0]:region[-1]] = color
        return arr


    def cover_image_3(self, array):
        high = array.shape[0]
        long = array.shape[1]
        #regions = index_2(self.get_regions(image, np.mean(image) - 10))
        arr = array
        color = torch.mean(arr)
        TH = random.random()
        while TH <= 0.6:
            TH = random.random()
            size_h = random.randint(int(high * self.size2_min), int(high * self.size2_max))
            start_h = random.randint(0, high)
            end_h = start_h + size_h
            if end_h >= high:
                end_h = high
            size_l = random.randint(int(long * self.size_min), int(long * self.size_max))
            start_l = random.randint(0, long)
            end_l = start_l + size_l
            if end_l >= long:
                end_l = long
            arr[start_h:end_h, start_l:end_l] = color

        return arr


class OSAnpyDataset_augment(data.Dataset):
    def __init__(self, floader_path, augment, doaugment=False):
        self.floader_path = floader_path + '/*/'
        self.labels = []
        self.datas = []
        self.doaugment = doaugment
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
        if self.doaugment == True:
            image = self.augment.cover_image_2(image)
        image = np.expand_dims(image, axis=0)

        return image, labels

    def check_size(self):
        image, label = self.__getitem__(1)
        return image.shape


def pre(arr):
    arr += 80
    arr *= (255.0/arr.max())
    return arr

def back(arr):
    arr *= (80.0/arr.max())
    arr -= 80.0
    return arr

def jitter(image, brightness, contrast):
    B = brightness / 255.0
    C = contrast / 255.0
    k = math.tan((45 + 44 * C) / 180 * math.pi)
    image = (image - 127.5 * (1 - B)) * k + 127.5 * (1 + B)
    image = np.clip(image, 0, 255.0).astype(np.float32)
    return image

class OSAnpyDataset_augment_3d(data.Dataset):
    def __init__(self, floader_path, augment, ret_size, train=True, doaugment_jitter=None, doaugment_erase=False):
        self.floader_path = floader_path + '/*/'
        self.labels = []
        self.datas = []
        self.train = train
        self.ret_size = ret_size
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
        ret_arr = np.zeros((self.ret_size[0], self.ret_size[1]))
        image = np.load(file_path)
        if self.train == True:
            if self.doaugment_erase == True:
                image = self.augment.cover_image(image)
            if self.doaugment_jitter != None:
                if random.random() > 1 - self.doaugment_jitter:
                    image = back(jitter(pre(image), random.randint(-30, 30), random.randint(-30, 30)))
        ret_arr[0:image.shape[0], 0:image.shape[1]] = image
        image = ret_arr
        image = np.expand_dims(image, axis=0)
        image = np.concatenate([image, image, image], 0)
        return image, labels

    def check_size(self):
        image, label = self.__getitem__(1)
        return image.shape
