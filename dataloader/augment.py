import torch
from torchvision import transforms
from PIL import Image
import random
import math
import numpy
from dataloader.utils import *
import matplotlib.pyplot as plt
import glob

class SpecAug(torch.nn.Module):

    def __init__(self, type='random_aug', min_time=1, size=[0.1, 0.6], size2=[0.1, 0.6], range=[0.2, 0.8], prob=[0.5, 0.5]):
        super().__init__()
        self.type = type
        self.size_min, self.size_max = size[0], size[1]
        self.size2_min, self.size2_max = size2[0], size2[1]
        self.range_min, self.range_max = range[0], range[1]
        self.prob1, self.prob2 = prob[0], prob[1]
        self.min_time = min_time

    def forward(self, img):
        if self.type == 'random_aug':
            return self.random_aug(img)
        elif self.type == 'choose_region':
            return self.choose_region_aug(img)

    def random_aug(self, img):
        p = torch.mean(img)
        high = img.shape[1]
        long = img.shape[2]
        img_o = img
        if random.randint(0, 100) >= 1 - self.prob1:
            s, e = int(high * self.range_min), int(high * self.range_max)
            size = random.randint(int(long * self.size_min), int(long * self.size_max))
            start = random.randint(s, e)
            end = start + size
            if end >= e:
                end = e
            img_o[:, :, start:end] = p
        if random.randint(0, 100) >= 1 - self.prob2:
            s, e = int(high * self.range_min), int(high * self.range_max)
            size = random.randint(int(high * self.size2_min), int(high * self.size2_max))
            start = random.randint(s, e)
            end = start + size
            if end >= e:
                end = e
            img_o[:, start:end, :] = p
        return img_o

    def choose_region_aug(self, img):
        p = torch.mean(img)
        high = img.shape[1]
        long = img.shape[2]
        arr = img
        for region in self.get_regions(img, p):
            s, e, l = region[0], region[-1], (region[-1] - region[0])
            if l <= self.min_time * (long / 8):
                continue
            s2 = s + int(l * self.range_min)
            e2 = s + int(l * self.range_max)
            l2 = e2 - s2
            if random.randint(0, 100) >= 1 - self.prob1:
                size = random.randint(int(l2 * self.size_min), int(l2 * self.size_max))
                start = random.randint(s2, e2)
                end = start + size
                if end >= e2:
                    end = e2
                arr[:, :, start:end] = p

        if random.randint(0, 100) >= 1 - self.prob2:
            s, e = int(high * self.range_min), int(high * self.range_max)
            size = random.randint(int(high * self.size2_min), int(high * self.size2_max))
            start = random.randint(s, e)
            end = start + size
            if end >= e:
                end = e
            arr[:, start:end, :] = p

        return arr

    def get_regions(self, img, mean):
        index = img.shape[2]
        have = []
        for i in range(index):
            crop = img[:, :, i:i + 4]
            mean_value = torch.mean(crop)
            if mean_value >= mean:
                have.append(i)

        return self.sort_index(have)

    def sort_index(self, input_list):
        start = 0
        list_out = []
        list_n = []
        for ind, n in enumerate(input_list):
            if start == 0:
                list_n.append(n)
                temp = n
                start = 1
            else:
                if n - temp == 1:
                    list_n.append(n)
                    temp = n
                else:
                    list_out.append(list_n)
                    list_n = [n]
                    temp = n
        list_out.append(list_n)
        return list_out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"

class jitter(torch.nn.Module):

    def __init__(self, p=0.5, brightness=[-30, 30], contrast=[-30, 30]):
        super().__init__()
        self.p = p
        self.brightness = brightness
        self.contrast = contrast

    def forward(self, img):
        if random.random() >= self.p:
            B = random.randint(self.brightness[0], self.brightness[1]) / 255.0
            C = random.randint(self.contrast[0], self.contrast[1]) / 255.0
            k = math.tan((45 + 44 * C) / 180 * math.pi)
            return (img - 127.5 * (1 - B)) * k + 127.5 * (1 + B)
        else:
            return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"



class Resort(torch.nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, img):
        p = torch.mean(img)
        arr = img
        list = []
        start = None
        for idx, region in enumerate(self.get_regions(img, p)):
            #print(idx, region[0], region[-1])
            if idx == 0:
                if region[0] == 0:
                    start = region[-1]
                    list.append((0, region[-1]))
                    continue
                #print(0, region[0] -1, 'empyt')
                list.append((0, region[0]))
                start = region[-1]
                #print(region[0], region[-1], 'event')
                list.append((region[0], region[-1]))
            else:
                #print(start, region[0] -1, 'empyt')
                list.append((start, region[0]))
                start = region[-1]
                #print(region[0], region[-1], 'event')
                list.append((region[0], region[-1]))
        if start != 690 and start != None:
            #print(start, 690, 'empyt')
            list.append((start, 690))
        #print(list)
        random.shuffle(list)
        #print(list)
        start = 0
        arr = torch.zeros_like(img)
        for idx, region in enumerate(list):
            len = region[1] - region[0]
            arr[:, :, start:start + len] = img[:, :, region[0]:region[1]]
            #print(start, start + len, region[0], region[1])
            #print(start, start + len)
            start += len
        return arr


    def get_regions(self, img, mean):
        index = img.shape[2]
        have = []
        for i in range(index):
            crop = img[:, :, i:i + 4]
            mean_value = torch.mean(crop)
            if mean_value >= mean:
                have.append(i)

        return self.sort_index(have)

    def sort_index(self, input_list):
        start = 0
        list_out = []
        list_n = []
        for ind, n in enumerate(input_list):
            if start == 0:
                list_n.append(n)
                temp = n
                start = 1
            else:
                if n - temp == 1:
                    list_n.append(n)
                    temp = n
                else:
                    list_out.append(list_n)
                    list_n = [n]
                    temp = n
        list_out.append(list_n)
        return list_out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"




if __name__ == '__main__':
    print('MAIN')
    #image = Image.fromarray(np.ones((128, 512, 3), dtype=np.uint8))
    imagefolder = glob.glob('J:/DATASET/OSA/DataSet/Dataset_stick_png/train/*/*/*')
    # image = Image.open('J:/DATASET/OSA/DataSet/Dataset_stick_png/train/0/BS/0.png')
    # transform_AUG = transforms.Compose([transforms.ToTensor(), SpecAug(type='choose_region'), jitter(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # transforms_NOAUG1 = transforms.Compose([transforms.ToTensor()])
    # transform_resort = transforms.Compose([Resort()])
    # #transforms_NOAUG2 = transforms.Compose([jitter(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # # img = transform_AUG(image)
    # # img = transforms_NOAUG2(transforms_NOAUG1(image))
    # image = transforms_NOAUG1(image)
    # plt.imshow(image.permute(1, 2, 0))
    # plt.show()
    # img = transform_resort(image)
    # plt.imshow(img.permute(1, 2, 0))
    # plt.show()
    print(imagefolder)
    for i in range(100):
        image = random.choice(imagefolder)
        image = Image.open(image)
        transform_AUG = transforms.Compose([transforms.ToTensor(), SpecAug(type='choose_region'), jitter(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transforms_NOAUG1 = transforms.Compose([transforms.ToTensor()])
        transform_resort = transforms.Compose([Resort()])
        image = transforms_NOAUG1(image)
        plt.figure()
        plt.imshow(image.permute(1, 2, 0))
        plt.savefig('../showAUG/{}_ORI.png'.format(i))
        img = transform_resort(image)
        plt.figure()
        plt.imshow(img.permute(1, 2, 0))
        plt.savefig('../showAUG/{}_ReSort.png'.format(i))