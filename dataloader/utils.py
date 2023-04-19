import numpy as np
import random
import math
import cv2
import torch
def sort_index(list):
    start = 0
    list_out = []
    list_n = []
    for ind, n in enumerate(list):
        if start == 0:
            list_n.append(n)
            temp = n
            start = 1
        else:
            if n - temp == 1:
                list_n.append(n)
                temp = n
            else:
                # print('new', n)
                list_out.append(list_n)
                list_n = [n]
                temp = n
    list_out.append(list_n)
    return list_out

def main_event_detect(mel):
    p = np.mean(mel)
    index = mel.shape[1]
    have = []
    for i in range(index):
        crop = mel[:, i:i + 4]
        mean_value = np.mean(crop)
        if mean_value >= p:
            # print(i, mean_value, p+3)
            have.append(i)
    have = sort_index(have)
    max = 0
    for i, listt in enumerate(have):
        if len(listt) >= max:
            max = i
    # print(have)
    return have[max]

def main_event_detect_tensor(mel):
    p = torch.mean(mel)
    index = mel.shape[1]
    have = []
    for i in range(index):
        crop = mel[:, i:i + 4]
        mean_value = torch.mean(crop)
        if mean_value >= p:
            # print(i, mean_value, p+3)
            have.append(i)
    have = sort_index(have)
    max = 0
    for i, listt in enumerate(have):
        if len(listt) >= max:
            max = i
    # print(have)
    return have[max]

def main_event_detect_tensor_True(mel):
    p = torch.mean(mel)
    index = mel.shape[2]
    have = []
    for i in range(index):
        crop = mel[:, :, i:i + 4]
        mean_value = torch.mean(crop)
        if mean_value >= p:
            # print(i, mean_value, p+3)
            have.append(i)
    have = sort_index(have)
    max = 0
    for i, listt in enumerate(have):
        if len(listt) >= max:
            max = i
    # print(have)
    return have[max]

def get_regions(mel, p):
    index = mel.shape[1]
    have = []
    for i in range(index):
        crop = mel[:, i:i + 4]
        mean_value = np.mean(crop)
        if mean_value >= p:
            have.append(i)

    return sort_index(have)

def get_regions_tensor(mel, p):
    index = mel.shape[2]
    have = []
    for i in range(index):
        crop = mel[:, :, i:i + 4]
        mean_value = torch.mean(crop)
        if mean_value >= p:
            have.append(i)

    return sort_index(have)

def cover_image(array, min_time=1,size=[0.1, 0.6], size2=[0.1, 0.6], range=[0.2, 0.8], prob=[0.85, 0.85]):
    p = np.mean(array)
    size_min, size_max = size[0], size[1]
    size2_min, size2_max = size2[0], size2[1]
    range_min, range_max = range[0], range[1]
    prob1, prob2 = prob[0], prob[1]
    high = array.shape[0]
    long = array.shape[1]
    arr = array
    for region in get_regions(array, p):
        s, e, l = region[0], region[-1], (region[-1] - region[0])
        if l <= min_time * (long / 8):
            continue
        s2 = s + int(l * range_min)
        e2 = s + int(l * range_max)
        l2 = e2 - s2
        if random.randint(0, 100) >= 1 - prob1:
            size = random.randint(int(l2 * size_min), int(l2 * size_max))
            start = random.randint(s2, e2)
            end = start + size
            if end >= e2:
                end = e2
            arr[:, start:end] = p

    if random.randint(0, 100) >= 1 - prob2:
        s, e = int(high * range_min), int(high * range_max)
        size = random.randint(int(high * size2_min), int(high * size2_max))
        start = random.randint(s, e)
        end = start + size
        if end >= e:
            end = e
        arr[start:end, :] = p

    return arr

def cover_image_random(array,size=[0.1, 0.6], size2=[0.1, 0.6], range=[0.2, 0.8], prob=[0.75, 0.75]):
    p = np.mean(array)
    size_min, size_max = size[0], size[1]
    size2_min, size2_max = size2[0], size2[1]
    range_min, range_max = range[0], range[1]
    prob1, prob2 = prob[0], prob[1]
    high = array.shape[0]
    long = array.shape[1]
    arr = array
    if random.randint(0, 100) >= 1 - prob1:
        s, e = int(high * range_min), int(high * range_max)
        size = random.randint(int(long * size_min), int(long * size_max))
        start = random.randint(s, e)
        end = start + size
        if end >= e:
            end = e
        arr[start:end, :] = p
    if random.randint(0, 100) >= 1 - prob2:
        s, e = int(high * range_min), int(high * range_max)
        size = random.randint(int(high * size2_min), int(high * size2_max))
        start = random.randint(s, e)
        end = start + size
        if end >= e:
            end = e
        arr[start:end, :] = p

    return arr


def cover_image_random2(array, size=[0.1, 0.6], size2=[0.1, 0.6], range=[0.2, 0.8], prob=[0.65, 0.65]):
    p = np.mean(array)
    size_min, size_max = size[0], size[1]
    size2_min, size2_max = size2[0], size2[1]
    range_min, range_max = range[0], range[1]
    prob1, prob2 = prob[0], prob[1]
    high = array.shape[0]
    long = array.shape[1]
    arr = array
    while random.randint(0, 100) >= 1 - prob1:
        s, e = int(high * range_min), int(high * range_max)
        size = random.randint(int(long * size_min), int(long * size_max))
        start = random.randint(s, e)
        end = start + size
        if end >= e:
            end = e
        arr[start:end, :] = p
    while random.randint(0, 100) >= 1 - prob2:
        s, e = int(high * range_min), int(high * range_max)
        size = random.randint(int(high * size2_min), int(high * size2_max))
        start = random.randint(s, e)
        end = start + size
        if end >= e:
            end = e
        arr[start:end, :] = p

    return arr

def jitter(image, brightness, contrast):
    B = brightness / 255.0
    C = contrast / 255.0
    k = math.tan((45 + 44 * C) / 180 * math.pi)
    image = (image - 127.5 * (1 - B)) * k + 127.5 * (1 + B)
    image = np.clip(image, 0, 255.0).astype(np.float32)
    return image

def pre(arr):
    arr += 80
    arr *= (255.0/arr.max())
    return arr

def back(arr):
    arr *= (80.0/arr.max())
    arr -= 80.0
    return arr

def main_event_detect_png(mel):
    p = np.mean(mel)
    index = mel.shape[1]
    have = []
    for i in range(index):
        crop = mel[:, i:i + 4, :]
        mean_value = np.mean(crop)
        if mean_value >= p:
            # print(i, mean_value, p+3)
            have.append(i)
    have = sort_index(have)
    max = 0
    for i, listt in enumerate(have):
        if len(listt) >= max:
            max = i
    # print(have)
    return have[max]

def get_regions_png(mel, p):
    index = mel.shape[1]
    have = []
    for i in range(index):
        crop = mel[:, i:i + 4, :]
        mean_value = np.mean(crop)
        if mean_value >= p:
            have.append(i)

    return sort_index(have)

def cover_image_random_png(array,size=[0.1, 0.6], size2=[0.1, 0.6], range=[0.2, 0.8], prob=[0.75, 0.75]):
    p = np.mean(array)
    size_min, size_max = size[0], size[1]
    size2_min, size2_max = size2[0], size2[1]
    range_min, range_max = range[0], range[1]
    prob1, prob2 = prob[0], prob[1]
    high = array.shape[0]
    long = array.shape[1]
    arr = array
    if random.randint(0, 100) >= 1 - prob1:
        s, e = int(high * range_min), int(high * range_max)
        size = random.randint(int(long * size_min), int(long * size_max))
        start = random.randint(s, e)
        end = start + size
        if end >= e:
            end = e
        arr[:, start:end, :] = p
    if random.randint(0, 100) >= 1 - prob2:
        s, e = int(high * range_min), int(high * range_max)
        size = random.randint(int(high * size2_min), int(high * size2_max))
        start = random.randint(s, e)
        end = start + size
        if end >= e:
            end = e
        arr[start:end, :, :] = p

    return arr


if __name__ == '__main__':
    #list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 19, 20, 21, 22, 23, 24, 27, 28, 29, 30, 31, 32]
    ## print(sort_index(list))
    #print(get_regions())
    # test_mel = np.load('J:/DATASET/OSA/DataSet/Dataset_newmel_npy/train/廖俊傑/BS/廖俊傑_25_1.npy')
    # print(cover_image(test_mel).shape)
    test_img = cv2.imread('J:/DATASET/OSA/DataSet/Dataset_stick_png/train/0/BS/0.png')
    print(test_img.shape)
    print(main_event_detect_png(test_img))
