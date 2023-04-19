import glob
import librosa
import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import augment
import random
from audiotransform.transform import SignalTransform


class OSASpectrogramDataset(data.Dataset):
    def __init__(
            self, floader_path, train=True,
            waveform_transforms=None, spectrogram_transforms=None, fr=4096, hop=690, n_fft=2048):

        self.floader_path = floader_path + '/*/'
        # self.img_size = settings['img_size']
        self.period = 4
        self.sr = 44100
        self.n_mels = 64
        self.hop = hop
        self.n_fft = n_fft

        self.labels = []
        self.datas = []
        self.fr = fr

        self.waveform_transforms = waveform_transforms
        self.spectrogram_transforms = spectrogram_transforms
        self.file_list = ''
        self.train = train
        snore_path = self.floader_path + 'snore'
        bs_path = self.floader_path + 'BS'
        noisepath = self.floader_path + 'noise'
        hypspath = self.floader_path + 'HS'
        snorefile_list = glob.glob(snore_path + "/*.wav")
        bsfile_list = glob.glob(bs_path + "/*.wav")
        hypsfile_list = glob.glob(hypspath + "/*.wav")
        noisefile_list = glob.glob(noisepath + "/*.wav")
        # self.file_list = snorefile_list + bsfile_list + noisefile_list # +hypsfile_list
        alllist = [bsfile_list, snorefile_list, noisefile_list, hypsfile_list]
        # alllist = [snorefile_list, bsfile_list, noisefile_list]
        for listt in alllist:

            if listt == bsfile_list:  # load boomsnore file
                for i in range(len(listt)):
                    self.datas.append(listt[i])
                    self.labels.append(0)
            elif listt == snorefile_list:  # load snore file
                for i in range(len(listt)):
                    self.datas.append(listt[i])
                    self.labels.append(1)
            # elif listt == hypsfile_list:  # load hyp file
            #     for i in range(len(listt)):
            #         self.datas.append(listt[i])
            #         self.labels.append(1)
            # else:  # load noise file
            #     for i in range(len(listt)):
            #         self.datas.append(listt[i])
            #         self.labels.append(3)

    def __len__(self):

        return len(self.datas)

    def __getitem__(self, idx):

        file_path, labels = self.datas[idx], self.labels[idx]

        y, fs = librosa.core.load(file_path, sr=None, mono=True, res_type='kaiser_fast')
        transforms = SignalTransform(sr=fs, n_mels=self.n_mels, fmax=self.fr, n_fft=self.n_fft, hop_length=self.hop)
        len_y = len(y)
        effective_length = fs * self.period
        np.random.seed(3)
        if len_y < effective_length:
            new_y = np.zeros(effective_length, dtype=y.dtype)
            start = np.random.randint(effective_length - len_y)
            new_y[start:start + len_y] = y
            y = new_y.astype(np.float32)
        elif len_y > effective_length:
            start = np.random.randint(len_y - effective_length)
            y = y[start:start + effective_length].astype(np.float32)
        else:
            y = y.astype(np.float32)

        # if self.waveform_transforms:
        #     for func_name in self.waveform_transforms:
        #         try:
        #             y = getattr(augment, func_name)(y)
        #         except AttributeError as e:
        #             print(f"{func_name} is None. {e}")
        # if self.train == True:
        #     if random.random() > 0.66:
        #         y = augment.add_white_noise(y)
        #     if random.random() > 0.66:
        #         y = augment.stretch(y)
        #     if random.random() > 0.66:
        #         y = augment.change_volume(y)
        # y = augment.add_white_noise(y)
        # y = augment.pitch_shift(y)

        image = transforms.melspectransform(y)
        if random.random() > 0.5 and self.train == True:
             mask_length = random.randint(6, 12)
             start_pos = random.randint(0, 64-mask_length-1)
             image[start_pos:start_pos+mask_length, :] = 0
        # z = np.zeros((64, 1))
        # image = np.concatenate((image, z), 1)

        # image = transforms.mfcctransform(y)
        # plt.imshow(image)
        # plt.show()
        # image = self.transforms.mfcctransform(y)
        # labels = np.zeros(len(self.labels), dtype="f")
        # labels[self.labels[label_code]] = 1

        # if self.spectrogram_transforms:
        #     image = self.spectrogram_transforms(image)

        return image, labels

    def check_size(self):
        image, label = self.__getitem__(1)
        return image.shape