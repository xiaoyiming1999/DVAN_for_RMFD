
import pandas as pd
import scipy
from scipy.io import loadmat
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *


def AddWhiteGaussian(seq, SNR):
    Ps = np.sum(seq ** 2) / (seq.shape[0])
    Pn = Ps / (10 ** (SNR / 10))
    noise = np.random.normal(loc=0, scale=1, size=seq.shape)
    noise = noise * np.sqrt(Pn)
    signal_add_noise = seq + noise
    return signal_add_noise

class_name = {0: 'C1', 1: 'C2', 2: 'C3', 3: 'C4', 4: 'C5', 5: 'C6', 6: 'C7', 7: 'C8'}

def data_load(root, SNR, num_train_samples, num_val_samples, num_test_samples, sig_size, class_label, flag):

    data_dictionary = scipy.io.loadmat(root)
    data = []
    label = []
    for lab in class_label:
        name = class_name[lab]
        num_sample = 0
        data_temp = np.array(data_dictionary[name][0:6250000, ]).reshape(-1, 1)

        # train
        if flag == 'train':
            start, end = 0, sig_size
            while end <= data_temp.shape[0] and num_sample < num_train_samples:
                current_sample = data_temp[start:end]
                current_sample = AddWhiteGaussian(current_sample, SNR)
                data.append(current_sample)
                label.append(lab)
                start += sig_size
                end += sig_size
                num_sample += 1
        # val
        elif flag == 'val':
            start, end = sig_size * num_train_samples, sig_size + sig_size * num_train_samples
            while end <= data_temp.shape[0] and num_sample < num_val_samples:
                current_sample = data_temp[start:end]
                current_sample = AddWhiteGaussian(current_sample, SNR)
                data.append(current_sample)
                label.append(lab)
                start += sig_size
                end += sig_size
                num_sample += 1

        elif flag == 'test':
            start, end = sig_size * (num_train_samples + num_val_samples), \
                         sig_size + sig_size * (num_train_samples + num_val_samples)
            while end <= data_temp.shape[0] and num_sample < num_test_samples:
                current_sample = data_temp[start:end]
                current_sample = AddWhiteGaussian(current_sample, SNR)
                data.append(current_sample)
                label.append(lab)
                start += sig_size
                end += sig_size
                num_sample += 1

    return [data, label]

class Loco_data_split(object):
    def __init__(self, data_length,
                 num_train_samples, num_val_samples, num_test_samples,
                 train_noise_SNR, val_noise_SNR, test_noise_SNR,
                 train_classes, val_classes, test_classes):

        self.data_length = data_length

        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples

        self.train_classes = train_classes
        self.val_classes = val_classes
        self.test_classes = test_classes

        self.train_noise_SNR = train_noise_SNR
        self.val_noise_SNR = val_noise_SNR
        self.test_noise_SNR = test_noise_SNR

        self.data_transforms = {
            'train': Compose([
                Reshape(),
                # AddGaussian(),
                # RandomAddGaussian(),
                RandomScale(),
                RandomStretch(),
                RandomCrop(),
                Retype(),
                # Scale(1)
            ]),
            'val': Compose([
                Reshape(),
                Retype(),
                # Scale(1)
            ]),
            'test': Compose([
                Reshape(),
                Retype(),
                # Scale(1)
            ])
        }

    def data_split(self):
        list_train_data = data_load(
            root=r'D:\datasets',
            SNR=self.train_noise_SNR,
            num_train_samples=self.num_train_samples,
            num_val_samples=self.num_val_samples,
            num_test_samples=self.num_test_samples,
            sig_size=self.data_length, class_label=self.train_classes, flag='train')

        list_val_data = data_load(
            root=r'D:\datasets',
            SNR=self.val_noise_SNR,
            num_train_samples=self.num_train_samples,
            num_val_samples=self.num_val_samples,
            num_test_samples=self.num_test_samples,
            sig_size=self.data_length, class_label=self.val_classes, flag='val')

        list_test_data = data_load(
            root=r'D:\datasets',
            SNR=self.test_noise_SNR,
            num_train_samples=self.num_train_samples,
            num_val_samples=self.num_val_samples,
            num_test_samples=self.num_test_samples,
            sig_size=self.data_length, class_label=self.test_classes, flag='test')

        train_data_pd = pd.DataFrame({"data": list_train_data[0], "label": list_train_data[1]})
        val_data_pd = pd.DataFrame({"data": list_val_data[0], "label": list_val_data[1]})
        test_data_pd = pd.DataFrame({"data": list_test_data[0], "label": list_test_data[1]})
        train = dataset(list_data=train_data_pd, transform=self.data_transforms['train'])
        val = dataset(list_data=val_data_pd, transform=self.data_transforms['val'])
        test = dataset(list_data=test_data_pd, transform=self.data_transforms['test'])

        return train, val, test

