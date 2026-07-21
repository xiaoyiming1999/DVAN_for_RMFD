from nptdms import TdmsFile
import pandas as pd
import os

from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *


def AddWhiteGaussian(seq, SNR):
    Ps = np.sum(seq ** 2) / (seq.shape[0])
    Pn = Ps / (10 ** (SNR / 10))
    noise = np.random.normal(loc=0, scale=1, size=seq.shape)
    noise = noise * np.sqrt(Pn)
    signal_add_noise = seq + noise
    return signal_add_noise


class_name = {0: 'gearbox1', 1: 'gearbox2', 2: 'gearbox3', 3: 'gearbox4', 4: 'gearbox5',
              5: 'gearbox6', 6: 'gearbox7', 7: 'gearbox8', 8: 'gearbox9'}

work_condition = ['_16Hz', '_20Hz', '_24Hz', '_28Hz', '_32Hz', '_36Hz', '_40Hz']


def data_load(root, conditions, SNR, num_train_samples, num_val_samples, num_test_samples, sig_size, class_label, flag):
    data = []
    label = []

    for lab in class_label:
        name = class_name[lab]
        for condition in conditions:
            num_sample = 0
            path = os.path.join(root, name, name + work_condition[condition] + '.tdms')
            with TdmsFile.open(path) as tdms_file:
                group_name = []
                channel_name = []
                for group in tdms_file.groups():
                    group_name.append(group.name)
                for channel in group.channels():
                    channel_name.append(channel.name)
                channel = tdms_file[group_name[0]][channel_name[1]]
                all_channel_data = channel[:]
                data_temp = np.array(all_channel_data)
                data_temp = np.expand_dims(data_temp, axis=1)

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

def pseudo_data_load(root, conditions, SNR, num_train_samples, num_val_samples, sig_size, class_label):
    data = []
    label = []

    for lab in class_label:
        name = class_name[lab]
        for condition in conditions:
            num_sample = 0
            path = os.path.join(root, name, name + work_condition[condition] + '.tdms')
            with TdmsFile.open(path) as tdms_file:
                group_name = []
                channel_name = []
                for group in tdms_file.groups():
                    group_name.append(group.name)
                for channel in group.channels():
                    channel_name.append(channel.name)
                channel = tdms_file[group_name[0]][channel_name[1]]
                all_channel_data = channel[:]
                data_temp = np.array(all_channel_data)
                data_temp = np.expand_dims(data_temp, axis=1)

            start, end = sig_size * num_train_samples, sig_size + sig_size * num_train_samples
            while end <= data_temp.shape[0] and num_sample < num_val_samples:
                current_sample = data_temp[start:end]
                current_sample = AddWhiteGaussian(current_sample, SNR)
                data.append(current_sample)
                label.append(lab)
                start += sig_size
                end += sig_size
                num_sample += 1

    return [data, label]


class THU_data_split(object):
    def __init__(self, data_length,
                 num_train_samples, num_val_samples, num_test_samples,
                 train_noise_SNR, val_noise_SNR, test_noise_SNR, pseudo_noise_SNR,
                 train_classes, val_classes, test_classes,
                 train_op_conditions, val_op_conditions, test_op_conditions):

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
        self.pseudo_noise_SNR = pseudo_noise_SNR

        self.train_op_conditions = train_op_conditions
        self.val_op_conditions = val_op_conditions
        self.test_op_conditions = test_op_conditions

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
            root=r'D:\datasets\HanTe\Hante',
            conditions=self.train_op_conditions,
            SNR=self.train_noise_SNR,
            num_train_samples=self.num_train_samples,
            num_val_samples=self.num_val_samples,
            num_test_samples=self.num_test_samples,
            sig_size=self.data_length, class_label=self.train_classes, flag='train')

        list_val_data = data_load(
            root=r'D:\datasets\HanTe\Hante',
            conditions=self.val_op_conditions,
            SNR=self.val_noise_SNR,
            num_train_samples=self.num_train_samples,
            num_val_samples=self.num_val_samples,
            num_test_samples=self.num_test_samples,
            sig_size=self.data_length, class_label=self.val_classes, flag='val')

        list_test_data = data_load(
            root=r'D:\datasets\HanTe\Hante',
            conditions=self.test_op_conditions,
            SNR=self.test_noise_SNR,
            num_train_samples=self.num_train_samples,
            num_val_samples=self.num_val_samples,
            num_test_samples=self.num_test_samples,
            sig_size=self.data_length, class_label=self.test_classes, flag='test')

        list_pseudo_data = pseudo_data_load(
            root=r'D:\datasets\HanTe\Hante',
            conditions=self.val_op_conditions,
            SNR=self.pseudo_noise_SNR,
            num_train_samples=self.num_train_samples,
            num_val_samples=self.num_val_samples,
            sig_size=self.data_length, class_label=self.val_classes)

        train_data_pd = pd.DataFrame({"data": list_train_data[0], "label": list_train_data[1]})
        val_data_pd = pd.DataFrame({"data": list_val_data[0], "label": list_val_data[1]})
        test_data_pd = pd.DataFrame({"data": list_test_data[0], "label": list_test_data[1]})
        pseudo_data_pd = pd.DataFrame({"data": list_pseudo_data[0], "label": list_pseudo_data[1]})
        train = dataset(list_data=train_data_pd, transform=self.data_transforms['train'])
        val = dataset(list_data=val_data_pd, transform=self.data_transforms['val'])
        test = dataset(list_data=test_data_pd, transform=self.data_transforms['test'])
        pseudo = dataset(list_data=pseudo_data_pd, transform=self.data_transforms['val'])

        return train, val, test, pseudo