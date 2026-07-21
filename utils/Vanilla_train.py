#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import time
import warnings

import numpy as np
import torch
import logging
from torch import nn, optim
from models import CNN_7l, SE, CBAM, SA
from models import _ECELoss, _BS_loss
from datasets import THU_data_split, Loco_data_split
from datasets import Mean_std_process
from utils.prediction import entropy, calculate_threshold, cm_plot, calculate_FAR_MAR

class train_Vanilla_utils(object):
    def __init__(self, args, save_dir):
        self.save_dir = save_dir
        self.args = args

    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        :param args:
        :return:
        """
        args = self.args

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        if args.method == 'softmax-output':
            self.model = CNN_7l(num_cls=args.num_classes)
        elif args.method == 'SE':
            self.model = SE(num_cls=args.num_classes)
        elif args.method == 'CBAM':
            self.model = CBAM(num_cls=args.num_classes)
        elif args.method == 'SA':
            self.model = SA(num_cls=args.num_classes)
        else:
            raise Exception("method not implement")

        # Define the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)

        # Define the learning rate decay
        steps = [int(step) for step in args.steps.split(',')]
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)

        # Invert the model
        self.model.to(self.device)

        # define the loss
        self.criterion = nn.CrossEntropyLoss()
        self.ece_loss = _ECELoss(n_bins=10)
        self.BS_loss = _BS_loss(num_cls=args.num_classes)

    def train(self):

        args = self.args

        # load dataset
        self.datasets = {}
        if args.data_name == 'THU_gearbox':
            self.datasets['train'], self.datasets['val'], self.datasets['test'], self.datasets['pseudo'] = \
                THU_data_split(data_length=args.data_length,
                               num_train_samples=args.num_train_samples,
                               num_val_samples=args.num_val_samples,
                               num_test_samples=args.num_test_samples,
                               train_noise_SNR=args.train_noise_SNR,
                               val_noise_SNR=args.val_noise_SNR,
                               test_noise_SNR=args.test_noise_SNR,
                               pseudo_noise_SNR=args.pseudo_noise_SNR,
                               train_classes=args.train_classes,
                               val_classes=args.val_classes,
                               test_classes=args.test_classes,
                               train_op_conditions=args.train_op_conditions,
                               val_op_conditions=args.val_op_conditions,
                               test_op_conditions=args.test_op_conditions).data_split()

            self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                               shuffle=(True if x == 'train' else False),
                                                               num_workers=args.num_workers,
                                                               pin_memory=True,
                                                               drop_last=False) for x in
                                ['train', 'val', 'test', 'pseudo']}

        elif args.data_name == 'locomotive':
            self.datasets['train'], self.datasets['val'], self.datasets['test'] = \
                Loco_data_split(data_length=args.data_length,
                                   num_train_samples=args.num_train_samples,
                                   num_val_samples=args.num_val_samples,
                                   num_test_samples=args.num_test_samples,
                                   train_noise_SNR=args.train_noise_SNR,
                                   val_noise_SNR=args.val_noise_SNR,
                                   test_noise_SNR=args.test_noise_SNR,
                                   train_classes=args.train_classes,
                                   val_classes=args.val_classes,
                                   test_classes=args.test_classes).data_split()

            self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x == 'train' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=True,
                                                           drop_last=False) for x in ['train', 'val', 'test']}

        for epoch in range(0, args.epoch):
            logging.info('-' * 20 + 'Epoch {}/{}'.format(epoch, args.epoch - 1) + '-' * 20)
            #  learning rate
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))

            # Each epoch has a training and val phase
            for phase in ['train', 'val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0
                epoch_length = 0

                # Set model to train mode or test mode
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                all_outputs = []
                all_labels = []
                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # mean-std normalize
                    inputs = Mean_std_process(inputs)
                    inputs = inputs.to(torch.float32)

                    with torch.set_grad_enabled(phase == 'train'):
                        # forward
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)

                        all_outputs.append(outputs)
                        all_labels.append(labels)

                        pred = outputs.argmax(dim=1)
                        correct = torch.eq(pred, labels).float().sum().item()
                        loss_temp = loss.item() * labels.size(0)
                        epoch_loss += loss_temp
                        epoch_acc += correct
                        epoch_length += labels.size(0)

                        # Calculate the training information
                        if phase == 'train':
                            # backward
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                all_outputs = torch.cat(all_outputs)
                all_labels = torch.cat(all_labels)
                epoch_ece = self.ece_loss(all_outputs, all_labels)
                epoch_ece = epoch_ece.item()
                epoch_NLL = self.criterion(all_outputs, all_labels)
                epoch_NLL = epoch_NLL.item()
                epoch_BS = self.BS_loss(all_outputs, all_labels)
                epoch_BS = epoch_BS.item()

                epoch_loss = epoch_loss / epoch_length
                epoch_acc = epoch_acc / epoch_length

                logging.info(
                    'Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, {}-ECE: {:.4f}, {}-NLL: {:.4f}, {}-BS: {:.4f}, Cost {:.1f} ms'.format(
                        epoch, phase, epoch_loss, phase, epoch_acc, phase, epoch_ece, phase, epoch_NLL, phase, epoch_BS,
                        1000 * (time.time() - epoch_start)
                    ))

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # calculate threshold
        all_outputs = all_outputs.softmax(dim=-1)
        uncertainty = entropy(all_outputs, dim=-1)
        self.threshold = calculate_threshold(uncertainty)
        logging.info('Uncertainty_threshold: {:.4f}'.format(self.threshold, ))

    def test(self):
        args = self.args

        all_outputs = []
        all_labels = []
        acc = 0

        for batch_idx, (inputs, labels) in enumerate(self.dataloaders['test']):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # mean-std normalize
            inputs = Mean_std_process(inputs)
            inputs = inputs.to(torch.float32)

            with torch.no_grad():
                # forward
                outputs = self.model(inputs)
                outputs = outputs.softmax(dim=-1)

            all_labels.append(labels)
            all_outputs.append(outputs)

        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        all_uncertainty = entropy(all_outputs, dim=-1)

        pred = all_outputs.argmax(dim=-1)
        new_pred = torch.where(all_uncertainty > self.threshold, len(args.train_classes), pred)
        new_labels = torch.where(all_labels > (len(args.train_classes) - 1), len(args.train_classes), all_labels)

        correct = torch.eq(new_pred, new_labels).float().sum().item()
        acc += correct
        num_samples = all_outputs.shape[0]

        acc = acc / num_samples
        logging.info('Acc: {:.4f}'.format(acc, ))

        # plot confusion matrix
        true_label = new_labels.detach().cpu().numpy()
        predicted_label = new_pred.detach().cpu().numpy()
        cm_plot(true_label, predicted_label, len(args.train_classes))

        # FAR MAR
        true_label = (true_label == len(args.train_classes)).astype(int)
        predicted_label = (predicted_label == len(args.train_classes)).astype(int)
        FAR, MAR = calculate_FAR_MAR(predicted_label, true_label)
        logging.info('FAR: {:.4f}'.format(FAR, ))
        logging.info('MAR: {:.4f}'.format(MAR, ))





