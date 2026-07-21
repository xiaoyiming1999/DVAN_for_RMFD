#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import time
import warnings

import numpy as np
import torch
import logging
from torch import nn, optim
from models import VSE, VCBAM, VSA
from models import _ECELoss, _BS_loss
from datasets import THU_data_split, Loco_data_split
from datasets import Mean_std_process
from variational_layer.KL_loss import get_beta
from utils.prediction import vattn_bayes_predict, calculate_threshold, vattn_bayes_test, calculate_mi, \
    cm_plot, calculate_FAR_MAR


class train_DVAN_utils(object):
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

        # define model
        self.priors = {'prior_mu': args.prior_mu, 'prior_std': args.prior_std}

        if args.method == 'VSE':
            self.model = VSE(num_cls=args.num_classes, priors=self.priors, device=self.device)
        elif args.method == 'VCBAM':
            self.model = VCBAM(num_cls=args.num_classes, priors=self.priors, device=self.device)
        elif args.method == 'VSA':
            self.model = VSA(num_cls=args.num_classes, priors=self.priors, device=self.device)
        else:
            raise Exception("method not implement")

        # define optimizer
        self.optimizer = torch.optim.Adam([
            {'params': (p for name, p in self.model.named_parameters() if 'V' not in name),
             'weight_decay': args.weight_decay},
            {'params': (p for name, p in self.model.named_parameters() if 'V' in name), 'weight_decay': 0}
        ], lr=args.lr)

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
                                                               drop_last=False) for x in ['train', 'val', 'test', ]}

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

                epoch_loss_likelihood = 0.0
                epoch_loss_prior = 0.0
                epoch_loss_MCSL = 0.0

                # Set model to train mode or test mode
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                all_outputs = []
                all_labels = []
                all_probs = []

                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # mean-std normalize
                    inputs = Mean_std_process(inputs)
                    inputs = inputs.to(torch.float32)

                    with torch.set_grad_enabled(phase == 'train'):
                        # forward
                        beta = get_beta(batch_idx=batch_idx, m=len(self.dataloaders[phase]),
                                        beta_type=args.beta_type, epoch=epoch, num_epochs=args.epoch)
                        outputs, loss_likelihood, loss_prior, loss_MCSL, probs = \
                                                            vattn_bayes_predict(inputs, labels, beta, self.criterion,
                                                            self.model, args.num_MC_sampling,
                                                            self.device, args.MC_shaping_mu, args.MC_shaping_std)

                        all_outputs.append(outputs)
                        all_labels.append(labels)
                        all_probs.append(probs)

                        if args.is_MC_shaping:
                            loss = loss_likelihood + loss_prior + loss_MCSL
                        else:
                            loss = loss_likelihood + loss_prior

                        pred = outputs.argmax(dim=1)
                        correct = torch.eq(pred, labels).float().sum().item()
                        loss_temp = loss.item() * labels.size(0)
                        epoch_loss += loss_temp

                        loss_likelihood_temp = loss_likelihood.item() * labels.size(0)
                        epoch_loss_likelihood += loss_likelihood_temp
                        loss_prior_temp = loss_prior.item() * labels.size(0)
                        epoch_loss_prior += loss_prior_temp
                        loss_MCSL_temp = loss_MCSL.item() * labels.size(0)
                        epoch_loss_MCSL += loss_MCSL_temp

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
                all_probs = torch.cat(all_probs, dim=1)
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
        self.val_uncertainty = calculate_mi(all_probs)
        self.threshold = calculate_threshold(self.val_uncertainty)
        logging.info('Uncertainty_threshold: {:.4f}'.format(self.threshold, ))

    def pseudo_ood_val(self):
        # only for THU_Gearbox dataset
        args = self.args

        all_uncertainty = []
        for batch_idx, (inputs, _) in enumerate(self.dataloaders['pseudo']):
            inputs = inputs.to(self.device)

            # mean-std normalize
            inputs = Mean_std_process(inputs)
            inputs = inputs.to(torch.float32)

            with torch.no_grad():
                # forward
                _, uncertainty = vattn_bayes_test(inputs, self.model, args.num_MC_sampling)

            all_uncertainty.append(uncertainty)

        all_uncertainty = torch.cat(all_uncertainty)
        pseudo_ood_label = torch.ones_like(all_uncertainty)
        id_label = torch.zeros_like(self.val_uncertainty)
        all_label = torch.cat([id_label, pseudo_ood_label]).detach().cpu().numpy()
        all_uncertainty = torch.cat([self.val_uncertainty, all_uncertainty])
        pred = torch.where(all_uncertainty > self.threshold, 1, 0).detach().cpu().numpy()
        FAR, MAR = calculate_FAR_MAR(pred, all_label)
        logging.info('-' * 20 + 'OOD detection performance on Pseudo OOD set' + '-' * 20)
        logging.info('FAR: {:.4f}'.format(FAR, ))
        logging.info('MAR: {:.4f}'.format(MAR, ))

        # FPR@95TPR
        # sorted_uncertainty, _ = torch.sort(all_uncertainty, dim=0)
        # Q5_index = ((sorted_uncertainty.shape[0] + 1) * 0.05)
        # threshold = sorted_uncertainty[int(Q5_index), ]
        # mask = self.val_uncertainty > threshold
        # ratio = mask.float().mean().item()
        # logging.info('FPR@95TPR oN pseudo OOD: {:.4f}'.format(ratio, ))


    def test(self):
        args = self.args

        all_uncertainty = []
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
                outputs, uncertainty = vattn_bayes_test(inputs, self.model, args.num_MC_sampling)

            all_uncertainty.append(uncertainty)
            all_labels.append(labels)
            all_outputs.append(outputs)

        all_outputs = torch.cat(all_outputs)
        all_uncertainty = torch.cat(all_uncertainty)
        all_labels = torch.cat(all_labels)

        pred = all_outputs.argmax(dim=-1)
        new_pred = torch.where(all_uncertainty > self.threshold, len(args.train_classes), pred)
        new_labels = torch.where(all_labels > (len(args.train_classes) - 1), len(args.train_classes), all_labels)

        correct = torch.eq(new_pred, new_labels).float().sum().item()
        acc += correct
        num_samples = all_outputs.shape[0]

        acc = acc / num_samples
        logging.info('-' * 20 + 'OOD detection performance on True OOD set' + '-' * 20)
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






















        



        