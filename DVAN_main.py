#!/usr/bin/python
# -*- coding:utf-8 -*-

import argparse
import os
from datetime import datetime
import logging
import warnings

from utils.logger import setlogger
from utils.DVAN_train import train_DVAN_utils
from utils.Vanilla_train import train_Vanilla_utils
from utils.Ensemble_train import train_ensemble_utils
from utils.MC_train import train_MC_utils

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # model parameters
    parser.add_argument('--model_name', type=str, default='DVAN', help='the name of the model')
    parser.add_argument('--method', type=str, default='VCBAM',
                        choices=['VSE', 'VCBAM', 'VSA', 'softmax-output', 'MC-dropout', 'ensemble', 'SE', 'CBAM', 'SA'])

    # data parameters
    parser.add_argument('--data_name', type=str, default='locomotive', choices=['THU_gearbox', 'locomotive'])
    parser.add_argument('--data_length', type=int, default=1024)
    parser.add_argument('--in_channel', type=int, default=1)

    parser.add_argument('--num_train_samples', type=int, default=200)
    parser.add_argument('--train_classes', type=list, default=[0, 1, 2, 3, 4])
    parser.add_argument('--train_op_conditions', type=list, default=[6], help='locomotive dataset do not have that')
    parser.add_argument('--train_noise_SNR', type=int, default=50)
    parser.add_argument('--num_classes', type=int, default=5)

    parser.add_argument('--num_val_samples', type=int, default=50)
    parser.add_argument('--val_classes', type=list, default=[0, 1, 2, 3, 4])
    parser.add_argument('--val_op_conditions', type=list, default=[6])
    parser.add_argument('--val_noise_SNR', type=int, default=50)

    parser.add_argument('--pseudo_noise_SNR', type=int, default=-5)

    parser.add_argument('--num_test_samples', type=int, default=50)
    parser.add_argument('--test_classes', type=list, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument('--test_op_conditions', type=list, default=[6])
    parser.add_argument('--test_noise_SNR', type=int, default=50)

    # training parameters
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='the directory to save the model')
    parser.add_argument('--batch_size', type=int, default=128, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')
    parser.add_argument('--num_MC_sampling', type=int, default=20)
    parser.add_argument('--is_MC_shaping', type=bool, default=True)
    parser.add_argument('--prior_mu', type=float, default=0)
    parser.add_argument('--prior_std', type=float, default=0.2)
    parser.add_argument('--MC_shaping_mu', type=float, default=0)
    parser.add_argument('--MC_shaping_std', type=float, default=1)

    # optimization information
    parser.add_argument('--lr', type=float, default=1e-3, help='the initial learning rate')
    parser.add_argument('--beta_type', type=str, default="Blundell")
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='L2 Regularization')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='15,25', help='the learning rate decay for step and stepLR')
    parser.add_argument('--epoch', type=int, default=30)

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    # Prepare the saving path for the model
    sub_dir = args.model_name + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, 'train.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    if args.method == 'VSE' or args.method == 'VCBAM' or args.method == 'VSA':
        trainer = train_DVAN_utils(args, save_dir)
    elif args.method == 'softmax-output' or args.method == 'SE' or args.method == 'CBAM' or args.method == 'SA':
        trainer = train_Vanilla_utils(args, save_dir)
    elif args.method == 'ensemble':
        trainer = train_ensemble_utils(args, save_dir)
    elif args.method == 'MC-dropout':
        trainer = train_MC_utils(args, save_dir)

    trainer.setup()
    trainer.train()
    # trainer.pseudo_ood_val()
    trainer.test()