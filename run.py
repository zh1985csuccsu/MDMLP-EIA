import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
 
fix_seed = 2021
# fix_seed = 2022
# fix_seed = 2023
# fix_seed = 2024
# fix_seed = 2025
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='MDMLP-EIA / Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=True, help='status')
parser.add_argument('--train_only', type=bool, required=False, default=False, help='perform training on full input dataset without validation and testing')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='MDMLP_EIA',
                    help='model name, options: [MDMLP_EIA, xPatch, DLinear, ...]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')


# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


# Patching
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')

# Moving Average
parser.add_argument('--ma_type', type=str, default='ema', help='reg, ema, dema')
parser.add_argument('--alpha', type=float, default=0.3, help='alpha')
parser.add_argument('--beta', type=float, default=0.3, help='beta')

# FreMLP
parser.add_argument('--FreMLP_embed_size', type=int, default=64, help='FreMLP_embed_size')
parser.add_argument('--FreMLP_hidden_size', type=int, default=256, help='FreMLP_hidden_size')

# DLinear
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

# CycleNet.
parser.add_argument('--cycle', type=int, default=24, help='cycle length')
parser.add_argument('--model_type', type=str, default='mlp', help='model type, options: [linear, mlp]')
parser.add_argument('--use_revin', type=int, default=1, help='1: use revin or 0: no revin')

# FilterNet
parser.add_argument('--embed_size', default=128, type=int)

# TimeMixer
parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--down_sampling_layers', type=int, default=3, help='num of down sampling layers')
parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
parser.add_argument('--down_sampling_method', type=str, default='avg',
                    help='down sampling method, only support avg, max, conv')
parser.add_argument('--use_future_temporal_feature', type=int, default=0,
                    help='whether to use future_temporal_feature; True 1 False 0')
parser.add_argument('--channel_independence', type=int, default=1,
                    help='0: channel dependence 1: channel independence for FreTS model')
parser.add_argument('--decomp_method', type=str, default='moving_avg',
                    help='method of series decompsition, only support moving_avg or dft_decomp')
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')


# optimization
parser.add_argument('--num_workers', type=int, default=2, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=10, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='sigmoid', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
# parser.add_argument('--warmup_epochs',type=int,default = 0)


# GPU
parser.add_argument('--use_gpu', type=int, default=1, help='use gpu: 1 for True, 0 for False')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')
parser.add_argument('--result_file', type=str, default='result_search.txt', help='file to save results')

# SparseTSF
parser.add_argument('--period_len', type=int, default=24, help='period_len')

# Formers
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--d_core', type=int, default=64, help='dimension of core for SOFTS model')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')


# CARD
parser.add_argument('--dp_rank', type=int, default=8)
parser.add_argument('--rescale', type=int, default=1)
parser.add_argument('--use_statistic',action='store_true', default=False)
parser.add_argument('--momentum', type=float, default=0.1, help='momentum')
parser.add_argument('--merge_size', type=int, default=2)
parser.add_argument('--warmup_epochs',type=int,default = 0)

# iTransformer
parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')

# FLinear
parser.add_argument('--train_mode', type=int,default=0)
parser.add_argument('--cut_freq', type=int,default=0)
parser.add_argument('--base_T', type=int,default=24)
parser.add_argument('--H_order', type=int,default=6)

# Amplifier
parser.add_argument('--SCI', type=int, default=0, help='1: using SCI Block, 1: not using SCI Block for Amplifier model')
parser.add_argument('--hidden_size', type=int, default=128)




if __name__ == '__main__':
    # Required for Windows: prevents DataLoader workers from re-running this module (spawn).
    # No-op on Linux; safe on both.
    import multiprocessing
    multiprocessing.freeze_support()

    args = parser.parse_args()

    if args.cut_freq == 0:
        args.cut_freq = int(args.seq_len // args.base_T + 1) * args.H_order + 10

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu == 1 else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    elif args.use_gpu:
        args.device_ids = [args.gpu]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            setting = args.model_id

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_{}_{}'.format(args.model_id,
                                                              args.model,
                                                              args.data,
                                                              args.features,
                                                              args.seq_len,
                                                              args.label_len,
                                                              args.pred_len,
                                                              args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
