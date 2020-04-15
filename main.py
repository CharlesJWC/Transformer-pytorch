#-*- coding:utf-8 -*-
'''
[AI502] Deep Learning Assignment
"Attention is all you need" Implementation
20193640 Jungwon Choi
'''
import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import random
import pickle
import time
import os
import numpy as np

# Implementation files
from dataloader import WMT2014_Dataloader, Multi30k_Dataloader
from model.Transformer import Transformer
from train import train
from val import val

# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

VERSION_CHECK_MESSAGE = 'NOW 19-11-04 17:34'

# Set the directory paths
RESULTS_PATH = './results/'
CHECKPOINT_PATH ='./checkpoints/'

#===============================================================================
class Warmup_scheduler():
    #===========================================================================
    ''' Initialization '''
    def __init__(self, optimizer, dim_model, warmup_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.lrate = 0
        # Initial learning rate
        self.lr_init = np.power(dim_model, -0.5)

    #===========================================================================
    ''' Update learning rate '''
    def step(self):
        self.current_step += 1
        self.lrate = self.lr_init * np.min([np.power(self.current_step, -0.5),
                        np.power(self.warmup_steps, -1.5)*self.current_step])
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lrate

#===============================================================================
class Criterion_LabelSmoothing(nn.Module):
    #===========================================================================
    ''' Initialization '''
    def __init__(self, vocab_size, padding_idx, smoothing_eps=0.0):
        super(Criterion_LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing_eps
        self.smoothing_eps = smoothing_eps
        self.vocab_size = vocab_size
    #===========================================================================
    ''' Label Smoothing '''
    def forward(self, preds, trgs):
        assert preds.size(1) == self.vocab_size
        true_dist = preds.data.clone()
        true_dist.fill_(self.smoothing_eps / (self.vocab_size - 2))
        true_dist.scatter_(1, trgs.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(trgs.data == self.padding_idx)
        if len(mask) > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return self.criterion(preds, true_dist)

#===============================================================================
''' Experiment1 : Transformer performance reproducing '''
''' Experiment2 : Attention visualizations '''
def main(args):
    #===========================================================================
    # Set the file name format
    FILE_NAME_FORMAT = "{0}_{1}_{2:d}_{3:d}_{4:d}_{5:d}_{6:d}_{7:d}{8}".format(
                                    args.model, args.dataset,
                                    args.batch_size, args.dim_model,
                                    args.dim_ff, args.dim_KV, args.num_layers,
                                    args.num_heads, args.flag)

    # Set the results file path
    RESULT_FILE_NAME = FILE_NAME_FORMAT+'_results.pkl'
    RESULT_FILE_PATH = os.path.join(RESULTS_PATH, RESULT_FILE_NAME)
    # Set the checkpoint file path
    CHECKPOINT_FILE_NAME = FILE_NAME_FORMAT+'.ckpt'
    CHECKPOINT_FILE_PATH = os.path.join(CHECKPOINT_PATH, CHECKPOINT_FILE_NAME)
    BEST_CHECKPOINT_FILE_NAME = FILE_NAME_FORMAT+'_best.ckpt'
    BEST_CHECKPOINT_FILE_PATH = os.path.join(CHECKPOINT_PATH,
                                                BEST_CHECKPOINT_FILE_NAME)

    # Set the random seed same for reproducibility
    random.seed(190811)
    torch.manual_seed(190811)
    torch.cuda.manual_seed_all(190811)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Setting constants
    dim_model = args.dim_model      # Dimension of model (=Embedding size)
    dim_ff = args.dim_ff            # Dimension of FeedForward
    dim_K = args.dim_KV             # Dimension of Key(=Query)
    dim_V = args.dim_KV             # Dimension of Value
    num_layers = args.num_layers    # Number of Encoder of Decoder Layer
    num_heads = args.num_heads      # Number of heads in Multi-Head Attention
    dropout_p = args.dropout_p      # Dropout probability
    warmup_steps = 4000             # Warming up learnimg rate steps
    label_smoothing_eps = 0.1       # Label smoothing epsilon
    max_src_len = 46                # Maximum source input length (Multi30k)
    max_trg_len = 45                # Maximum target input length (Multi30k)

    # Step1 ====================================================================
    # Load dataset
    if args.dataset == 'WMT2014':
        dataloader = WMT2014_Dataloader()
    elif args.dataset == 'Multi30k':
        dataloader = Multi30k_Dataloader()
    else:
        assert False, "Please select the proper dataset."

    train_loader = dataloader.get_train_loader(batch_size=args.batch_size)
    val_loader = dataloader.get_val_loader(batch_size=args.batch_size)
    print('==> DataLoader ready.')

    # Step2 ====================================================================
    # Make Translation model
    if args.model == 'Transformer':
        src_vocab_size = len(dataloader.SRC.vocab)
        trg_vocab_size = len(dataloader.TRG.vocab)
        model = Transformer(src_vocab_size, trg_vocab_size,
                            max_src_len, max_trg_len, dim_model, dim_K,
                            num_layers, num_heads, dim_ff, dropout_p)
    else:
        assert False, "Please select the proper model."

    # Check DataParallel available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Check CUDA available
    if torch.cuda.is_available():
        model.cuda()

    print('==> Model ready.')

    # Step3 ====================================================================
    # Set loss function and optimizer (+ lrate scheduler)
    if args.smoothing:
        criterion = Criterion_LabelSmoothing(vocab_size=trg_vocab_size,
                                            padding_idx=dataloader.pad_idx,
                                            smoothing_eps=label_smoothing_eps)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=dataloader.pad_idx)
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = Warmup_scheduler(optimizer, dim_model, warmup_steps)
    print('==> Criterion and optimizer ready.')

    # Step4 ====================================================================
    # Train and validate the model
    start_epoch = 0
    best_val_metric = 0

    if args.resume:
        assert os.path.exists(CHECKPOINT_FILE_PATH), 'No checkpoint file!'
        checkpoint = torch.load(CHECKPOINT_FILE_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        lr_scheduler.current_step = checkpoint['current_step']
        best_val_metric = checkpoint['best_val_metric']

    # Save the training information
    result_data = {}
    result_data['model']            = args.model
    result_data['dataset']          = args.dataset
    result_data['target epoch']     = args.epochs
    result_data['batch_size']       = args.batch_size

    # Initialize the result lists
    train_loss = []
    train_ppl = []
    train_bleu = []

    val_loss = []
    val_ppl = []
    val_bleu = []

    # Check the directory of the file path
    if not os.path.exists(os.path.dirname(RESULT_FILE_PATH)):
        os.makedirs(os.path.dirname(RESULT_FILE_PATH))
    if not os.path.exists(os.path.dirname(CHECKPOINT_FILE_PATH)):
        os.makedirs(os.path.dirname(CHECKPOINT_FILE_PATH))
    print('==> Train ready.')

    for epoch in range(args.epochs):
        # strat after the checkpoint epoch
        if epoch < start_epoch:
            continue
        print("\n[Epoch: {:3d}/{:3d}]".format(epoch+1, args.epochs))
        epoch_time = time.time()
        #=======================================================================
        # train the model
        tloss, tmetric = train(model, train_loader, criterion,
                                    optimizer, lr_scheduler, dataloader)
        train_loss.append(tloss)
        train_ppl.append(tmetric[0])
        train_bleu.append(tmetric[1])

        # validate the model
        vloss, vmetric = val(model, val_loader, criterion, dataloader)
        val_loss.append(vloss)
        val_ppl.append(vmetric[0])
        val_bleu.append(vmetric[1])

        #=======================================================================
        current = time.time()

        # Save the current result
        result_data['current epoch']    = epoch
        result_data['train_loss']       = train_loss
        result_data['train_ppl']        = train_ppl
        result_data['train_bleu']       = train_bleu
        result_data['val_loss']         = val_loss
        result_data['val_ppl']          = val_ppl
        result_data['val_bleu']         = val_bleu

        # Save result_data as pkl file
        with open(RESULT_FILE_PATH, 'wb') as pkl_file:
            pickle.dump(result_data, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

        # Save the best checkpoint
        if vmetric[1] > best_val_metric:
            best_val_metric = vmetric[1]
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'current_step': lr_scheduler.current_step,
                'best_val_metric': best_val_metric,
                }, BEST_CHECKPOINT_FILE_PATH)

        # Save the current checkpoint
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'current_step': lr_scheduler.current_step,
            'val_metric': vmetric[0],
            # 'best_val_metric': best_val_metric,
            }, CHECKPOINT_FILE_PATH)

        # Print the information on the console
        print("model                : {}".format(args.model))
        print("dataset              : {}".format(args.dataset))
        print("batch_size           : {}".format(args.batch_size))
        print("current step         : {:d}".format(lr_scheduler.current_step))
        print("current lrate        : {:f}".format(optimizer.param_groups[0]['lr']))
        print("train/val loss       : {:f}/{:f}".format(tloss,vloss))
        print("train/val PPL        : {:f}/{:f}".format(tmetric[0],vmetric[0]))
        print("train/val BLEU       : {:f}/{:f}".format(tmetric[1],vmetric[1]))
        print("epoch time           : {0:.3f} sec".format(current - epoch_time))
        print("Current elapsed time : {0:.3f} sec".format(current - start))
    print('==> Train done.')

    print(' '.join(['Results have been saved at', RESULT_FILE_PATH]))
    print(' '.join(['Checkpoints have been saved at', CHECKPOINT_FILE_PATH]))

#===============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer Implementation')
    parser.add_argument('--model', default='Transformer', type=str)
    parser.add_argument('--dataset', default='Multi30k', type=str,
                                        help='WMT2014, Multi30k')
    parser.add_argument('--epochs', default=23, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--dim_model', default=512, type=int)
    parser.add_argument('--dim_ff', default=2048, type=int)
    parser.add_argument('--dim_KV', default=64, type=int)
    parser.add_argument('--num_layers', default=6, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--dropout_p', default=0.1, type=float)
    parser.add_argument('--flag', default='', type=str)
    parser.add_argument('--smoothing', action='store_true')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    # Code version check message
    print(VERSION_CHECK_MESSAGE)

    start = time.time()
    #===========================================================================
    main(args)
    #===========================================================================
    end = time.time()
    print("Total elapsed time: {0:.3f} sec\n".format(end - start))
    print("[Finih time]",time.strftime('%c', time.localtime(time.time())))
