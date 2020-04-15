#-*- coding:utf-8 -*-
'''
[AI502] Deep Learning Assignment
"Attention is all you need" Implementation
20193640 Jungwon Choi
'''
import os
import sys
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
# import seaborn as sns

import PIL
from PIL import Image

import torch
import torch.nn as nn

from dataloader import WMT2014_Dataloader, Multi30k_Dataloader
from model.Transformer import Transformer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction # for short sentence
smoothing_func = SmoothingFunction().method4

FIGURE_PATH = './figures'
RESULT_PATH = './results'
CHECKPOINT_PATH ='./checkpoints/'

os.environ["CUDA_VISIBLE_DEVICES"]="3"

#===============================================================================
TEST_CHECKPOINT_LIST = [
    # 'Transformer_Multi30k_64_512_2048_64_6_8_smoothing.ckpt',
    # 'Transformer_Multi30k_64_512_2048_64_6_8_smoothing100.ckpt',
    # 'Transformer_Multi30k_64_512_2048_64_6_8_smoothing_clip.ckpt',
    # 'Transformer_Multi30k_64_512_2048_64_6_8.ckpt',
    # 'Transformer_Multi30k_64_512_2048_64_6_8_100.ckpt',
    # 'Transformer_Multi30k_64_512_2048_64_6_8_clip.ckpt',
    #===========================================================================
    # 'Transformer_Multi30k_64_512_2048_64_6_8_batch64.ckpt',
    # 'Transformer_Multi30k_128_512_2048_64_6_8_batch128.ckpt',
    # 'Transformer_Multi30k_256_512_2048_64_6_8_batch256.ckpt',
    #===========================================================================
    'Transformer_Multi30k_64_512_2048_64_6_8_exBASE.ckpt',
    #---------------------------------------------------------------------------
    'Transformer_Multi30k_64_512_2048_512_6_1_exA.ckpt',
    'Transformer_Multi30k_64_512_2048_128_6_4_exA.ckpt',
    'Transformer_Multi30k_64_512_2048_32_6_16_exA.ckpt',
    'Transformer_Multi30k_64_512_2048_16_6_32_exA.ckpt',
    #---------------------------------------------------------------------------
    'Transformer_Multi30k_64_512_2048_64_2_8_exC.ckpt',
    'Transformer_Multi30k_64_512_2048_64_4_8_exC.ckpt',
    'Transformer_Multi30k_64_512_2048_64_8_8_exC.ckpt',
    'Transformer_Multi30k_64_256_2048_32_6_8_exC.ckpt',
    'Transformer_Multi30k_64_1024_2048_128_6_8_exC.ckpt',
    'Transformer_Multi30k_64_512_1024_64_6_8_exC.ckpt',
    'Transformer_Multi30k_64_512_4096_64_6_8_exC.ckpt',
    #---------------------------------------------------------------------------
    'Transformer_Multi30k_64_1024_4096_64_6_16_exBIG.ckpt',
]

PKL_FILE_LIST = [
    # 'Transformer_Multi30k_64_512_2048_64_6_8_smoothing_results.pkl',
    # 'Transformer_Multi30k_64_512_2048_64_6_8_smoothing100_results.pkl',
    # 'Transformer_Multi30k_64_512_2048_64_6_8_smoothing_clip_results.pkl',
    # 'Transformer_Multi30k_64_512_2048_64_6_8_results.pkl',
    # 'Transformer_Multi30k_64_512_2048_64_6_8_100_results.pkl',
    # 'Transformer_Multi30k_64_512_2048_64_6_8_clip_results.pkl',
]

#===============================================================================
''' Evaluate the trained model and visualize the attention matrix '''
def evaluate_and_visualize_attention(ckpt_list):
    #===========================================================================
    for ckpt_name in ckpt_list:
        #=======================================================================
        # Parsing the hyper-parameters
        parsing_list = ckpt_name.split('.')[0].split('_')

        # Setting constants
        model_type          = parsing_list[0]
        dataset_type        = parsing_list[1]
        batch_size          = 512#512#int(parsing_list[2])
        dim_model           = int(parsing_list[3])
        dim_ff              = int(parsing_list[4])
        dim_K               = int(parsing_list[5])
        num_layers          = int(parsing_list[6])
        num_heads           = int(parsing_list[7])
        dropout_p           = 0.1   # Dropout probability
        label_smoothing_eps = 0.1   # Label smoothing epsilon
        max_src_len         = 46    # Maximum source input length (Multi30k)
        max_trg_len         = 45    # Maximum target input length (Multi30k)

        # Step1 ================================================================
        # Load dataset
        if dataset_type == 'WMT2014':
            dataloader = WMT2014_Dataloader()
        elif dataset_type == 'Multi30k':
            dataloader = Multi30k_Dataloader()
        else:
            assert False, "Please select the proper dataset."

        test_loader = dataloader.get_test_loader(batch_size=batch_size)
        print('==> DataLoader ready.')

        # Step2 ================================================================
        # Make Translation model
        if model_type == 'Transformer':
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

        # Count the model parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('==> Model ready.')

        # Step3 ====================================================================
        # Set loss function
        criterion = nn.CrossEntropyLoss(ignore_index=dataloader.pad_idx)
        print('==> Criterion ready.')

        # Step4 ====================================================================
        # Test the model
        checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, ckpt_name))
        model.load_state_dict(checkpoint['model_state_dict'])

        # test the model
        loss, metric = test(model, test_loader, criterion, dataloader)

        # Print the result on the console
        print("model                  : {}".format(model_type))
        print("dataset                : {}".format(dataset_type))
        print('# of model parameters  : {:d}'.format(num_params))
        print("batch_size             : {}".format(batch_size))
        print("test loss              : {:f}".format(loss))
        print("test PPL               : {:f}".format(metric[0]))
        print("test BLEU              : {:f}".format(metric[1]))
        print('-'*50)
    print('==> Evaluation done.')

#===============================================================================
''' Test sequence '''
def test(model, test_loader, criterion, dataloader):
    model.eval()
    device = next(model.parameters()).device.index
    losses = []
    total_iter = len(test_loader)
    sum_bleu = 0.0
    num_sentence = 0.0
    sos_idx = dataloader.sos_idx

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            srcs, trgs = batch.src.cuda(device), batch.trg.cuda(device)
            # Predict targets (Forward propagation)
            preds, enc_self, dec_self, dec_enc = model(srcs, trgs)

            # Unroll the preds and trgs
            preds_unroll = preds[1:].view(-1, preds.shape[-1])
            trgs_unroll = trgs[1:].view(-1)

            # Calculate loss
            loss = criterion(preds_unroll, trgs_unroll)
            losses.append(loss.item())

            #===================================================================
            # For BLEU score
            # Target Decoding
            trans_preds = preds
            trans_preds = trans_preds.argmax(dim=2)

            # Greedy Decoding
            # trans_preds = model.translate_forward(srcs, sos_idx, trgs.size(1))
            trans_preds = trans_preds.cpu().detach().numpy()
            trgs = trgs.cpu().detach().numpy()

            for idx, (trans_pred, trg, src) in enumerate(zip(trans_preds, trgs, srcs)):
                # Translate each sentence
                pred_sentence = dataloader.translate_sentence(trans_pred)
                trg_sentence = dataloader.translate_sentence(trg)
                src_sentence = dataloader.translate_sentence(src, type='src')
                # Calculate each sentence bleu score
                if len(pred_sentence) > 1:
                    each_belu = sentence_bleu([trg_sentence], pred_sentence,
                                        smoothing_function=smoothing_func)*100
                    sum_bleu += each_belu
                    num_sentence += 1
                    #===========================================================
                    # Monitoring the results
                    # print('SRC :', src_sentence)
                    # print('TRG :', trg_sentence)
                    # print('PRED:', pred_sentence)
                    # print(each_belu)
                    # input()
                    #===========================================================
                    # Visualize the attentions
                    # visualize_attention(enc_self, idx, src_sentence, src_sentence, 'enc', i)
                    # visualize_attention(dec_self, idx, trg_sentence, trg_sentence, 'dec', i)
                    # visualize_attention(dec_enc, idx, src_sentence, trg_sentence, 'edc', i)

            #===================================================================
            sys.stdout.write("[{:5d}/{:5d}]\r".format(i+1, total_iter))

    # Calculate average loss
    avg_loss = sum(losses)/len(losses)

    # Calculate the metrics
    # Perplexity
    ppl = np.exp(avg_loss)
    # Bilingual Evaluation Understudy Score
    bleu = sum_bleu/num_sentence

    return avg_loss, (ppl, bleu)

#===============================================================================
def visualize_loss_graph(plk_file_list):
    for plk_file_name in plk_file_list:
        #=======================================================================
        # Load results data
        plk_file_path = os.path.join(RESULT_PATH, plk_file_name)
        with open(plk_file_path, 'rb') as pkl_file:
            result_dict = pickle.load(pkl_file)

        train_loss = result_dict['train_loss']
        val_loss = result_dict['val_loss']

        #=======================================================================
        # Save figure
        RESULT_NAME = os.path.splitext(plk_file_name)[0]

        num_epoch = len(train_loss)
        epochs = np.arange(1, num_epoch+1)
        fig = plt.figure(dpi=150)
        plt.title('Train error'), plt.xlabel('Epochs'), plt.ylabel('Loss')
        plt.xlim([0, num_epoch])#, plt.ylim([0, 60])
        plt.plot(epochs, train_loss,'--', markersize=1, alpha=0.8, label='train')
        plt.plot(epochs, val_loss,'-', markersize=1, alpha=0.8, label='val')
        plt.legend()
        file_name = "Loss_graph_{}.png".format(RESULT_NAME)
        fig.savefig(os.path.join(FIGURE_PATH, file_name),format='png')
    print('==> Loss graph visualization done.')

#===============================================================================
def visualize_attention(attention_mat, batch_idx, src_sentence, trg_sentence, type, group):
    src_len = len(src_sentence)
    trg_len = len(trg_sentence)
    attention_mat = attention_mat[0][batch_idx][0]
    attention_mat = attention_mat.data.cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot()
    cax = ax.matshow(attention_mat, cmap='bone')
    fig.colorbar(cax)

    ax.set_xticklabels([''] + src_sentence + ['<eos>'], rotation=45)
    ax.set_yticklabels([''] + trg_sentence)
    # plt.show()
    if not os.path.exists(os.path.join(FIGURE_PATH,type)):
        os.makedirs(os.path.join(FIGURE_PATH,type))
    plt.savefig(os.path.join(FIGURE_PATH,type, '_'.join([type,str(group),str(batch_idx)+'.png'])))
    plt.close()

#===============================================================================
def bleu_test():
    reference = 'Hello my name is James'.split()
    hypothesis = 'Hello my name is James'.split()
    print(sentence_bleu([reference],hypothesis)*100)
    hypothesis = 'Hello my name James is'.split()
    print(sentence_bleu([reference],hypothesis,
                            smoothing_function=smoothing_func)*100)

#===============================================================================
if __name__ == '__main__':
    evaluate_and_visualize_attention(TEST_CHECKPOINT_LIST)
    # visualize_loss_graph(PKL_FILE_LIST)
    # bleu_test()
    pass
