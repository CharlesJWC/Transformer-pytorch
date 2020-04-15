#-*- coding:utf-8 -*-
'''
[AI502] Deep Learning Assignment
"Attention is all you need" Implementation
20193640 Jungwon Choi
'''
import torch
import torch.nn as nn
import numpy as np
import sys

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction # for short sentence
smoothing_func = SmoothingFunction().method4

#===============================================================================
''' Validate sequence '''
def val(model, val_loader, criterion, dataloader):
    model.eval()
    device = next(model.parameters()).device.index
    losses = []
    total_iter = len(val_loader)
    sum_bleu = 0.0
    num_sentence = 0.0
    sos_idx = dataloader.sos_idx

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            srcs, trgs = batch.src.cuda(device), batch.trg.cuda(device)
            # Predict targets (Forward propagation)
            preds, _, _, _ = model(srcs, trgs)

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

            for trans_pred, trg in zip(trans_preds, trgs):
                # Translate each sentence
                pred_sentence = dataloader.translate_sentence(trans_pred)
                trg_sentence = dataloader.translate_sentence(trg)
                # Calculate each sentence bleu score
                if len(pred_sentence) > 1:
                    sum_bleu += sentence_bleu([trg_sentence], pred_sentence,
                                        smoothing_function=smoothing_func)*100
                    num_sentence += 1
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
