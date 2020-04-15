#-*- coding:utf-8 -*-
'''
[AI502] Deep Learning Assignment
"Attention is all you need" Implementation
20193640 Jungwon Choi
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#===============================================================================
class ScaledDotProductAttention(nn.Module):
    ''' Initialization '''
    #===========================================================================
    def __init__(self, dim_K):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = np.sqrt(dim_K)
        self.softmax = nn.Softmax(dim=-1)
    #===========================================================================
    def forward(self, Q, K, V, atten_mask):
        scores = torch.matmul(Q, K.transpose(-1,-2)) / self.scale # Scale
        if atten_mask is not None:
            scores.masked_fill_(atten_mask.bool(), -1e9) # Option
        attention = self.softmax(scores)
        context = torch.matmul(attention, V)
        return context, attention

#===============================================================================
class MultiHeadAttention(nn.Module):
    #===========================================================================
    ''' Initialization '''
    def __init__(self, dim_model, dim_K, num_heads, dropout_p):
        super(MultiHeadAttention, self).__init__()
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_K = dim_K

        self.weight_Q = nn.Linear(dim_model, dim_K*num_heads)
        self.weight_K = nn.Linear(dim_model, dim_K*num_heads)
        self.weight_V = nn.Linear(dim_model, dim_K*num_heads)
        self.scaledotproduct_attention = ScaledDotProductAttention(dim_K)
        self.fc = nn.Linear(num_heads*dim_K, dim_model)
        self.dropout = nn.Dropout(dropout_p)
        self.layernorm = nn.LayerNorm(dim_model)
    #===========================================================================
    def forward(self, query, key, value, atten_mask=None):
        batch_size = query.size(0)
        # Foward each linear layers
        Q = self.weight_Q(query)
        K = self.weight_K(key)
        V = self.weight_V(value)
        # Transfrom for Scaled Dot-Product Attention
        Q = Q.view(batch_size, -1, self.num_heads, self.dim_K).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.dim_K).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.dim_K).transpose(1, 2)
        if atten_mask is not None:
            atten_mask = atten_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        # Forward Scaled Dot-Product Attention
        context, attention = self.scaledotproduct_attention(Q, K, V, atten_mask)
        # Transfrom for Concat
        context = context.transpose(1, 2).contiguous()
        # Concat heads
        context = context.view(batch_size, -1, self.num_heads*self.dim_K)
        # Foward linear layer
        out = self.fc(context)
        out = self.dropout(context)
        # residual sum
        out = self.layernorm(query + out)
        return out, attention

#===============================================================================
class PositionwiseFeedforward(nn.Module):
    #===========================================================================
    ''' Initialization '''
    def __init__(self, dim_model, dim_ff, dropout_p):
        super(PositionwiseFeedforward, self).__init__()
        self.fc1 = nn.Linear(dim_model, dim_ff)
        self.fc2 = nn.Linear(dim_ff, dim_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.layernorm = nn.LayerNorm(dim_model)
    #===========================================================================
    def forward(self, x):
        #  Sub-layer
        out = self.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        # residual sum
        out = self.layernorm(x + out)
        return out

#===============================================================================
class EncoderLayer(nn.Module):
    #===========================================================================
    ''' Initialization '''
    def __init__(self, dim_model, dim_K, num_heads, dim_ff, dropout_p):
        super(EncoderLayer, self).__init__()

        self.encoder_self_atten = MultiHeadAttention(dim_model, dim_K,
                                                    num_heads, dropout_p)
        self.poswise_feedfoward = PositionwiseFeedforward(dim_model,
                                                    dim_ff, dropout_p)
    #===========================================================================
    def forward(self, enc_inputs, enc_self_atten_mask):
        # Self-Attention
        enc_outputs, enc_self_atten = self.encoder_self_atten(enc_inputs,
                                    enc_inputs, enc_inputs, enc_self_atten_mask)
        # Feed-Forward
        enc_outputs = self.poswise_feedfoward(enc_outputs)
        return enc_outputs, enc_self_atten

#===============================================================================
class Encoder(nn.Module):
    #===========================================================================
    ''' Initialization '''
    def __init__(self, src_vocab_size, max_src_len, dim_model, dim_K,
                        num_layers, num_heads, dim_ff, dropout_p):
        super(Encoder, self).__init__()

        self.src_embedding = nn.Embedding(src_vocab_size, dim_model)
        self.pos_embedding = nn.Embedding.from_pretrained(self.sinusoid_encoding(
                                        max_src_len+1, dim_model), freeze=True)
        self.enc_layers = nn.ModuleList([EncoderLayer(dim_model, dim_K,
                                                num_heads, dim_ff, dropout_p)
                                                for _ in range(num_layers)])
        self.scale = np.sqrt(dim_K)
    #===========================================================================
    def forward(self, src_inputs):
        # Source embedding
        pos = torch.arange(0, src_inputs.size(1)).unsqueeze(0).repeat(src_inputs.size(0), 1)
        enc_outputs = self.src_embedding(src_inputs)*self.scale + self.pos_embedding(pos.cuda())
        # Create encoder attention mask
        enc_self_atten_mask = self.attention_pad_mask(src_inputs, src_inputs)
        # Foward encoder layers
        enc_self_attens = []
        for enc_layer in self.enc_layers:
            enc_outputs, enc_self_atten = enc_layer(enc_outputs, enc_self_atten_mask)
            enc_self_attens.append(enc_self_atten)
        return enc_outputs, enc_self_attens
    #===========================================================================
    def sinusoid_encoding(self, num_position, dim_model):
        def cal_angle(pos, hid_idx):
            return pos / np.power(10000, 2 * (hid_idx // 2) / dim_model)
        def pos_angle_vector(pos):
            return [cal_angle(pos, hid_j) for hid_j in range(dim_model)]

        sinusoid_table = np.array([pos_angle_vector(pos_i) for pos_i in range(num_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table)
    #===========================================================================
    def attention_pad_mask(self, sequence_Q, sequence_K):
        batch_size, length_Q = sequence_Q.size()
        batch_size, length_K = sequence_K.size()
        attention_mask = sequence_K.data.eq(0).unsqueeze(1)
        return attention_mask.expand(batch_size, length_Q, length_K)

#===============================================================================
class DecoderLayer(nn.Module):
    #===========================================================================
    ''' Initialization '''
    def __init__(self, dim_model, dim_K, num_heads, dim_ff, dropout_p):
        super(DecoderLayer, self).__init__()
        self.decoder_self_atten = MultiHeadAttention(dim_model, dim_K,
                                                    num_heads, dropout_p)
        self.decoder_encoder_atten = MultiHeadAttention(dim_model, dim_K,
                                                    num_heads, dropout_p)
        self.poswise_feedfoward = PositionwiseFeedforward(dim_model,
                                                    dim_ff, dropout_p)
    #===========================================================================
    def forward(self, dec_inputs, enc_outputs, dec_self_atten_mask, dec_enc_atten_mask):
        dec_outputs, dec_self_atten = self.decoder_self_atten(dec_inputs, dec_inputs, dec_inputs, dec_self_atten_mask)
        dec_outputs, dec_enc_atten = self.decoder_encoder_atten(dec_outputs, enc_outputs, enc_outputs, dec_enc_atten_mask)
        dec_outputs = self.poswise_feedfoward(dec_outputs)
        return dec_outputs, dec_self_atten, dec_enc_atten

#===============================================================================
class Decoder(nn.Module):
    #===========================================================================
    ''' Initialization '''
    def __init__(self, trg_vocab_size, max_trg_len, dim_model, dim_K,
                        num_layers, num_heads, dim_ff, dropout_p):
        super(Decoder, self).__init__()
        self.trg_embedding = nn.Embedding(trg_vocab_size, dim_model)
        self.pos_embedding = nn.Embedding.from_pretrained(self.sinusoid_encoding(
                                        max_trg_len+1, dim_model), freeze=True)
        self.dec_layers = nn.ModuleList([DecoderLayer(dim_model, dim_K,
                                                num_heads, dim_ff, dropout_p)
                                     for _ in range(num_layers)])
        self.scale = np.sqrt(dim_K)

    #===========================================================================
    def forward(self, trg_inputs, src_inputs, enc_outputs):
        # Target embedding
        pos = torch.arange(0, trg_inputs.size(1)).unsqueeze(0).repeat(trg_inputs.size(0), 1)
        dec_outputs = self.trg_embedding(trg_inputs)*self.scale + self.pos_embedding(pos.cuda())
        # Create decoder attention mask & subsequent mask & dec-enc attention
        dec_self_atten_pad_mask = self.attention_pad_mask(trg_inputs, trg_inputs)
        dec_self_atten_subsequent_mask = self.attention_subsequent_mask(trg_inputs)
        dec_self_atten_mask = torch.gt((dec_self_atten_pad_mask+dec_self_atten_subsequent_mask.cuda()), 0)
        dec_enc_atten_mask = self.attention_pad_mask(trg_inputs, src_inputs)

        # Foward decoder layers
        dec_self_attens, dec_enc_attens = [], []
        for dec_layer in self.dec_layers:
            dec_outputs, dec_self_atten, dec_enc_atten = dec_layer(dec_outputs, enc_outputs, dec_self_atten_mask, dec_enc_atten_mask)
            dec_self_attens.append(dec_self_atten)
            dec_enc_attens.append(dec_enc_atten)
        return dec_outputs, dec_self_attens, dec_enc_attens

    #===========================================================================
    def sinusoid_encoding(self, num_position, dim_model):
        def cal_angle(pos, hid_idx):
            return pos / np.power(10000, 2 * (hid_idx // 2) / dim_model)
        def pos_angle_vector(pos):
            return [cal_angle(pos, hid_j) for hid_j in range(dim_model)]

        sinusoid_table = np.array([pos_angle_vector(pos_i) for pos_i in range(num_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table)
    #===========================================================================
    def attention_pad_mask(self, sequence_Q, sequence_K):
        batch_size, length_Q = sequence_Q.size()
        batch_size, length_K = sequence_K.size()
        attention_mask = sequence_K.data.eq(0).unsqueeze(1)
        return attention_mask.expand(batch_size, length_Q, length_K).byte()
    #===========================================================================
    def attention_subsequent_mask(self, sequence):
        atten_shape = [sequence.size(0), sequence.size(1), sequence.size(1)]
        subsequent_mask = np.triu(np.ones(atten_shape), k=1)
        subsequent_mask = torch.from_numpy(subsequent_mask).byte()
        return subsequent_mask

#===============================================================================
class Transformer(nn.Module):
    #===========================================================================
    ''' Initialization '''
    def __init__(self, src_vocab_size, trg_vocab_size,
                        max_src_len, max_trg_len, dim_model, dim_K,
                        num_layers, num_heads, dim_ff, dropout_p):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, max_src_len, dim_model, dim_K,
                                num_layers, num_heads, dim_ff, dropout_p)
        self.decoder = Decoder(trg_vocab_size, max_trg_len, dim_model, dim_K,
                                num_layers, num_heads, dim_ff, dropout_p)
        self.projection = nn.Linear(dim_model, trg_vocab_size, bias=False)
    #===========================================================================
    def forward(self, src_inputs, trg_inputs):
        enc_outputs, enc_self_attens = self.encoder(src_inputs)
        dec_outputs, dec_self_attens, dec_enc_attens = self.decoder(trg_inputs,
                                                    src_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)
        return dec_logits, enc_self_attens, dec_self_attens, dec_enc_attens
    #===========================================================================
    ''' Greedy decoder method '''
    def translate_forward(self, src_inputs, sos_idx, trg_len=45):
        batch_size, _ = src_inputs.size()
        trg_inputs = torch.zeros((batch_size,trg_len)).type_as(src_inputs.data)
        enc_outputs, _ = self.encoder(src_inputs)
        next_idx = sos_idx
        target_pos = 0
        while target_pos < trg_len:
            trg_inputs[:, target_pos] = next_idx
            dec_outputs, _, _ = self.decoder(trg_inputs, src_inputs, enc_outputs)
            out = self.projection(dec_outputs)
            out = out[:, target_pos].argmax(dim=1)
            next_idx = out
            target_pos += 1
        return trg_inputs
