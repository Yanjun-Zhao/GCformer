__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp
from layers.FourierCorrelation import *
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.SelfAttention_Family import AttentionLayer, FullAttention, ProbAttention,  LogSparseAttention
from layers.gconv_standalone import GConv
from layers.Film3 import Film, FNO
from layers.RevIN import RevIN
from layers.TCN import TemporalConvNet
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear
import pdb

class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        self.context_window = configs.context_len
        target_window = configs.pred_len
        self.global_layers = configs.global_layers
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        
        
        
        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window = self.context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window = self.context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window = self.context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
        
        self.d_model = configs.d_model
        self.batch_size = configs.batch_size
        self.enc_in = configs.enc_in
        #self.local_bias = configs.local_bias
        self.atten_bias = configs.atten_bias
        self.context_len=configs.context_len
        self.pred_len=configs.pred_len
        self.seq_len = configs.seq_len
        
        
        patch_num = int((self.context_window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            patch_num += 1
        self.h_token = configs.h_token
        self.h_channel = configs.h_channel
        self.linear1 = nn.Linear(1, self.h_channel, bias=True)
        self.linear2 = nn.Linear(self.h_channel, 1, bias=True)
        self.linear3 = nn.Linear(configs.seq_len, configs.pred_len, bias=True)
        self.linear4 = nn.Linear(self.h_channel, configs.enc_in, bias=True)
        self.linear5 = nn.Linear(configs.enc_in, self.h_channel, bias=True)
        self.linear6 = nn.Linear(configs.pred_len, self.h_token, bias=True)
        self.linear7 = nn.Linear(self.h_token, configs.pred_len, bias=True)
        self.linear8 = nn.Linear(configs.context_len, configs.pred_len, bias=True)
        self.norm_channel = nn.BatchNorm1d(self.h_channel)
        self.norm_token = nn.BatchNorm1d(self.h_token)
        self.norm_FNO = nn.BatchNorm1d(self.seq_len)
        self.dp = nn.Dropout(configs.fc_dropout)
        self.gelu = nn.GELU()
        self.ff = nn.Sequential(nn.GELU(),
                                nn.Dropout(configs.fc_dropout))

        decoder_cross_att = ProbAttention()#FullAttention(configs.pred_len)#   #FullAttention()#   
        self.decoder = LogSparseAttention(n_head=configs.n_heads, n_embd=configs.pred_len, win_len=configs.seq_len, scale=False, q_len=configs.pred_len, sub_len=1)
        self.decoder_channel = AttentionLayer(
                        decoder_cross_att,
                        self.h_channel,  configs.n_heads)
        
        self.decoder_token = AttentionLayer(
                        decoder_cross_att,
                        self.h_token,  configs.n_heads)
        
        self.global_layer = GConv(configs.batch_size, d_model=configs.enc_in, d_state=configs.enc_in, l_max=configs.seq_len, channels=configs.n_heads,
                                  bidirectional=True, kernel_dim=32, n_scales=None, decay_min=2, decay_max=2, transposed=False)
        self.global_layers_GConv = nn.ModuleList([self.global_layer for i in range(configs.global_layers)])
        #self.global_layers_FNO = nn.Sequential(nn.Linear(1,self.h_channel), FNO(self.h_channel,self.h_channel,self.enc_in), nn.GELU(), nn.Dropout(configs.dropout))
        self.global_layers_FNO = nn.Sequential(FNO(1,1,self.enc_in), nn.GELU(), nn.Dropout(configs.dropout))
        self.global_layers_Film = nn.Sequential(Film(1,1,self.enc_in,self.seq_len,self.pred_len), nn.GELU(), nn.Dropout(configs.dropout))
        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last = False)
        
        self.TCN = TemporalConvNet(configs.enc_in, [configs.h_channel, configs.enc_in])
        self.local_NLinear = NLinear.Model(configs)
        self.local_Autoformer = Autoformer.Model(configs)
        self.local_Informer = Informer.Model(configs)
        
        if configs.local_bias==2:
            self.local_bias = nn.Parameter(torch.rand(1)*0.1+0.5)
        else:
            self.local_bias = configs.local_bias
        if configs.global_bias==2:
            self.global_bias = nn.Parameter(torch.rand(1)*0.1+0.5)
        else:
            self.global_bias = configs.global_bias 

        
    def forward(self, x, batch_x_mark, dec_inp, batch_y_mark):           # x: [Batch, Input length, Channel]

        ###################baseline  over toekn
        x = self.revin_layer(x, 'norm')
        #x_mean = torch.mean(x,dim = (-2),keepdims = True)
        #x_std = torch.std(x,dim = (-2),keepdims = True) 
        global_x = x#(x - x_mean)/(x_std + 1e-4)
        local_x = global_x[:,-self.context_len:,:]

        for global_layer in self.global_layers_GConv:
            global_x = global_layer(global_x, return_kernel=False)
        if self.seq_len != self.pred_len:
            #global_x = self.linear3(global_x.permute(0,2,1)).permute(0,2,1)
            global_x = global_x[:, -self.pred_len:, :] 

        batch_x_mark = batch_x_mark[:,-self.context_len:,:]
        local_x = self.local_Autoformer(local_x, batch_x_mark, dec_inp, batch_y_mark)#.permute(0,2,1)
        #local_x = self.local_Informer(local_x, batch_x_mark, dec_inp, batch_y_mark)
        '''
        #data_shape[batch len  channel]
        global_x_ = self.linear5(global_x)
        local_x_ = self.linear5(local_x)
        #pdb.set_trace()
        output1 = self.ff(self.decoder_channel(global_x_, local_x_, local_x_)) +local_x_            # x: [Batch, pred_len, d_model]
        output2 = self.ff(self.decoder_channel(local_x_, global_x_, global_x_)) + global_x_
        output1 = self.norm_channel(output1.permute(0,2,1)).permute(0,2,1)
        output2 = self.norm_channel(output2.permute(0,2,1)).permute(0,2,1)
        output = self.atten_bias*output1 + (1-self.atten_bias)*output2 #+ self.global_bias*global_x + self.local_bias*local_x
        output = self.ff(self.linear4(output)) + self.global_bias*global_x + self.local_bias*local_x # x: [Batch, pred_len, channel]
        #output = output *(x_std+1e-4)  + x_mean
        output = self.revin_layer(output, 'denorm')
        #print("global_bias:{:.3f} local_bias:{:.3f}".format(float(self.global_bias),float(self.local_bias)))
        return output,None,None,self.global_bias,self.local_bias
        '''
        
        #data_shape[batch len  channel]
        global_x_ = self.linear6(global_x.permute(0,2,1))
        local_x_ = self.linear6(local_x.permute(0,2,1))
        #pdb.set_trace()
        output1 = self.ff(self.decoder_token(global_x_, local_x_, local_x_)) +local_x_            # x: [Batch, pred_len, d_model]
        output2 = self.ff(self.decoder_token(local_x_, global_x_, global_x_)) + global_x_
        output1 = self.norm_token(output1.permute(0,2,1)).permute(0,2,1)
        output2 = self.norm_token(output2.permute(0,2,1)).permute(0,2,1)
        output = self.atten_bias*output1 + (1-self.atten_bias)*output2 #+ self.global_bias*global_x + self.local_bias*local_x
        output = self.ff(self.linear7(output).permute(0,2,1)) + self.global_bias*global_x + self.local_bias*local_x # x: [Batch, pred_len, channel]
        #output = output *(x_std+1e-4)  + x_mean
        output = self.revin_layer(output, 'denorm')
        
        return output,local_x,global_x,self.global_bias,self.local_bias
        
    