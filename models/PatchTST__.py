__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp#,series_decomp_multi
from layers.FourierCorrelation import *
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.SelfAttention_Family import AttentionLayer, FullAttention, ProbAttention, LogSparseAttention
from layers.gconv_standalone import GConv
from layers.RevIN import RevIN
from layers.TCN import TemporalConvNet
from layers.Film3 import Film, FNO
import pdb
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear


class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.context_len
        target_window = configs.pred_len
        
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
        self.device = configs.gpu
        self.d_model = configs.d_model
        self.batch_size = configs.batch_size
        self.enc_in = configs.enc_in
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.context_len = configs.context_len
        self.h_token =configs.h_token
        self.h_channel =configs.h_channel
        self.norm_type = configs.norm_type
        self.decoder_type = configs.decoder_type
        self.global_layers = configs.global_layers
        self.global_model = configs.global_model
        self.atten_bias = configs.atten_bias
        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin,
                                  affine=affine,subtract_last=subtract_last, verbose=verbose, **kwargs)
            
                                  
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)

        
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            patch_num += 1
           

        encoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            seq_len=configs.seq_len,n_heads=configs.n_heads)
        decoder_cross_att = ProbAttention()#FullAttention()  



        self.global_encoder = AutoCorrelationLayer(
                        encoder_self_att,
                        configs.d_model, configs.d_model, configs.n_heads )
        # Decoder
        self.decoder_channel = AttentionLayer(
                        decoder_cross_att,
                        configs.h_channel, configs.n_heads)
        self.decoder_token = AttentionLayer(
                        decoder_cross_att,
                        self.h_token, configs.n_heads)
        
        self.global_layer = GConv(batch_size =configs.batch_size, d_model=configs.enc_in, d_state=configs.enc_in, l_max=configs.seq_len, channels=configs.n_heads,bidirectional=True, kernel_dim=32, n_scales=None, decay_min=2, decay_max=2, transposed=False)
        self.global_layers_FNO = FNO(1,1,self.enc_in)
        self.global_layers_Film =Film(1,1,self.enc_in,self.seq_len,self.pred_len)
        
        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last = False)
        self.local_model = DLinear.Model(configs)
        self.TCN = TemporalConvNet(configs.enc_in, [configs.d_model, configs.enc_in])
        self.linear1 = nn.Linear(self.seq_len, configs.pred_len, bias=True)
        self.linear2 = nn.Linear(self.pred_len*2, configs.pred_len, bias=True)
        self.linear6 = nn.Linear(configs.pred_len, self.h_token, bias=True)
        self.linear7 = nn.Linear(self.h_token, configs.pred_len, bias=True)
        
        self.linear9 = nn.Linear(configs.enc_in, configs.h_channel, bias=True)
        self.linear10 = nn.Linear(configs.h_channel, configs.enc_in, bias=True)
        self.ff = nn.Sequential(nn.GELU(),nn.Dropout(configs.fc_dropout))
        
        self.local_bias = nn.Parameter(torch.rand(1)*0.1+0.5)#configs.local_bias)
        self.global_bias = nn.Parameter(torch.rand(1)*0.1+0.5)#configs.global_bias)
        self.norm_channel = nn.BatchNorm1d(self.h_channel)
        self.norm_token = nn.BatchNorm1d(self.h_token)
    
    def forward(self, x):           # x: [Batch, Input length, Channel]
        
        ##############preprocess
        seq_last = x[:,-1:,:].detach()
        #seq_last = torch.mean(x[:,-10:,:],dim = (-2),keepdims = True)
        if self.norm_type == 'revin':
            x = self.revin_layer(x, 'norm')
        elif self.norm_type == 'seq_last':
            x = x - seq_last
        #x_mean = torch.mean(x,dim = (-2),keepdims = True)
        #x_std = torch.std(x,dim = (-2),keepdims = True) 
        #x = (x - x_mean)/(x_std + 1e-4)
        
        #################encoder-global
        global_x = x 
        if self.global_model == 'Gconv':
            for i in range(self.global_layers):
                global_x = self.global_layer(global_x, return_kernel=False)
            global_x = self.linear1(global_x.permute(0,2,1)).permute(0,2,1)
            #global_x = global_x[:,:self.pred_len,:]
        elif self.global_model == 'FNO':
            global_x = global_x.permute(0,2,1).unsqueeze(3)
            global_x = self.global_layers_FNO(global_x)
            global_x = self.linear1(global_x.squeeze(2)).permute(0,2,1)
        
        #################encoder-local
        x = x[:,-self.context_len:,:]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            #pdb.set_trace()
            res = self.model_res(res_init)                                   # x: [Batch, Channel, Input length]
            trend = self.model_trend(trend_init)
            local_x = res + trend  
            
            local_x = local_x.permute(0,2,1)  # x: [Batch, pred length, Channel]
        else:
            local_x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            local_x = self.model(local_x)      # x: [Batch, channel, pred length]
            local_x = local_x.permute(0,2,1)    # x: [Batch, pred length, Channel]
            
        #################decoder    
        if self.decoder_type == 'token':  
            global_x_ = self.linear6(global_x.permute(0,2,1))
            local_x_ = self.linear6(local_x.permute(0,2,1))#.permute(0,2,1)
            output_token1 = self.ff(self.decoder_token(global_x_, local_x_, local_x_)) + local_x_
            output_token2 = self.ff(self.decoder_token(local_x_, global_x_, global_x_)) + global_x_
            output_token1 = self.norm_token(output_token1.permute(0,2,1)).permute(0,2,1)
            output_token2 = self.norm_token(output_token2.permute(0,2,1)).permute(0,2,1)
            output_token = self.atten_bias*output_token1 + (1-self.atten_bias)*output_token2
            output = self.ff(self.linear7(output_token).permute(0,2,1)) 
        elif self.decoder_type == 'channel':
            global_x_channel = self.linear9(global_x)
            local_x_channel = self.linear9(local_x)#.permute(0,2,1)
            output_channel1 = self.ff(self.decoder_channel(global_x_channel, local_x_channel, local_x_channel)) + local_x_channel
            output_channel2 = self.ff(self.decoder_channel(local_x_channel, global_x_channel,global_x_channel))+ global_x_channel
            output_channel1 = self.norm_channel(output_channel1.permute(0,2,1)).permute(0,2,1)
            output_channel2 = self.norm_channel(output_channel2.permute(0,2,1)).permute(0,2,1)
            output_channel = self.atten_bias*output_channel1 + (1-self.atten_bias)*output_channel2
            output = self.ff(self.linear10(output_channel))
            
        output = output + self.local_bias*local_x+ self.global_bias*global_x
        #output = output[:,:self.pred_len,:]
        
        ##############final process
        #output = output[:,:self.pred_len,:] * (x_std + 1e-4) +x_mean
        if self.norm_type == 'revin':
            output = self.revin_layer(output, 'denorm')
        elif self.norm_type == 'seq_last':  
            output = output + seq_last
        return output,local_x,global_x,self.local_bias,self.global_bias