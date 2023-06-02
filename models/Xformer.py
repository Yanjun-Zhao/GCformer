import os
import random
import numpy as np
import time
import warnings
import matplotlib.pyplot as plt
import math


# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

#from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.RevIN import RevIN


from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

__all__ = ['Xformer']
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 
import pdb


# from args import getArgs
import data_provider as dp
# import models

# parser = getArgs()

# from scipy.fftpack import next_fast_len
# from scipy import stats
# from lion import Lion

import numpy as np

# from scipy.special import inv_boxcox


# from ema import EMA 



# from torch_ema import ExponentialMovingAverage



class Model(nn.Module):
    def __init__(self, configs, **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
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
        
        
        self.model = DSformer(configs)
    
    
    def forward(self, x, *args, **kwargs):            # x: [Batch, Input length, Channel]
        
#         # print(x.shape)
#         if self.decomposition:
#             res_init, trend_init = self.decomp_module(x)
#             res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
#             res = self.model_res(res_init)
#             trend = self.model_trend(trend_init)
#             x = res + trend
#             x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
            
#         else:
        x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            # print(x.shape)
        x= self.model(x)
        x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        return x#,mean,std
    
    
    
    
class DSformer(nn.Module):
    def __init__(self, 
                 config,**kwargs):
        
        super().__init__()
        context_window = config.seq_len
        patch_len =config.patch_len
        stride = config.stride
        c_in =  config.enc_in
        d_model = config.d_model
        target_window = config.pred_len
        dropout = config.dropout
        self.rescale = config.rescale
        
        

        
        patch_num = int((context_window*config.rescale - patch_len)/stride + 1)
        # patch_num = int((720 - patch_len)/stride + 1)
        # self.W_pos_1 = nn.Parameter(torch.rand(c_in,d_model))
        self.W_pos_embed = nn.Parameter(torch.rand(patch_num,d_model))
        # self.W_pos_3 = nn.Parameter(torch.rand(patch_len,d_model))
        
     
        self.W_P1 = nn.Linear(d_model,d_model) 
        self.W_input_projection = nn.Linear(patch_len, d_model)  
        self.W_input_mean_bias = nn.Linear(1, d_model)
        self.W_input_std_bias = nn.Linear(1, d_model)
        self.W_P3 = nn.Linear(d_model,d_model) 
        self.W_P4 = nn.Linear(d_model,d_model) 
        # self.W_P3 = nn.Linear(patch_num, patch_num*d_model) 
        
        self.W_statistic = nn.Linear(2,d_model) 
        self.W_out = nn.Linear((patch_num+1)*d_model, target_window) 
        self.patch_len  = config.patch_len
        self.W_out_2 = nn.Linear(d_model, target_window) 
        
        self.W_P2_back = nn.Linear(d_model,patch_len)  
        self.W_P3_back = nn.Linear(d_model,patch_num) 
        self.patch_len = patch_len
        self.stride = stride
        
        self.mean_fore = nn.Linear(context_window,1)
        self.std_fore  = nn.Linear(context_window,1)
        
        self.input_dropout  = nn.Dropout(dropout)
        
        # config.d_model = config.d_model*3
        
        self.Attentions_over_token = nn.ModuleList([Attenion(config) for i in range(config.e_layers)])
        self.Attentions_over_channel = nn.ModuleList([Attenion(config) for i in range(config.e_layers)])
        self.Attentions_mlp = nn.ModuleList([nn.Linear(d_model,d_model)  for i in range(config.e_layers)])
        self.Attentions_dropout = nn.ModuleList([nn.Dropout(dropout)  for i in range(config.e_layers)])
        self.Attentions_norm = nn.ModuleList([nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model,momentum = 0.1), Transpose(1,2)) for i in range(config.e_layers)])
        # self.input_norm = nn.Sequential(Transpose(1,2), nn.InstanceNorm1d(context_window), Transpose(1,2)) 
        # self.Attentions_norm = nn.ModuleList([nn.LayerNorm(d_model) for i in range(config.e_layers)])
        
        
        self.m = nn.GELU()
    def forward(self, z, *args, **kwargs):                                                                   
        # z: [bs x nvars x seq_len]
        
        b,c,s = z.shape
        # z = F.interpolate(z.unsqueeze(1),scale_factor = scale,mode = 'bilinear').squeeze(1)
        # norm
        
        # z_mean_fore = self.mean_fore(z)
        # z_std_fore = self.std_fore(z - z_mean_fore)
        z_mean = torch.mean(z,dim = (-1),keepdims = True)
        z_std = torch.std(z,dim = (-1),keepdims = True) 
        z =  (z - z_mean)/(z_std + 1e-4)
        # z = self.input_norm(z)
        # z = F.interpolate(z.unsqueeze(1),scale_factor = self.rescale,mode = 'bilinear').squeeze(1)
 
        # z = F.interpolate(z,size = 720,mode = 'linear')
        # self.patch_len 
   
        #pdb.set_trace()    
        zcube = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        
        
        
        
       
        # z2: [bs x nvars x patch_num x patch_len]    
        
        
        # zcube = zcube.reshape(b,c,t,h/4,4)
        
        z2 = zcube
        # zcube_mean = torch.mean(z2,dim = -1,keepdims = True)
        # zcube_std = torch.mean(z2,dim = -1,keepdims = True)
 
        # z2 = (z2 - zcube_mean) / (1e-4 + zcube_std)
        # bcth -> bctd
        # print(zcube_mean.shape,z_mean.shape)
        z_embed = self.input_dropout(self.W_input_projection(z2))+ self.W_pos_embed #
        # z_m_embed = self.W_input_mean_bias(F.sigmoid(zcube_mean*z_std.unsqueeze(-1)+z_mean.unsqueeze(-1)) )
        # z_s_embed = self.W_input_std_bias(F.sigmoid(zcube_std*z_std.unsqueeze(-1)))
        
              # z_embed = z2+ self.W_pos_embed
        # print(z_embed.shape,z_m_embed.shape,z_s_embed.shape)
        # z_embed = z_embed + z_m_embed + z_s_embed
        # print(z_embed.shape,z_mean.repeat(1,1,z_embed.shape[-2],z_embed.shape[-1]).shape,z_mean.shape)
        # m_embed = z_mean.repeat(1,1,z_embed.shape[-2]).unsqueeze(-1).repeat(1,1,1,z_embed.shape[-1])
        # m_embed = z_mean.repeat(1,1,z_embed.shape[-2]).unsqueeze(-1).repeat(1,1,1,z_embed.shape[-1])
        # z_embed = torch.cat((z_embed,z_mean.repeat(1,1,z_embed.shape[-2],z_embed.shape[-1]),z_std.repeat(1,1,z_embed.shape[-2],z_embed.shape[-1])),dim = 1)
        # z_embed = torch.cat((z_embed,z_m_embed,z_s_embed),dim = -1)
        
        
        
        z_stat = torch.cat((z_mean,z_std),dim = -1)
        #pdb.set_trace()
        if z_stat.shape[-2]>1:
            z_stat = (z_stat - torch.mean(z_stat,dim =-2,keepdims = True))/( torch.std(z_stat,dim =-2,keepdims = True)+1e-4)
        z_stat = self.W_statistic(z_stat)
        # print(z_stat.squeeze(-2).shape,z_embed.shape)
        z_embed = torch.cat((z_stat.unsqueeze(-2),z_embed),dim = -2) #+ self.W_pos_embed
        b,c,t,h = z_embed.shape 
        # print(z_embed.shape , zcube.shape)
        inputs = z_embed
        for a_2,a_1,mlp,drop,norm  in zip(self.Attentions_over_token, self.Attentions_over_channel,self.Attentions_mlp ,self.Attentions_dropout,self.Attentions_norm ):
            output_1 = a_1(inputs.permute(0,2,1,3)).permute(0,2,1,3)
            output_2 = a_2(output_1)
            outputs = drop(mlp(output_1+output_2))+inputs
            outputs = norm(outputs.reshape(b*c,t,-1)).reshape(b,c,t,-1) 
            inputs = outputs
            
        # outputs = outputs* (1e-4 + zcube_std) + zcube_mean
        z_out = self.W_out(outputs.reshape(b,c,-1)) #+ self.W_out_2(torch.mean(outputs,dim = -2))
#         
        z = z_out
        # z = torch.mean(z_out.reshape(b,c//self.rescale,self.rescale,-1),dim = -2)#[:,:,:]
        # z =  z*(z_std + 1e-4) + z_mean
        z = z *(z_std+1e-4)  + z_mean
        # denorm
        # z = F.interpolate(z,size = self.patch_len ,mode = 'linear')
    
        return z#,z_mean_fore  + z_mean,z_std*torch.exp(-z_std_fore/1e3)

    
    

class Attenion(nn.Module):
    def __init__(self,configs,*args,**kwargs):
        super().__init__()
        print(configs)
        
        
        
        self.n_heads = configs.n_heads
        self.qkv = nn.Linear(configs.d_model, configs.d_model * 3, bias=True)
        
        
        self.qk_layer = nn.Linear(configs.d_model, configs.d_model * 2, bias=True)
        self.v_layer = nn.Linear(configs.d_model, configs.d_model, bias=True)
 
        self.attn_dropout = nn.Dropout(configs.dropout)
        self.head_dim = configs.d_model // configs.n_heads
        
        self.mlp = nn.Linear(configs.d_model, configs.d_model)
        self.dropout_mlp = nn.Dropout(configs.dropout)

        momentum = '5e_3'
        self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model,momentum = 0.1), Transpose(1,2))

        self.dropout_ffn = nn.Dropout(configs.dropout)
        self.norm_post = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model,momentum = 0.1), Transpose(1,2))
        self.norm_pre = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model,momentum = 0.1), Transpose(1,2))
        self.norm_pre1  = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model,momentum = 0.1), Transpose(1,2))
        self.norm_pre2  = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model,momentum = 0.1), Transpose(1,2))
        
        
        # self.norm_attn = nn.LayerNorm(configs.d_model)

#         self.dropout_ffn = nn.Dropout(configs.dropout)
#         self.norm_post =nn.LayerNorm(configs.d_model)
#         self.norm_pre = nn.LayerNorm(configs.d_model)
        # self.norm_pre1  = nn.LayerNorm(configs.d_model)
        # self.norm_pre2  = nn.LayerNorm(configs.d_model)
        
        # self.scale = nn.Parameter(torch.tensor(self.head_dim ** -0.5), requires_grad=False)
        
    
        ema_matrix = torch.zeros((1000,1000))
        alpha = configs.alpha
        ema_matrix[0][0] = 1
        for i in range(1,5):
            for j in range(i):
                ema_matrix[i][j] =  ema_matrix[i-1][j]*(1-alpha)
            ema_matrix[i][i] = alpha
            
        self.register_buffer('ema_matrix',ema_matrix)
        
        
        # self.alpha = nn.Parameter(torch.tensor([0.5]))
        # self.m = nn.Sigmoid()
        
        
        
        
        
        self.dp_rank = configs.dp_rank
        self.to_dynamic_projection_k = nn.Linear(self.head_dim, self.dp_rank)
        self.to_dynamic_projection_v = nn.Linear(self.head_dim, self.dp_rank)
        
        
        self.ff_1 = nn.Sequential(nn.Linear(configs.d_model, configs.d_ff, bias=True),
                        nn.GELU(),
                        nn.Dropout(configs.dropout),
                        nn.Linear(configs.d_ff, configs.d_model, bias=True)
                       )
        
        self.ff_2= nn.Sequential(nn.Linear(configs.d_model, configs.d_ff, bias=True),
                        nn.GELU(),
                        nn.Dropout(configs.dropout),
                        nn.Linear(configs.d_ff, configs.d_model, bias=True)
                                )     
                                 
        self.alpha = nn.Parameter(torch.zeros((self.head_dim,1))*1e2)

        # alpha = torch.empty(self.n_heads,1)
        # nn.init.kaiming_normal_(alpha)
        # self.alpha = nn.Parameter(alpha)
        
        arange = torch.arange(configs.seq_len)
        arange = torch.flip(arange, dims = (0,))  
        self.register_buffer('arange',arange)

        self.beta = nn.Parameter(torch.zeros((1,1))*1e2)
        arange_hidden = torch.arange(self.head_dim)
        arange_hidden = torch.flip(arange_hidden, dims = (0,))  
        self.register_buffer('arange_hidden',arange_hidden)

 
    def ema(self,x,dim = 4,alpha=0.5):
        
        # print(x.shape)
        # alpha = self.m(alpha)
        # alpha = 0.5
        # x_tmp = x.transpose(dim,0).contiguous()
        # ema_x = x_tmp
        # for i in range(1,x.shape[dim]):
            # ema_x[i] = (1-alpha)*ema_x[i-1]+alpha*x_tmp[i]
            # a,ba ->b
            
        length = x.shape[dim]
        if dim == 4:
            return torch.einsum('bnqhad,ga ->bnqhgd',x,self.ema_matrix[:length,:length])
        elif dim == 5:
            return torch.einsum('bnqhxa,ga ->bnqhxg',x,self.ema_matrix[:length,:length])

        

        
        
        
    def forward(self, src, *args,**kwargs):

        # src = src.unsqueeze(2)
        B,nvars, H, C, = src.shape
#         qkv = self.qkv(src).reshape(B,nvars, H, 3, self.n_heads, C // self.n_heads).permute(3, 0, 1,4, 2, 5)
   

#         q, k, v = qkv[0], qkv[1], qkv[2] 
        
        
        src1 = src/torch.linalg.norm(src,dim = -1,keepdims = True)
        qk = self.qk_layer(src1).reshape(B,nvars, H, 2, self.n_heads, C // self.n_heads).permute(3, 0, 1,4, 2, 5)
        q,k = qk[0], qk[1]
        v = self.v_layer(src).reshape(B,nvars, H, 1, self.n_heads, C // self.n_heads).permute(3, 0, 1,4, 2, 5)
        v = v[0]

        
#         beta = F.sigmoid(self.beta)#.reshape(-1,1)
#         weights = beta * (1 - beta) ** self.arange_hidden#torch.flip(arange, dims = (0,))   
            
#             # print(self.arange.shape,torch.flip(arange, dims = (0,)).shape)
#             # torch.Size([104]) torch.Size([42])

#         w_f = torch.fft.rfft(weights,n = self.head_dim*2)
#         q_f = torch.fft.rfft(q,dim = -1,n = self.head_dim*2)

#         k_f = torch.fft.rfft(k,dim = -1,n = self.head_dim*2)
#         v_f = torch.fft.rfft(v,dim = -1,n = self.head_dim*2)
#             # print(w_f.shape,q_f.shape)
#             # torch.Size([8, 12]) torch.Size([16, 7, 8, 12, 2]) 16 7 2 8 12
# #             q_f = (q_f.permute(0,1,4,2,3)*w_f)

# #             k_f = (k_f.permute(0,1,4,2,3)*w_f)

# #             q1 =torch.fft.irfft(q_f,dim = -1,n=q.shape[-2]*2)[...,:q.shape[-2]].permute(0,1,3,4,2)

# #             k1 =torch.fft.irfft(k_f,dim = -1,n=q.shape[-2]*2)[...,:q.shape[-2]].permute(0,1,3,4,2)

#         q_f = (q_f*w_f)

#         k_f = (k_f*w_f)
#         # v_f = (v_f*w_f)

#         q =torch.fft.irfft(q_f,dim = -1,n=self.head_dim*2)[...,:self.head_dim]
#         k =torch.fft.irfft(k_f,dim = -1,n=self.head_dim*2)[...,:self.head_dim]
        # v =torch.fft.irfft(v_f,dim = -1,n=self.head_dim*2)[...,:self.head_dim]
        
        if q.shape[3] !=7:
            
            
            
            alpha = F.sigmoid(self.alpha)#.reshape(-1,1)
            weights = alpha * (1 - alpha) ** self.arange[-q.shape[-2]:]#torch.flip(arange, dims = (0,))   
            
            # print(self.arange.shape,torch.flip(arange, dims = (0,)).shape)
            # torch.Size([104]) torch.Size([42])

            w_f = torch.fft.rfft(weights,n = q.shape[-2]*2)
            q_f = torch.fft.rfft(q,dim = -2,n = q.shape[-2]*2)

            k_f = torch.fft.rfft(k,dim = -2,n = q.shape[-2]*2)
            # print(w_f.shape,q_f.shape)
            # torch.Size([8, 12]) torch.Size([16, 7, 8, 12, 2]) 16 7 2 8 12
#             q_f = (q_f.permute(0,1,4,2,3)*w_f)

#             k_f = (k_f.permute(0,1,4,2,3)*w_f)

#             q1 =torch.fft.irfft(q_f,dim = -1,n=q.shape[-2]*2)[...,:q.shape[-2]].permute(0,1,3,4,2)

#             k1 =torch.fft.irfft(k_f,dim = -1,n=q.shape[-2]*2)[...,:q.shape[-2]].permute(0,1,3,4,2)

            q_f = (q_f.permute(0,1,2,4,3)*w_f)

            k_f = (k_f.permute(0,1,2,4,3)*w_f)

            q1 =torch.fft.irfft(q_f,dim = -1,n=q.shape[-2]*2)[...,:q.shape[-2]].permute(0,1,2,4,3)

            k1 =torch.fft.irfft(k_f,dim = -1,n=q.shape[-2]*2)[...,:q.shape[-2]].permute(0,1,2,4,3)

        
            attn_score_along_token = torch.einsum('bnhed,bnhfd->bnhef', q1, k1,)/ self.head_dim ** -0.5

            attn_along_token = self.attn_dropout(F.softmax(attn_score_along_token, dim=-1) )
   
            output_along_token = torch.einsum('bnhef,bnhfd->bnhed', attn_along_token, v)
            
            
        else:
            
            
            
            alpha = F.sigmoid(self.alpha)#.reshape(-1,1)
            weights = alpha * (1 - alpha) ** self.arange[-q.shape[1]:]#torch.flip(arange, dims = (0,))             

            w_f = torch.fft.rfft(weights,n = q.shape[1]*2)
            q_f = torch.fft.rfft(q,dim = 1,n = q.shape[1]*2)

            k_f = torch.fft.rfft(k,dim = 1,n = q.shape[1]*2)
            
            # print(w_f.shape,q_f.shape)
            # torch.Size([8, 12]) torch.Size([16, 12, 8, 7, 2]) -> 16 7 2 8 12
            # torch.Size([4, 8]) torch.Size([16, 8, 4, 11, 4])
            q_f = (q_f.permute(0,2,3,4,1)*w_f)

            k_f = (k_f.permute(0,2,3,4,1)*w_f)

            q1 = torch.fft.irfft(q_f,dim = -1,n=q.shape[1]*2)[...,:q.shape[1]].permute(0,4,1,2,3)

            k1 = torch.fft.irfft(k_f,dim = -1,n=q.shape[1]*2)[...,:q.shape[1]].permute(0,4,1,2,3)
            # bnqhed
            v_dp = self.to_dynamic_projection_v(v)
            v_dp = F.softmax(v_dp,dim = -1)
            v_dp = torch.einsum('bnhef,bnhec -> bnhcf',v,v_dp)
                  
            k_dp = self.to_dynamic_projection_k(k1)        
            k_dp = F.softmax(k_dp,dim = -1)
            k_dp = torch.einsum('bnhef,bnhec -> bnhcf',k1,k_dp)
            

            attn_score_along_token = torch.einsum('bnhed,bnhfd->bnhef', q1, k_dp)/ self.head_dim ** -0.5

            attn_along_token = self.attn_dropout(F.softmax(attn_score_along_token, dim=-1) )

            # print(attn_along_token.shape,v.shape)
            # torch.Size([16, 7, 1, 2, 41, 41]) torch.Size([16, 7, 1, 2, 41, 8])

            output_along_token = torch.einsum('bnhef,bnhfd->bnhed', attn_along_token, v_dp)

       
            
            
            
            
            # bnqhef
            
        
        
        output3 = output_along_token.transpose(1, 2).contiguous().view(B,nvars, -1, self.n_heads * self.head_dim)    
        output3 = output_along_token.reshape(B*nvars, -1, self.n_heads * self.head_dim)
        output3 = self.norm_pre2(output3)
        output3 = output3.reshape(B,nvars, -1, self.n_heads * self.head_dim)#.squeeze(2)


        
#         beta = F.sigmoid(self.beta)#.reshape(-1,1)
#         weights = beta * (1 - beta) ** self.arange_hidden#torch.flip(arange, dims = (0,))   
            
#             # print(self.arange.shape,torch.flip(arange, dims = (0,)).shape)
#             # torch.Size([104]) torch.Size([42])

#         w_f = torch.fft.rfft(weights,n = self.head_dim*2)
#         q_f = torch.fft.rfft(q,dim = -1,n = self.head_dim*2)

#         k_f = torch.fft.rfft(k,dim = -1,n = self.head_dim*2)
#             # print(w_f.shape,q_f.shape)
#             # torch.Size([8, 12]) torch.Size([16, 7, 8, 12, 2]) 16 7 2 8 12
# #             q_f = (q_f.permute(0,1,4,2,3)*w_f)

# #             k_f = (k_f.permute(0,1,4,2,3)*w_f)

# #             q1 =torch.fft.irfft(q_f,dim = -1,n=q.shape[-2]*2)[...,:q.shape[-2]].permute(0,1,3,4,2)

# #             k1 =torch.fft.irfft(k_f,dim = -1,n=q.shape[-2]*2)[...,:q.shape[-2]].permute(0,1,3,4,2)

#         q_f = (q_f*w_f)

#         k_f = (k_f*w_f)

#         q1 =torch.fft.irfft(q_f,dim = -1,n=self.head_dim*2)[...,:self.head_dim]

#         k1 =torch.fft.irfft(k_f,dim = -1,n=self.head_dim*2)[...,:self.head_dim]

        
        
        
        attn_score_along_hidden = torch.einsum('bnhae,bnhaf->bnhef', q,k)/ q.shape[-2] ** -0.5 #d x d#
        attn_along_hidden = self.attn_dropout(F.softmax(attn_score_along_hidden, dim=-1) )    
        output_along_hidden = torch.einsum('bnhef,bnhaf->bnhae', attn_along_hidden, v)
        
        
        output2 = output_along_hidden.transpose(1, 2).contiguous().view(B,nvars, -1, self.n_heads * self.head_dim) 
        output2 = output_along_hidden.reshape(B*nvars, -1, self.n_heads * self.head_dim)
        output2 = self.norm_pre1(output2)
        output2 = output2.reshape(B,nvars, -1, self.n_heads * self.head_dim)#.squeeze(2)
        
        
        
        
        # src2 =  self.dropout_mlp(self.mlp(output3+output2))#.squeeze(2)
        src2 =  self.dropout_mlp(self.ff_1(output3)+self.ff_2(output2))#.squeeze(2)
        # print(src2.shape,src.shape)
        src = src + src2
        src = src.reshape(B*nvars, -1, self.n_heads * self.head_dim)
        src = self.norm_attn(src)

        src = src.reshape(B,nvars, -1, self.n_heads * self.head_dim)#.squeeze(2)
        return src

