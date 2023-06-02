# coding=utf-8
# author=maziqing
# email=maziqing.mzq@alibaba-inc.com


import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np
from scipy import signal
from scipy import linalg as la
from scipy import special as ss
from einops import rearrange, repeat, reduce
import pdb


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def transition(measure, N, **measure_args):
    # Laguerre (translated)
    if measure == 'lagt':
        b = measure_args.get('beta', 1.0)
        A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    # Legendre (translated)
    elif measure == 'legt':
        Q = np.arange(N, dtype=np.float64)
        R = (2*Q + 1) ** .5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.)**(i-j), 1) * R[None, :]
        B = R[:, None]
        A = -A
    # Legendre (scaled)
    elif measure == 'legs':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy() # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
    elif measure == 'fourier':
        freqs = np.arange(N//2)
        d = np.stack([np.zeros(N//2), freqs], axis=-1).reshape(-1)[1:]
        A = 2*np.pi*(-np.diag(d, 1) + np.diag(d, -1))
        B = np.zeros(N)
        B[0::2] = 2
        B[0] = 2**.5
        A = A - B[:, None] * B[None, :]
        # A = A - np.eye(N)
        B *= 2**.5
        B = B[:, None]

    return A, B


def basis(method, N, vals, c=0.0, truncate_measure=True):
    """
    vals: list of times (forward in time)
    returns: shape (T, N) where T is length of vals
    """
    if method == 'legt':
        eval_matrix = ss.eval_legendre(np.arange(N)[:, None], 2 * vals - 1).T
        eval_matrix *= (2 * np.arange(N) + 1) ** .5 * (-1) ** np.arange(N)
    elif method == 'legs':
        _vals = np.exp(-vals)
        eval_matrix = ss.eval_legendre(np.arange(N)[:, None], 1 - 2 * _vals).T  # (L, N)
        eval_matrix *= (2 * np.arange(N) + 1) ** .5 * (-1) ** np.arange(N)
    elif method == 'lagt':
        vals = vals[::-1]
        eval_matrix = ss.eval_genlaguerre(np.arange(N)[:, None], 0, vals)
        eval_matrix = eval_matrix * np.exp(-vals / 2)
        eval_matrix = eval_matrix.T
    elif method == 'fourier':
        cos = 2 ** .5 * np.cos(2 * np.pi * np.arange(N // 2)[:, None] * (vals))  # (N/2, T/dt)
        sin = 2 ** .5 * np.sin(2 * np.pi * np.arange(N // 2)[:, None] * (vals))  # (N/2, T/dt)
        cos[0] /= 2 ** .5
        eval_matrix = np.stack([cos.T, sin.T], axis=-1).reshape(-1, N)  # (T/dt, N)
    #     print("eval_matrix shape", eval_matrix.shape)

    if truncate_measure:
        eval_matrix[measure(method)(vals) == 0.0] = 0.0

    p = torch.tensor(eval_matrix)
    p *= np.exp(-c * vals)[:, None]  # [::-1, None]
    return p


def measure(method, c=0.0):
    if method == 'legt':
        fn = lambda x: np.heaviside(x, 0.0) * np.heaviside(1.0-x, 0.0)
    elif method == 'legs':
        fn = lambda x: np.heaviside(x, 1.0) * np.exp(-x)
    elif method == 'lagt':
        fn = lambda x: np.heaviside(x, 1.0) * np.exp(-x)
    elif method in ['fourier']:
        fn = lambda x: np.heaviside(x, 1.0) * np.heaviside(1.0-x, 1.0)
    else: raise NotImplementedError
    fn_tilted = lambda x: np.exp(c*x) * fn(x)
    return fn_tilted


class HiPPO(nn.Module):
    """ Linear time invariant x' = Ax + Bu """

    def __init__(self, N, seq_len, method='legt', dt=1.0, T=1.0, discretization='bilinear',
                 scale=False, c=0.0, fast=True):
        """
        N: the order of the HiPPO projection
        dt: discretization step size - should be roughly inverse to the length of the sequence
        """
        super().__init__()
        self.method = method
        self.N = N
        self.dt = dt
        self.T = T
        self.c = c
        self.pred_len = int(1/dt)
        self.fast = fast

        A, B = transition(method, N)
        A = A + np.eye(N) * c
        self.A = A
        self.B = B.squeeze(-1)
        self.measure_fn = measure(method)

        C = np.ones((1, N))
        D = np.zeros((1,))
        dA, dB, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)
        dB = dB.squeeze(-1)

        self.register_buffer('dA', torch.Tensor(dA))  # (N, N)
        self.register_buffer('dB', torch.Tensor(dB))  # (N,)
        if fast:
            Ks = []
            AB = self.dB
            A = self.dA
            for i in range(0, seq_len):
                AB = A @ AB
                Ks += [AB]
            K = torch.stack(Ks).transpose(-1, -2)
            self.register_buffer('K', torch.Tensor(K))

        self.vals = np.arange(0.0, T, dt)
        self.eval_matrix = basis(self.method, self.N, self.vals, c=self.c).to(device).to(torch.float32)  # (T/dt, N)
        self.measure = measure(self.method)(self.vals)

    def project(self, inputs):
        if self.fast:
            k = self.K
            T = inputs.shape[-1]
            k_f = torch.fft.rfft(k, n=2 * T)  # (H L)
            # print(k_f.shape)
            u_f = torch.fft.rfft(inputs, n=2 * T)  # (B H L)
            y_f = torch.einsum('bedl,nl->bednl', u_f, k_f)
            y = torch.fft.irfft(y_f, n=2 * T)[..., :T]
            y = y[..., -1]
            return y

        # inputs = rearrange(inputs, 'b l e -> l b e')
        # """
        # inputs : (length, ...)
        # output : (length, ..., N) where N is the order of the HiPPO projection
        # """
        # inputs = inputs.unsqueeze(-1)
        # u = inputs * self.dB  # (length, ..., N)
        # c = torch.zeros(u.shape[1:]).to(inputs)
        # cs = []
        # for f in inputs:
        #     c = F.linear(c, self.dA) + self.dB * f
        #     cs.append(c)
        # out = torch.stack(cs, dim=0)
        # out = rearrange(out, 'l b e n -> b l e n')
        # return out

    def reconstruct(self, c, evals=None):  # TODO take in a times array for reconstruction
        """
        c: (..., N,) HiPPO coefficients (same as x(t) in S4 notation)
        output: (..., L,)
        """
        eval_matrix = self.eval_matrix
        y = c @ (eval_matrix.T)
        return y


class FNO(nn.Module):
    def __init__(self, in_channels, out_channels, enc_in, modes=32):
        super(FNO, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = (1 / (in_channels*out_channels))
        self.weights = nn.Parameter(scale * torch.rand(enc_in, in_channels,
                                                       out_channels, modes, dtype=torch.cfloat).to(device))

    def forward(self, x):
        B, E, L, D = x.shape  # shape=(batch, num_seq,seq_length, hidden_dim)
        x = rearrange(x, 'b e l d -> b e d l')
        #pdb.set_trace()
        x_f = torch.fft.rfft(x, n=2*L)
        k_f = self.weights
        y_f = torch.einsum('bedm,edcm->becm', x_f[..., :self.modes], k_f)
        y = torch.fft.irfft(y_f, n=2 * L)[..., :L]
        return y


class Film(nn.Module):
    def __init__(self, in_channels, out_channels, enc_in, seq_len, pred_len, N=128):
        super(Film, self).__init__()
        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.legt = HiPPO(method='legt', seq_len=seq_len, N=N, dt=1. / pred_len, fast=True)
        scale = (1 / (in_channels*out_channels))
        self.weights = nn.Parameter(scale * torch.rand(enc_in, in_channels,
                                                       out_channels, N).to(device))

    def forward(self, x):
        B, E, L, D = x.shape  # shape=(batch, num_seq,seq_length, hidden_dim)
        x = rearrange(x, 'b e l d -> b e d l')
        x_f = self.legt.project(x)
        k_f = self.weights
        y_f = torch.einsum('bedn,edcn->becn', x_f, k_f)
        y = self.legt.reconstruct(y_f)
        return y


if __name__ == '__main__':
    x = 7
    class Configs(object):
        ours = 0
        ab = 3
        # modes1 = 32
        seq_len = 192
        label_len = 96
        pred_len = 192
        output_attention = True
        enc_in = x
        dec_in = x
        d_model = 16
        embed = 'timeF'
        dropout = 0.05
        freq = 'h'
        factor = 1
        n_heads = 8
        d_ff = 16
        e_layers = 2
        d_layers = 1
        moving_avg = 25
        c_out = 1
        activation = 'gelu'
        wavelet = 0
        modes = 32
        N = 256

    configs = Configs()
    model1 = FNO(in_channels=1, out_channels=1, enc_in=configs.enc_in, modes=64)
    model2 = Film(in_channels=1, out_channels=1, enc_in=configs.enc_in, N=128,
                  seq_len=configs.seq_len, pred_len=configs.pred_len)

    enc = torch.randn([3, 1, configs.seq_len, configs.enc_in])
    out1 = model1.forward(enc)
    out2 = model2.forward(enc)
    assert out2.shape == out1.shape
    a = 1


