import torch
import torch.nn as nn
import torch.fft as fft

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.mean = None
        self.stdev = None
        self.last = None
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class Mlp_feat(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super(Mlp_feat, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):  # B, L, D -> B, L, D
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Mlp_time(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super(Mlp_time, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):  # B, D, L -> B, D, L
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Mixer_Layer(nn.Module):
    def __init__(self, time_dim, feat_dim):
        super(Mixer_Layer, self).__init__()
        self.batchNorm2D = nn.BatchNorm1d(time_dim)
        self.MLP_time = Mlp_time(time_dim, time_dim)
        self.MLP_feat = Mlp_feat(feat_dim, feat_dim)

    def forward(self, x):  # # B, L, D -> B, L, D
        res1 = x
        x = self.batchNorm2D(x)
        x = self.MLP_time(x.permute(0, 2, 1)).permute(0, 2, 1)  # B, L, D -> B, D, L -> B, D, L -> B, L, D
        x = x + res1

        res2 = x
        x = self.batchNorm2D(x)
        x = self.MLP_feat(x)  # B, L, D -> B, L, D
        x = x + res2

        return x


class Backbone(nn.Module):
    def __init__(self, configs):
        super(Backbone, self).__init__()

        self.seq_len = seq_len = configs.seq_len
        self.pred_len = pred_len = configs.pred_len
        self.enc_in = enc_in = configs.enc_in
        self.layer_num = layer_num = 1

        self.mix_layer = Mixer_Layer(seq_len, enc_in)
        self.cae = CAE(self.enc_in, self.enc_in)
        self.conv_layer = nn.Conv1d(in_channels=self.enc_in, out_channels=self.enc_in, kernel_size=3, padding=1)

    def forward(self, x):  # B, L, D -> B, H, D
        res3 = self.cae(x)  # Ensure `res3` shape matches the desired output
        n_block = 6
        for _ in range(n_block):
            x = self.mix_layer(x)  # Ensure `x` shape matches `self.seq_len`

        # Ensure `res3` and `x` have matching dimensions for addition
        if res3.size(1) != x.size(1):
            
            # print(f"Adjusting res3 from {res3.shape} to match {x.shape}")
            res3 = res3[:, :x.size(1), :]  # Adjust `res3` to match `x`

        return res3 + x

import torch.nn as nn

class CAE(nn.Module):
    def __init__(self, input_channels, feature_dim):
        super(CAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, stride=2, padding=1),  # B, 64, L/2
            nn.ReLU(True),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),  # B, 128, L/4
            nn.ReLU(True),
            nn.Conv1d(128, feature_dim, kernel_size=3, stride=2, padding=1),  # B, feature_dim, L/8
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(feature_dim, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # B, 128, L/4
            nn.ReLU(True),
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # B, 64, L/2
            nn.ReLU(True),
            nn.ConvTranspose1d(64, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # B, input_channels, L
            nn.Tanh()
        )

    def forward(self, x):
        # Reshape x to match the expected input shape [batch_size, input_channels, seq_len]
        x = x.permute(0, 2, 1)  # Change shape from [batch_size, seq_len, num_channels] to [batch_size, num_channels, seq_len]
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        x_reconstructed = x_reconstructed.permute(0, 2, 1)  # Change shape back to [batch_size, seq_len, num_channels]
        return x_reconstructed



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.rev = RevIN(configs.enc_in)

        self.backbone = Backbone(configs)

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forward(self, x, batch_x_mark, dec_inp, batch_y_mark):
        z = self.rev(x, 'norm')  # B, L, D -> B, L, D
        z = self.backbone(z)  # B, L, D -> B, H, D
        z = self.rev(z, 'denorm')  # B, H, D -> B, H, D
     
        return z
