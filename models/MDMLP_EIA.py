import torch
import torch.nn as nn
import numpy as np
import math
from layers.decomp import DECOMP
from layers.revin import RevIN
from layers.fre_network import FreNetwork


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.type = configs.model_type

        self.attention_mlp = nn.Sequential(
            nn.Linear(self.channels * 2, self.channels),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(self.channels, self.channels),
            nn.Sigmoid()
        )
        self._init_eia_weights()

        # Normalization
        self.revin = configs.revin
        self.revin_layer = RevIN(configs.enc_in,affine=True,subtract_last=False)

        # Moving Average
        self.ma_type = configs.ma_type
        alpha = configs.alpha       # smoothing factor for EMA (Exponential Moving Average)
        beta = configs.beta         # smoothing factor for DEMA (Double Exponential Moving Average)

        self.decomp = DECOMP(self.ma_type, alpha, beta, kernel_size=25)
        self.Seasonal = FreNetwork(configs)

        cof = math.ceil((math.sqrt(self.channels))/5)

        if self.type == 'mlp':
            self.Trend = nn.Sequential(
                nn.Linear(self.seq_len, self.seq_len* cof),
                nn.Tanh(),
                nn.Dropout(0.4),
                nn.Linear(self.seq_len * cof, self.seq_len * cof),
                nn.Tanh(),
                nn.Dropout(0.4),
                nn.Linear(self.seq_len * cof, self.pred_len)
            )

        elif self.type == 'linear':
            self.Trend = nn.Sequential(
                nn.Linear(self.seq_len, self.pred_len)
            )
        else:
            raise ValueError(f"Invalid model type: {self.type}")

    def _init_eia_weights(self):
        for layer in self.attention_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
      
        if self.revin:
            x = self.revin_layer(x, 'norm')

        seasonal_init, trend_init = self.decomp(x)

        seasonal_output = self.Seasonal(seasonal_init)
        trend_output = self.Trend(trend_init.permute(0, 2, 1)).permute(0, 2, 1)

       
        fusion_weights = self.attention_mlp(torch.cat([seasonal_output, trend_output], dim=-1))  
        x = 2*(fusion_weights * seasonal_output + (1 - fusion_weights) * trend_output)  

        if self.revin:
            x = self.revin_layer(x, 'denorm')

        return  x
