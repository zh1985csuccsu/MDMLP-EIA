import torch
from torch import nn

from layers.ema import EMA
from layers.dema import DEMA

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class DECOMP(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, ma_type, alpha, beta, kernel_size=25):
        super(DECOMP, self).__init__()
        if ma_type == 'ema':
            self.ma = EMA(alpha)
        elif ma_type == 'dema':
            self.ma = DEMA(alpha, beta)
        elif ma_type == 'ma':
            self.ma = moving_avg(kernel_size, stride=1)
        else:
            raise ValueError(f"DECOMP: ma_type must be one of ['ema', 'dema', 'ma'], got '{ma_type}'")

    def forward(self, x):
        moving_average = self.ma(x)
        res = x - moving_average
        return res, moving_average