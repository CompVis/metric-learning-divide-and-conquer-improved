
from torch import nn
from torch.nn import init
import torch
import scipy.stats as stats

def init_weights(m, stddev=0.1):
    if type(m) == nn.Conv2d or type(m) == nn.Linear or type(m) == nn.Conv1d:
        X = stats.truncnorm(-2, 2, scale=stddev)
        w_values = torch.Tensor(X.rvs(m.weight.data.numel()))
        w_values = w_values.view(m.weight.data.size())
        m.weight.data.copy_(w_values)

def init_bias(m, const=0.1):
    if type(m) == nn.Parameter:
        init.constant_(m, const)
    elif type(m) == nn.Conv2d or type(m) == nn.Linear or type(m) == nn.Conv1d:
        init.constant_(m.bias, const)

