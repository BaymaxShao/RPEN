import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def unity_quaternion_to_logq(q):
    u = q[:, -1]
    v = q[:, :-1]
    norm = torch.norm(v, dim=1, keepdim=True).clamp(min=1e-8)
    out = v * (torch.acos(torch.clamp(u, min=-1.0, max=1.0)).reshape(-1, 1) / norm)
    return out


class QuatLoss(nn.Module):
    def __init__(self, criterion=nn.L1Loss()):
        super(QuatLoss, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor([-3]).cuda(), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor([0]).cuda(), requires_grad=True)
        self.criterion = criterion
        self.rot_criterion = nn.L1Loss()

    def forward(self, q_hat, q):
        t = q[:, 4:]
        log_q = unity_quaternion_to_logq(q[:, :4])
        t_hat = q_hat[:, :3]
        log_q_hat = q_hat[:, 3:]

        loss = self.criterion(t_hat, t) * torch.exp(-self.beta) + self.beta + self.rot_criterion(log_q_hat, log_q) * torch.exp(-self.gamma) + self.gamma
        if loss > 100:
            raise ValueError('Unexpected Loss')
        return loss




