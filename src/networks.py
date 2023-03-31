import torch
import matplotlib.pyplot as plt
import numpy as np
from src.utils import bmtm, bmtv, bmmt, bbmv
from src.lie_algebra import SO3


class BaseNet(torch.nn.Module):
    def __init__(self, in_dim, out_dim, c0, dropout, ks, ds, momentum):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # channel dimension
        c1 = 2 * c0  # 2*16=32
        c2 = 2 * c1
        c3 = 2 * c2
        # kernel dimension (odd number)
        k0 = ks[0]  # 7 7 7 7
        k1 = ks[1]
        k2 = ks[2]
        k3 = ks[3]
        # dilation dimension
        d0 = ds[0]  # 4 4 4
        d1 = ds[1]
        d2 = ds[2]
        # padding
        p0 = (k0 - 1) + d0 * (k1 - 1) + d0 * d1 * (k2 - 1) #+ d0 * d1 * d2 * (k3 - 1)
        # nets
        self.cnn = torch.nn.Sequential(
            torch.nn.ReplicationPad1d((p0, 0)),  # padding at start
            torch.nn.Conv1d(in_dim, c0, k0, dilation=1),
            torch.nn.BatchNorm1d(c0, momentum=momentum),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(c0, c1, k1, dilation=d0),
            torch.nn.BatchNorm1d(c1, momentum=momentum),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(c1, c2, k2, dilation=d0 * d1),
            torch.nn.BatchNorm1d(c2, momentum=momentum),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(c2, out_dim, 1, dilation=1),
            torch.nn.ReplicationPad1d((0, 0)),  # no padding at end
        )

        self.lin = torch.nn.Sequential(
            torch.nn.Linear(out_dim, out_dim),
            torch.nn.Tanh(),
        )
        # for normalizing inputs
        self.mean_u = torch.nn.Parameter(torch.zeros(in_dim),
                                         requires_grad=False)
        self.std_u = torch.nn.Parameter(torch.ones(in_dim),
                                        requires_grad=False)

    def forward(self, us):
        u = self.norm(us).transpose(1, 2)
        y_cov = self.cnn(u).transpose(1, 2)
        y = self.lin(y_cov)
        return y

    def norm(self, us):
        return (us - self.mean_u) / self.std_u

    def set_normalized_factors(self, mean_u, std_u):
        self.mean_u = torch.nn.Parameter(torch.as_tensor(mean_u, dtype=torch.float32), requires_grad=False)
        self.std_u = torch.nn.Parameter(torch.as_tensor(std_u, dtype=torch.float32), requires_grad=False)


class GyroNet(BaseNet):
    def __init__(self, in_dim, out_dim, c0, dropout, ks, ds, momentum,
                 gyro_std):
        super().__init__(in_dim, out_dim, c0, dropout, ks, ds, momentum)
        gyro_std = torch.Tensor(gyro_std)
        self.gyro_std = torch.nn.Parameter(gyro_std, requires_grad=False)

        self.Id3 = torch.eye(3)

    def forward(self, us):
        ys_temp = super().forward(us.float())

        # ys = 3 * ys_temp.transpose(1, 2).double()
        ys = 3 * ys_temp.double()

        cali_rate0 = torch.Tensor([1, 1, 1,
                                   1, 1, 1,
                                   0, 0, 0,
                                   0, 0, 0,
                                   2, 20]).double()
        cali_rate0 = cali_rate0.unsqueeze(0)

        cali_rate = torch.zeros(ys.shape[0], ys.shape[1], cali_rate0.shape[1])
        cali_rate[:, :, :6] = cali_rate0[:, :6] * (10 ** (0.01*ys[:, :, :6]))
        cali_rate[:, :, 6:12] = cali_rate0[:, 6:12] + 1e-0 * ys[:, :, 6:12]
        cali_rate[:, :, 12:14] = (cali_rate0[:, 12:14] * (10 ** ys[:, :, 12:14]))


        return cali_rate
