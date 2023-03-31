import numpy as np
import torch
from src.utils import bmmt, bmv, bmtv, bbmv, bmtm
from src.lie_algebra import SO3
import matplotlib.pyplot as plt


class BaseLoss(torch.nn.Module):

    def __init__(self,  dt):
        super().__init__()
        # windows sizes
        self.min_N = 4
        self.max_N = 5
        self.min_train_freq = 2 ** self.min_N
        self.max_train_freq = 2 ** self.max_N
        # sampling time
        self.dt = dt # (s)


class GyroLoss(BaseLoss):
    """Loss for low-frequency orientation increment"""

    def __init__(self, w, dt, target, huber):
        super().__init__(dt)
        # weights on loss
        self.w = w
        self.sl = torch.nn.SmoothL1Loss()
        # self.sl = torch.nn.SmoothL1Loss(reduction='none')
        if target == 'all':
            self.forward = self.forward_with_all
        self.huber = huber
        self.weight = torch.ones(1, 1,
                                 self.min_train_freq)/ self.min_train_freq
        self.N0 = 5 # remove first N0 increment in loss due not account padding

    def f_huber(self, rs):
        """Huber loss function"""
        loss = self.w * self.sl(rs/self.huber, torch.zeros_like(rs)) * (self.huber**2)
        return loss

    def forward_with_all(self, xs, hat_xs):
        """Forward errors with SE2(3)"""


        N = xs.shape[0]

        # IEKF
        xs = xs.reshape(-1, 9).double()
        omegas_xs = xs[:, :3].double()
        dv_xs = xs[:, 3:6].double()
        dp_xs = xs[:, 6:9].double()

        Omegas_Xs = SO3.exp(omegas_xs).double()


        hat_xs = hat_xs.reshape(-1, 15).double()
        hat_omegas = hat_xs[:, :3].double()
        hat_acc = hat_xs[:, 3:6].double()
        hat_dxi = hat_xs[:, 6:9].double()
        hat_dv = hat_xs[:, 9:12].double()
        hat_dp = hat_xs[:, 12:15].double()
        hat_Omegas = SO3.exp(hat_omegas).double()
        hat_Xi = SO3.exp(hat_dxi).double()



        rs1 = 6e0 * SO3.log(bmtm(Omegas_Xs, hat_Omegas)).reshape(N, -1, 3)[:, self.N0:]
        rs2 = 6e0 * (dv_xs - hat_acc).reshape(N, -1, 3)[:, self.N0:]
        rs3 = SO3.log(bmtm(Omegas_Xs, hat_Xi)).reshape(N, -1, 3)[:, self.N0:]
        rs4 = (dv_xs - hat_dv).reshape(N, -1, 3)[:, self.N0:]
        rs5 = (dp_xs - hat_dp).reshape(N, -1, 3)[:, self.N0:]
        rs = torch.cat((rs1, rs2, rs3, rs4, rs5), dim=2)
        rs_mean = rs.mean(dim=0, keepdim=False).mean(dim=0, keepdim=False)
        print('rs_mean =',  rs_mean/self.huber)


        loss = self.f_huber(rs)


        return loss
