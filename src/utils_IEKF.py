import torch
from src.lie_algebra import SO3
import time
from src.utils import *
# torch.set_default_tensor_type(torch.cuda.FloatTensor)

class InitProcessCovNet(torch.nn.Module):

    def __init__(self):
        super(InitProcessCovNet, self).__init__()

        self.beta_process = 3 * torch.ones(2).double()
        self.beta_initialization = 3 * torch.ones(2).double()

        self.factor_initial_covariance = torch.nn.Linear(1, 6, bias=False).double()
        """parameters for initializing covariance"""
        self.factor_initial_covariance.weight.data[:] /= 10

        self.factor_process_covariance = torch.nn.Linear(1, 6, bias=False).double()
        """parameters for process noise covariance"""
        self.factor_process_covariance.weight.data[:] /= 10
        self.tanh = torch.nn.Tanh()

    def forward(self, iekf):
        return

    def init_cov(self):
        alpha = self.factor_initial_covariance(torch.ones(1).double()).squeeze()
        beta = 10 ** (self.tanh(alpha))
        return beta

    def init_processcov(self):
        alpha = self.factor_process_covariance(torch.ones(1).double())
        beta = 10 ** (self.tanh(alpha))
        return beta


class IEKF:
    Id1 = torch.eye(1).double()
    Id2 = torch.eye(2).double()
    Id3 = torch.eye(3).double()
    Id6 = torch.eye(6).double()
    IdP = torch.eye(21).double()

    def __init__(self):

        self.initprocesscov_net = InitProcessCovNet()

        self.g = torch.Tensor([0, 0, -9.80665])
        """gravity vector"""
        self.P_dim = 21
        """covariance dimension"""
        self.Q_dim = 18
        """process noise covariance dimension"""
        # Process noise covariance
        self.cov_omega = 2e-4
        """gyro covariance"""
        self.cov_acc = 1e-3
        """accelerometer covariance"""
        self.cov_b_omega = 1e-8
        """gyro bias covariance"""
        self.cov_b_acc = 1e-6
        """accelerometer bias covariance"""
        self.cov_Rot_c_i = 1e-8
        """car to IMU orientation covariance"""
        self.cov_t_c_i = 1e-8
        """car to IMU translation covariance"""
        self.cov_lat = 1
        """Zero lateral velocity covariance"""
        self.cov_up = 10
        """Zero lateral velocity covariance"""
        self.cov_Rot0 = 1e-6
        """initial pitch and roll covariance"""
        self.cov_b_omega0 = 1e-8
        """initial gyro bias covariance"""
        self.cov_b_acc0 = 1e-3
        """initial accelerometer bias covariance"""
        self.cov_v0 = 1e-1
        """initial velocity covariance"""
        self.cov_Rot_c_i0 = 1e-5
        """initial car to IMU pitch and roll covariance"""
        self.cov_t_c_i0 = 1e-2
        """initial car to IMU translation covariance"""
        self.Q = torch.diag(torch.Tensor([self.cov_omega, self.cov_omega, self.cov_omega,
                                          self.cov_acc, self.cov_acc, self.cov_acc,
                                          self.cov_b_omega, self.cov_b_omega, self.cov_b_omega,
                                          self.cov_b_acc, self.cov_b_acc, self.cov_b_acc,
                                          self.cov_Rot_c_i, self.cov_Rot_c_i, self.cov_Rot_c_i,
                                          self.cov_t_c_i, self.cov_t_c_i, self.cov_t_c_i])
                            )
        self.cov0_measurement = torch.Tensor([self.cov_lat, self.cov_up])

    def set_Q(self):
        """
        Update the process noise covariance
        :return:
        """

        self.Q = torch.diag(torch.Tensor([self.cov_omega, self.cov_omega, self. cov_omega,
                                           self.cov_acc, self.cov_acc, self.cov_acc,
                                           self.cov_b_omega, self.cov_b_omega, self.cov_b_omega,
                                           self.cov_b_acc, self.cov_b_acc, self.cov_b_acc,
                                           self.cov_Rot_c_i, self.cov_Rot_c_i, self.cov_Rot_c_i,
                                           self.cov_t_c_i, self.cov_t_c_i, self.cov_t_c_i])
                            ).double()
        self.Q = self.Q
        # self = self

        beta = self.initprocesscov_net.init_processcov()
        beta = beta
        self.Q = torch.zeros(self.Q.shape[0], self.Q.shape[0]).double()

        self.Q = self.Q

        # self.cov_omega = torch.from_numpy(np.array(self.cov_omega))
        # self.Id3 = torch.from_numpy(np.array(self.Id3))

        self.Q[:3, :3] = self.cov_omega*beta[0]*self.Id3
        self.Q[3:6, 3:6] = self.cov_acc*beta[1]*self.Id3
        self.Q[6:9, 6:9] = self.cov_b_omega*beta[2]*self.Id3
        self.Q[9:12, 9:12] = self.cov_b_acc*beta[3]*self.Id3
        self.Q[12:15, 12:15] = self.cov_Rot_c_i*beta[4]*self.Id3
        self.Q[15:18, 15:18] = self.cov_t_c_i*beta[5]*self.Id3

    def run(self, t, u, measurements_covs, v_mes, p_mes, N, ang0):


        dt = t[:,1:] - t[:,:-1] # (s)
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P = self.init_run(dt, u, p_mes, v_mes,
                                                                     N, ang0)
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P \
            = Rot.double(), v.double(), p.double(), b_omega.double(), b_acc.double(), \
              Rot_c_i.double(), t_c_i.double(), P.double()


        dt = dt.double()
        for i in range(1, N):

            Rot_i, v_i, p_i, b_omega_i, b_acc_i, Rot_c_i_i, t_c_i_i, P_i = \
                self.propagate(Rot[:, i - 1], v[:, i - 1], p[:, i - 1], b_omega[:, i - 1], b_acc[:, i - 1], Rot_c_i[:, i - 1],
                               t_c_i[:, i - 1], P, u[:, i], dt[:, i - 1])
            torch.cuda.empty_cache()
            Rot[:, i], v[:, i], p[:, i], b_omega[:, i], b_acc[:, i], Rot_c_i[:, i], t_c_i[:, i], P = \
                self.update(Rot_i, v_i, p_i, b_omega_i, b_acc_i, Rot_c_i_i, t_c_i_i, P_i,
                            u[:, i], i, measurements_covs[:, i, :])
            torch.cuda.empty_cache()

        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i

    def init_run(self, dt, u, p_mes, v_mes, N, ang0):
        N0 = u.size(0)

        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = \
            self.init_saved_state(dt, N, N0, ang0)
        Rot[:, 0] = SO3.from_rpy(ang0[:, 0], ang0[:, 1], ang0[:, 2])
        v[:, 0, :] = v_mes[:, 0, :].double()
        P = self.init_covariance(N0)
        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P

    def init_covariance(self, N0):
        beta = self.initprocesscov_net.init_cov()
        P = torch.zeros(N0, self.P_dim, self.P_dim).double()
        P[:, :2, :2] = self.cov_Rot0*beta[0]*self.Id2  # no yaw error
        P[:, 3:5, 3:5] = self.cov_v0*beta[1]*self.Id2
        P[:, 9:12, 9:12] = self.cov_b_omega0*beta[2]*self.Id3
        P[:, 12:15, 12:15] = self.cov_b_acc0*beta[3]*self.Id3
        P[:, 15:18, 15:18] = self.cov_Rot_c_i0*beta[4]*self.Id3
        P[:, 18:21, 18:21] = self.cov_t_c_i0*beta[5]*self.Id3
        return P

    def init_saved_state(self, dt, N, N0, ang0):
        Rot = dt.new_zeros(N0, N, 3, 3).double()
        v = dt.new_zeros(N0, N, 3).double()
        p = dt.new_zeros(N0, N, 3).double()
        b_omega = dt.new_zeros(N0, N, 3).double()
        b_acc = dt.new_zeros(N0, N, 3).double()
        Rot_c_i = dt.new_zeros(N0, N, 3, 3).double()
        t_c_i = dt.new_zeros(N0, N, 3).double()
        Rot_c_i[:, 0] = self.Id3
        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i

    def propagate(self, Rot_prev, v_prev, p_prev, b_omega_prev, b_acc_prev, Rot_c_i_prev, t_c_i_prev,
                  P_prev, u, dt):
        Rot_prev = Rot_prev.clone().double()
        dt = dt.clone().double()
        acc_b = u[:, 3:6].clone() - b_acc_prev.clone()
        # acc = Rot_prev.mv(acc_b) + self.g
        acc = (bmv(Rot_prev, acc_b) + self.g).double()
        v = v_prev.clone() + torch.einsum('ij, i -> ij', acc, dt).double()

        p = p_prev.clone() \
            + torch.einsum('ij, i -> ij', v_prev.clone(), dt) \
            + 1 / 2 * torch.einsum('ij, i -> ij', acc, dt ** 2).double()

        omega = torch.einsum('ij, i -> ij', (u[:, :3].clone() - b_omega_prev.clone()), dt).double()
        Omega = SO3.exp(omega).double()
        Rot = Rot_prev.bmm(Omega).double()

        b_omega = b_omega_prev.clone().double()
        b_acc = b_acc_prev.clone().double()
        Rot_c_i = Rot_c_i_prev.clone().double()
        t_c_i = t_c_i_prev.clone().double()

        P = self.propagate_cov(P_prev, Rot_prev, v_prev, p_prev, b_omega_prev, b_acc_prev,
                               u, dt)
        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P

    def propagate_cov(self, P, Rot_prev, v_prev, p_prev, b_omega_prev, b_acc_prev, u,
                      dt):
        N0 = u.size(0)
        F = P.new_zeros(N0, self.P_dim, self.P_dim).double()
        G = P.new_zeros(N0, self.P_dim, self.Q.shape[0]).double()
        Q = self.Q.clone().double()
        F[:, 3:6, :3] = self.skew(self.g)
        F[:, 6:9, 3:6] = self.Id3
        G[:, 3:6, 3:6] = Rot_prev
        F[:, 3:6, 12:15] = -Rot_prev
        v_skew_rot = self.bskew(v_prev).bmm(Rot_prev)
        p_skew_rot = self.bskew(p_prev).bmm(Rot_prev)
        G[:, :3, :3] = Rot_prev
        G[:, 3:6, :3] = v_skew_rot
        G[:, 6:9, :3] = p_skew_rot
        F[:, :3, 9:12] = -Rot_prev
        F[:, 3:6, 9:12] = -v_skew_rot
        F[:, 6:9, 9:12] = -p_skew_rot
        G[:, 9:12, 6:9] = self.Id3
        G[:, 12:15, 9:12] = self.Id3
        G[:, 15:18, 12:15] = self.Id3
        G[:, 18:21, 15:18] = self.Id3

        F = torch.einsum('bij, b -> bij', F, dt)
        G = torch.einsum('bij, b -> bij', G, dt)
        F_square = F.bmm(F)
        F_cube = F_square.bmm(F)
        Phi = self.IdP + F + 1 / 2 * F_square + 1 / 6 * F_cube
        mm_GQ = torch.einsum('bij, jk -> bik', G, Q)
        P_GQGT = P + bmmt(mm_GQ, G)
        P_new = bmmt(Phi.bmm(P_GQGT), Phi)
        return P_new

    def update(self, Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, u, i, measurement_cov):
        # # orientation of body frame
        # Rot_body = Rot.bmm(Rot_c_i).double()
        # # velocity in imu frame
        # v_imu = bmtv(Rot, v).double()
        # omega = (u[:, :3] - b_omega).double()
        # # velocity in body frame
        # v_body = bmtv(Rot_c_i, v_imu).double() + bmv(self.bskew(t_c_i), omega).double()
        # # v_body = Rot_c_i.t().mv(v_imu) + self.bskew(t_c_i).mv(omega)
        # Omega = self.bskew(omega).double()
        # # Jacobian in car frame
        # H_v_imu = bmtm(Rot_c_i, self.bskew(v_imu)).double()
        # H_t_c_i = self.bskew(t_c_i).double()

        # N0 = u.shape[0]
        # H = P.new_zeros(N0, 2, self.P_dim).double()
        # H[:, :, 3:6] = Rot_body.transpose(1, 2)[:, 1:]
        # H[:, :, 15:18] = H_v_imu[:, 1:]
        # H[:, :, 9:12] = H_t_c_i[:, 1:]
        # H[:, :, 18:21] = -Omega[:, 1:]
        # r = - v_body[:, 1:]
        # R = self.bdiag(measurement_cov)
        Rot_body = Rot.bmm(Rot_c_i).double()

        v_imu = bmtv(Rot, v).double()
        omega = (u[:, :3] - b_omega).double()

        Omega = self.bskew(omega).double()
        v_body = bmtv(Rot_c_i, v_imu + bmv(Omega, t_c_i).double()).double()

        # Jacobian in car frame
        H_v_imu = bmtm(Rot_c_i, self.bskew(v_imu + bmv(Omega, t_c_i))).double()
        H_v_imu = H_v_imu.bmm(Rot_c_i)
        H_t_c_i = bmtm(Rot_c_i,self.bskew(t_c_i)).double()

        H_R_imu1 = bmtm(Rot, self.bskew(v)).double()
        H_R_imu = bmtm(Rot_c_i, H_R_imu1).double()
        H_R_imu = H_R_imu.bmm(Rot)

        N0 = u.shape[0]
        H = P.new_zeros(N0, 2, self.P_dim).double()
        H[:, :, :3] = H_R_imu[:,1:]
        H[:, :, 3:6] = Rot_body.transpose(1, 2)[:, 1:]
        H[:, :, 15:18] = H_v_imu[:, 1:]
        H[:, :, 9:12] = H_t_c_i[:, 1:]
        H[:, :, 18:21] = -Omega[:, 1:]
        r = - v_body[:, 1:]
        R = self.bdiag(measurement_cov)
        
        Rot_up, v_up, p_up, b_omega_up, b_acc_up, Rot_c_i_up, t_c_i_up, P_up = \
            self.state_and_cov_update(Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, H, r, R)
        return Rot_up, v_up, p_up, b_omega_up, b_acc_up, Rot_c_i_up, t_c_i_up, P_up

    @staticmethod
    def state_and_cov_update(Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, H, r, R):
        H_t = H.transpose(1, 2).double()
        S = (H.bmm(P).bmm(H_t) + R).double()

        Kt = torch.linalg.solve(S, P.bmm(H_t).transpose(1, 2)).double()
        K = Kt.transpose(1, 2).double()
        dx = bmv(K, r).double()

        dR = torch.zeros(v.shape[0], 3, 3).double()
        dxi = torch.zeros(v.shape[0], 3, 2).double()
        for i in range(0, v.shape[0]):
            dR[i], dxi[i] = IEKF.sen3exp(dx[i, :9])
        dv = dxi[:, :, 0].double()
        dp = dxi[:, :, 1].double()
        Rot_up = dR.bmm(Rot).double()
        v_up = (bmv(dR, v) + dv).double()
        p_up = (bmv(dR, p) + dp).double()

        b_omega_up = b_omega + dx[:, 9:12]
        b_acc_up = b_acc + dx[:, 12:15]

        dR = SO3.exp(dx[:, 15:18]).double()
        Rot_c_i_up = dR.bmm(Rot_c_i).double()
        t_c_i_up = t_c_i + dx[:, 18:21]

        I_KH = IEKF.IdP.double() - K.bmm(H).double()
        P_upprev = I_KH.bmm(P).bmm(I_KH.transpose(1, 2)) + K.bmm(R).bmm(Kt).double()
        P_up = (P_upprev + P_upprev.transpose(1, 2)).double() / 2
        return Rot_up, v_up, p_up, b_omega_up, b_acc_up, Rot_c_i_up, t_c_i_up, P_up

    @staticmethod
    def skew(x):
        X = torch.Tensor([[0, -x[2], x[1]],
                          [x[2], 0, -x[0]],
                          [-x[1], x[0], 0]])
        return X

    @staticmethod
    def bskew(bx):
        X = torch.zeros(bx.shape[0], 3, 3).double()
        for i in range(0, bx.shape[0]):
            x = bx[i]
            X[i] = torch.Tensor([[0, -x[2], x[1]],
                              [x[2], 0, -x[0]],
                              [-x[1], x[0], 0]])
        return X

    @staticmethod
    def bdiag(bx):
        X = torch.zeros(bx.shape[0], bx.shape[1], bx.shape[1]).double()
        for i in range(0, bx.shape[0]):
            x = bx[i].double()
            X[i] = torch.diag(x).double()
        return X
    @staticmethod
    def sen3exp(xi):
        phi = xi[:3]
        angle = torch.norm(phi)

        # Near |phi|==0, use first order Taylor expansion
        if isclose(angle, 0.):
            skew_phi = torch.Tensor([[0, -phi[2], phi[1]],
                          [phi[2], 0, -phi[0]],
                          [-phi[1], phi[0], 0]]).double()
            J = IEKF.Id3 + 0.5 * skew_phi
            Rot = IEKF.Id3 + skew_phi
        else:
            axis = phi / angle
            skew_axis = torch.Tensor([[0, -axis[2], axis[1]],
                              [axis[2], 0, -axis[0]],
                              [-axis[1], axis[0], 0]]).double()
            s = torch.sin(angle)
            c = torch.cos(angle)

            J = (s / angle) * IEKF.Id3 + (1 - s / angle) * IEKF.outer(axis, axis)\
                   + ((1 - c) / angle) * skew_axis
            Rot = c * IEKF.Id3 + (1 - c) * IEKF.outer(axis, axis) \
                 + s * skew_axis

        x = J.mm(xi[3:].view(-1, 3).t())
        return Rot, x

    @staticmethod
    def so3exp(phi):
        angle = phi.norm()

        # Near phi==0, use first order Taylor expansion
        if isclose(angle, 0.):
            skew_phi = torch.Tensor([[0, -phi[2], phi[1]],
                          [phi[2], 0, -phi[0]],
                          [-phi[1], phi[0], 0]])
            Xi = IEKF.Id3 + skew_phi
            return Xi
        axis = phi / angle
        skew_axis = torch.Tensor([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
        c = angle.cos()
        s = angle.sin()
        Xi = c * IEKF.Id3 + (1 - c) * IEKF.outer(axis, axis) \
             + s * skew_axis
        return Xi.double()

    @staticmethod
    def outer(a, b):
        ab = a.view(-1, 1)*b.view(1, -1)
        return ab


def isclose(mat1, mat2, tol=1e-10):
    return (mat1 - mat2).abs().lt(tol)
