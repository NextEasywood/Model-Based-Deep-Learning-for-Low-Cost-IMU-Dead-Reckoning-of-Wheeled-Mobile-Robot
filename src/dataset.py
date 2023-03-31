from src.utils import pdump, pload, bmtm
from src.lie_algebra import SO3
from termcolor import cprint
from torch.utils.data.dataset import Dataset
# from scipy.interpolate import interp1d
import numpy as np
# import matplotlib.pyplot as plt
# import pickle
import os
import torch
# import sys
import glob
from collections import OrderedDict
from collections import namedtuple
from navpy import lla2ned
import datetime

class BaseDataset(Dataset):

    def __init__(self, predata_dir, train_seqs, val_seqs, test_seqs, mode, N,  dt=0.01):
        super().__init__()
        # where record pre loaded data
        self.predata_dir = predata_dir
        self.path_normalize_factors = os.path.join(predata_dir, 'nf.p')
        self.mode = mode  # train, val or test
        # choose between training, validation or test sequences
        train_seqs, self.sequences = self.get_sequences(train_seqs, val_seqs,
            test_seqs)
        # get and compute value for normalizing inputs
        self.mean_u, self.std_u = self.init_normalize_factors(train_seqs)

        self._train = False
        self._val = False
        # noise density
        self.imu_std = torch.Tensor([8e-5, 1e-3]).double()
        # bias repeatability (without in-run bias stability)
        self.imu_b0 = torch.Tensor([1e-3, 1e-3]).double()
        # IMU sampling time
        self.dt = dt # (s)
        # sequence size during training
        self.N = N # power of 2

        self.uni = torch.distributions.uniform.Uniform(-torch.ones(1), torch.ones(1))
        self.normal = torch.distributions.normal.Normal(torch.Tensor([0.0]), torch.Tensor([1.0]))
        self.gamma = torch.distributions.gamma.Gamma(torch.tensor([1.0]), torch.tensor([1.0]))

    def get_sequences(self, train_seqs, val_seqs, test_seqs):
        """Choose sequence list depending on dataset mode"""
        sequences_dict = {
            'train': train_seqs,
            'val': val_seqs,
            'test': test_seqs,
        }
        return sequences_dict['train'], sequences_dict[self.mode]

    def __getitem__(self, i):
        mondict = self.load_seq(i)
        N_max = mondict['xs'].shape[0]
        if self._train: # random start
            # n0 = torch.randint(0, 100, (1,))
            n0 = torch.randint(0, 40*100, (1, ))
            nend = n0 + self.N
        elif self._val: # end sequence
            # n0 = 0
            n0 = 40 * 100 + self.N
            # nend = N_max
            nend = n0 + self.N
        else:  # full sequence
            n0 = 0
            # nend = n0 + 3*self.N+100
            nend = N_max
            # nend = 25*100
        t = mondict['t'][n0: nend]
        u = mondict['us'][n0: nend]
        x = mondict['xs'][n0: nend]
        p_gt = mondict['p_gt'][n0: nend]
        v_gt = mondict['v_gt'][n0: nend]
        ang_gt = mondict['ang_gt'][n0: nend]
        name = mondict['name']
        return t, u, x, p_gt, v_gt, ang_gt, name

    def __len__(self):
        return len(self.sequences)

    def add_noise(self, u):
        """Add Gaussian noise and bias to input"""

        # noise density
        imu_std = torch.Tensor([1e-3, 1e-2]).double()
        # bias repeatability (without in-run bias stability)
        imu_b0 = torch.Tensor([[1e-5, 1e-5], [2e-2, 5e-1]]).double()


        noise = torch.randn_like(u)
        # noise = self.normal.sample(u.shape).cuda()
        # noise = self.uni.sample(u.shape).cuda()
        # noise = self.normal.sample(u.shape).cuda()
        # noise = self.gamma.sample(u.shape).cuda()
        # noise = noise[:, :, :, 0].clone()
        # noise[:, :, :3] = noise[:, :, :3] * self.imu_std[0]
        # noise[:, :, 3:6] = noise[:, :, 3:6] * self.imu_std[1]
        # noise[:, :, :3] = noise[:, :, :3] * 8e-5 + 0e-2
        # noise[:, :, 3:6] = noise[:, :, 3:6] * 1e-3 + 0e-2
        noise[:, :, :3] = noise[:, :, :3] * imu_std[0]
        noise[:, :, 3:6] = noise[:, :, 3:6] * imu_std[1]


        # bias repeatability (without in run bias stability)
        b0 = self.uni.sample(u[:, 0].shape)
        # b0 = self.normal.sample(u[:, 0].shape).cuda()
        # b0 = self.gamma.sample(u[:, 0].shape).cuda()

        # b0 = self.uni.sample(u.shape).cuda()
        # b0 = self.normal.sample(u.shape).cuda()
        # b0 = self.gamma.sample(u.shape).cuda()

        # b0[:, :, :3] = b0[:, :, :3] * self.imu_b0[0]*0 + 0e-2
        # b0[:, :, 3:6] = b0[:, :, 3:6] * self.imu_b0[1]*0 + 0e-2
        # b0[:, :, :3] = b0[:, :, :3] * 1e-3 + 5e-2
        # b0[:, :, 3:6] = b0[:, :, 3:6] * 1e-3 + 5e-2

        b0[:, :3, :] = b0[:, :3, :] * imu_b0[0, 0] + imu_b0[1, 0]
        b0[:, 3:6, :] = b0[:, 3:6, :] * imu_b0[0, 1] + imu_b0[1, 1]


        u = u + noise + b0.transpose(1, 2)
        # u = u + noise + b0[:, :, :, 0]
        return u

    def init_train(self):
        self._train = True
        self._val = False

    def init_val(self):
        self._train = False
        self._val = True

    def length(self):
        return self._length

    def load_seq(self, i):
        return pload(self.predata_dir, self.sequences[i] + '.p')

    def load_gt(self, i):
        return pload(self.predata_dir, self.sequences[i] + '_gt.p')

    def init_normalize_factors(self, train_seqs):
        if os.path.exists(self.path_normalize_factors):
            mondict = pload(self.path_normalize_factors)
            return mondict['mean_u'], mondict['std_u']

        path = os.path.join(self.predata_dir, train_seqs[0] + '.p')
        if not os.path.exists(path):
            print("init_normalize_factors not computed")
            return 0, 0

        print('Start computing normalizing factors ...')
        cprint("Do it only on training sequences, it is vital!", 'yellow')
        # first compute mean
        num_data = 0

        for i, sequence in enumerate(train_seqs):
            pickle_dict = pload(self.predata_dir, sequence + '.p')
            us = pickle_dict['us']
            sms = pickle_dict['xs']
            if i == 0:
                mean_u = us.sum(dim=0)
                num_positive = sms.sum(dim=0)
                num_negative = sms.shape[0] - sms.sum(dim=0)
            else:
                mean_u += us.sum(dim=0)
                num_positive += sms.sum(dim=0)
                num_negative += sms.shape[0] - sms.sum(dim=0)
            num_data += us.shape[0]
        mean_u = mean_u / num_data
        pos_weight = num_negative / num_positive

        # second compute standard deviation
        for i, sequence in enumerate(train_seqs):
            pickle_dict = pload(self.predata_dir, sequence + '.p')
            us = pickle_dict['us']
            if i == 0:
                std_u = ((us - mean_u) ** 2).sum(dim=0)
            else:
                std_u += ((us - mean_u) ** 2).sum(dim=0)
        std_u = (std_u / num_data).sqrt()
        normalize_factors = {
            'mean_u': mean_u,
            'std_u': std_u,
        }
        print('... ended computing normalizing factors')
        print('pos_weight:', pos_weight)
        print('This values most be a training parameters !')
        print('mean_u    :', mean_u)
        print('std_u     :', std_u)
        print('num_data  :', num_data)
        pdump(normalize_factors, self.path_normalize_factors)
        return mean_u, std_u

    def read_data(self, data_dir):
        raise NotImplementedError

    @staticmethod
    def interpolate(x, t, t_int):
            """
            Interpolate ground truth at the sensor timestamps
            """

            # vector interpolation
            x_int = np.zeros((t_int.shape[0], x.shape[1]))
            for i in range(x.shape[1]):
                if i in [4, 5, 6, 7]:
                    continue
                x_int[:, i] = np.interp(t_int, t, x[:, i])
            # quaternion interpolation
            t_int = torch.Tensor(t_int - t[0])
            t = torch.Tensor(t - t[0])
            qs = SO3.qnorm(torch.Tensor(x[:, 4:8]))
            x_int[:, 4:8] = SO3.qinterp(qs, t, t_int).numpy()
            return x_int


class KITTIDataset(BaseDataset):
    """
        Dataloader for the KITTI Data Set.
    """
    OxtsPacket = namedtuple('OxtsPacket',
                            'lat, lon, alt, ' + 'roll, pitch, yaw, ' + 'vn, ve, vf, vl, vu, '
                                                                       '' + 'ax, ay, az, af, al, '
                                                                            'au, ' + 'wx, wy, wz, '
                                                                                     'wf, wl, wu, '
                                                                                     '' +
                            'pos_accuracy, vel_accuracy, ' + 'navstat, numsats, ' + 'posmode, '
                                                                                  'velmode, '
                                                                                  'orimode')

    # Bundle into an easy-to-access structure
    OxtsData = namedtuple('OxtsData', 'packet, T_w_imu')
    min_seq_dim = 80 * 100  # 25 s

    def __init__(self, data_dir, predata_dir, train_seqs, val_seqs,
                test_seqs, mode, N,  dt=0.01):
        super().__init__(predata_dir, train_seqs, val_seqs, test_seqs, mode, N,  dt)
        # convert raw data to pre loaded data
        self.read_data(data_dir)

    def read_data(self, data_dir):

        f = os.path.join(self.predata_dir, '2011_09_26_drive_0022_extract.p')
        if True and os.path.exists(f):
            return

        print("Start read_data")
        t_tot = 0  # sum of times for the all dataset
        date_dirs = os.listdir(data_dir)
        for n_iter, date_dir in enumerate(date_dirs):
            # get access to each sequence
            path1 = os.path.join(data_dir, date_dir)
            if not os.path.isdir(path1):
                continue
            date_dirs2 = os.listdir(path1)

            for date_dir2 in date_dirs2:
                path2 = os.path.join(path1, date_dir2)
                if not os.path.isdir(path2):
                    continue
                # read data
                oxts_files = sorted(glob.glob(os.path.join(path2, 'oxts', 'data', '*.txt')))
                oxts = self.load_oxts_packets_and_poses(oxts_files)

                print("\n Sequence name : " + date_dir2)
                if len(oxts) < KITTIDataset.min_seq_dim:  #  sequence shorter than 30 s are rejected
                    cprint("Dataset is too short ({:.2f} s)".format(len(oxts) / 100), 'yellow')
                    continue
                lat_oxts = np.zeros(len(oxts))
                lon_oxts = np.zeros(len(oxts))
                alt_oxts = np.zeros(len(oxts))
                roll_oxts = np.zeros(len(oxts))
                pitch_oxts = np.zeros(len(oxts))
                yaw_oxts = np.zeros(len(oxts))
                roll_gt = np.zeros(len(oxts))
                pitch_gt = np.zeros(len(oxts))
                yaw_gt = np.zeros(len(oxts))
                t = self.load_timestamps(path2)
                acc = np.zeros((len(oxts), 3))
                acc_bis = np.zeros((len(oxts), 3))
                gyro = np.zeros((len(oxts), 3))
                gyro_bis = np.zeros((len(oxts), 3))
                p_gt = np.zeros((len(oxts), 3))
                v_gt = np.zeros((len(oxts), 3))
                v_rob_gt = np.zeros((len(oxts), 3))

                k_max = len(oxts)
                for k in range(k_max):
                    oxts_k = oxts[k]
                    t[k] = 3600 * t[k].hour + 60 * t[k].minute + t[k].second + t[k].microsecond / 1e6
                    lat_oxts[k] = oxts_k[0].lat
                    lon_oxts[k] = oxts_k[0].lon
                    alt_oxts[k] = oxts_k[0].alt
                    acc[k, 0] = oxts_k[0].af
                    acc[k, 1] = oxts_k[0].al
                    acc[k, 2] = oxts_k[0].au
                    acc_bis[k, 0] = oxts_k[0].ax
                    acc_bis[k, 1] = oxts_k[0].ay
                    acc_bis[k, 2] = oxts_k[0].az
                    gyro[k, 0] = oxts_k[0].wf
                    gyro[k, 1] = oxts_k[0].wl
                    gyro[k, 2] = oxts_k[0].wu
                    gyro_bis[k, 0] = oxts_k[0].wx
                    gyro_bis[k, 1] = oxts_k[0].wy
                    gyro_bis[k, 2] = oxts_k[0].wz
                    roll_oxts[k] = oxts_k[0].roll
                    pitch_oxts[k] = oxts_k[0].pitch
                    yaw_oxts[k] = oxts_k[0].yaw
                    v_gt[k, 0] = oxts_k[0].ve
                    v_gt[k, 1] = oxts_k[0].vn
                    v_gt[k, 2] = oxts_k[0].vu
                    v_rob_gt[k, 0] = oxts_k[0].vf
                    v_rob_gt[k, 1] = oxts_k[0].vl
                    v_rob_gt[k, 2] = oxts_k[0].vu
                    p_gt[k] = oxts_k[1][:3, 3]
                    Rot_gt_k = oxts_k[1][:3, :3]
                    roll_gt[k], pitch_gt[k], yaw_gt[k] = KITTIDataset.to_rpy(Rot_gt_k)

                t0 = t[0]
                np_array_t = np.array(t)
                t = np_array_t - t0
                if np.max(t[:-1] - t[1:]) > 0.1:
                    cprint(date_dir2 + " has time problem", 'yellow')
                ang_gt = np.zeros((roll_gt.shape[0], 3))
                ang_gt[:, 0] = roll_gt
                ang_gt[:, 1] = pitch_gt
                ang_gt[:, 2] = yaw_gt

                p_oxts = lla2ned(lat_oxts, lon_oxts, alt_oxts, lat_oxts[0], lon_oxts[0],
                                 alt_oxts[0], latlon_unit='deg', alt_unit='m', model='wgs84')
                p_oxts[:, [0, 1]] = p_oxts[:, [1, 0]]  # see note

                imu = np.concatenate((gyro_bis, acc_bis), -1)

                t = torch.from_numpy(t)
                p_gt = torch.from_numpy(p_gt)
                v_gt = torch.from_numpy(v_gt)
                ang_gt = torch.from_numpy(ang_gt)
                imu = torch.from_numpy(imu)


                Rot_gt = SO3.from_rpy(ang_gt[:, 0], ang_gt[:, 1], ang_gt[:, 2])
                dRot_ij = bmtm(Rot_gt[:-1], Rot_gt[1:])
                dRot_ij = SO3.dnormalize(dRot_ij.cuda())
                dxi_ij = SO3.log(dRot_ij).cpu()

                dv_ij = v_gt[1:] - v_gt[:-1]
                dp_ij = p_gt[1:] - p_gt[:-1]

                xs = np.concatenate((dxi_ij, dv_ij, dp_ij), -1)

                xs = torch.from_numpy(xs)


                xs = xs


                mondict = {
                    't': t,
                    'xs': xs,
                    'us': imu,
                    'p_gt': p_gt,
                    'ang_gt': ang_gt,
                    'v_gt': v_gt,
                    'name': date_dir2,
                    't0': t0
                    }

                t_tot += t[-1] - t[0]
                pdump(mondict, self.predata_dir, date_dir2 + ".p")

        print("\n Total dataset duration : {:.2f} s".format(t_tot))


    @staticmethod
    def pose_from_oxts_packet(packet, scale):
        """Helper method to compute a SE(3) pose matrix from an OXTS packet.
        """
        er = 6378137.  # earth radius (approx.) in meters

        # Use a Mercator projection to get the translation vector
        tx = scale * packet.lon * np.pi * er / 180.
        ty = scale * er * np.log(np.tan((90. + packet.lat) * np.pi / 360.))
        tz = packet.alt
        t = np.array([tx, ty, tz])

        # Use the Euler angles to get the rotation matrix
        Rx = KITTIDataset.rotx(packet.roll)
        Ry = KITTIDataset.roty(packet.pitch)
        Rz = KITTIDataset.rotz(packet.yaw)
        R = Rz.dot(Ry.dot(Rx))

        # Combine the translation and rotation into a homogeneous transform
        return R, t


    @staticmethod
    def transform_from_rot_trans(R, t):
        """Transformation matrix from rotation matrix and translation vector."""
        R = R.reshape(3, 3)
        t = t.reshape(3, 1)
        return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


    @staticmethod
    def load_oxts_packets_and_poses(oxts_files):
        """Generator to read OXTS ground truth data.
           Poses are given in an East-North-Up coordinate system
           whose origin is the first GPS position.
        """
        # Scale for Mercator projection (from first lat value)
        scale = None
        # Origin of the global coordinate system (first GPS position)
        origin = None

        oxts = []

        for filename in oxts_files:
            with open(filename, 'r') as f:
                for line in f.readlines():
                    line = line.split()
                    # Last five entries are flags and counts
                    line[:-5] = [float(x) for x in line[:-5]]
                    line[-5:] = [int(float(x)) for x in line[-5:]]

                    packet = KITTIDataset.OxtsPacket(*line)

                    if scale is None:
                        scale = np.cos(packet.lat * np.pi / 180.)

                    R, t = KITTIDataset.pose_from_oxts_packet(packet, scale)

                    if origin is None:
                        origin = t

                    T_w_imu = KITTIDataset.transform_from_rot_trans(R, t - origin)

                    oxts.append(KITTIDataset.OxtsData(packet, T_w_imu))
        return oxts


    @staticmethod
    def load_timestamps(data_path):
        """Load timestamps from file."""
        timestamp_file = os.path.join(data_path, 'oxts', 'timestamps.txt')

        # Read and parse the timestamps
        timestamps = []
        with open(timestamp_file, 'r') as f:
            for line in f.readlines():
                # NB: datetime only supports microseconds, but KITTI timestamps
                # give nanoseconds, so need to truncate last 4 characters to
                # get rid of \n (counts as 1) and extra 3 digits
                t = datetime.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
                timestamps.append(t)
        return timestamps


    @staticmethod
    def rotx(t):
        """Rotation about the x-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    @staticmethod
    def roty(t):
        """Rotation about the y-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    @staticmethod
    def rotz(t):
        """Rotation about the z-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    @staticmethod
    def to_rpy(Rot):
        pitch = np.arctan2(-Rot[2, 0], np.sqrt(Rot[0, 0]**2 + Rot[1, 0]**2))

        if np.isclose(pitch, np.pi / 2.):
            yaw = 0.
            roll = np.arctan2(Rot[0, 1], Rot[1, 1])
        elif np.isclose(pitch, -np.pi / 2.):
            yaw = 0.
            roll = -np.arctan2(Rot[0, 1], Rot[1, 1])
        else:
            sec_pitch = 1. / np.cos(pitch)
            yaw = np.arctan2(Rot[1, 0] * sec_pitch,
                             Rot[0, 0] * sec_pitch)
            roll = np.arctan2(Rot[2, 1] * sec_pitch,
                              Rot[2, 2] * sec_pitch)
        return roll, pitch, yaw

