import torch
import torch.nn as nn
import sys
from pathlib import Path

TRAIN_DIR = Path("__file__").absolute().parent
sys.path.append(str(TRAIN_DIR))
from .koopman_net import KoopmanNet
import numpy as np


class KoopmanNetInvCtrl(KoopmanNet):
    def __init__(self, net_params, standardizer_x=None, standardizer_u=None):
        super(KoopmanNetInvCtrl, self).__init__(
            net_params, standardizer_x=standardizer_x, standardizer_u=standardizer_u
        )
        self.inn_config = self.net_params["inn_config"]
        self.inn_layers = self.net_params["inn_layers"]
        encoder_output_dim = self.net_params["encoder_output_dim"]
        first_obs_const = int(self.net_params["first_obs_const"])
        n_fixed_states = self.net_params["n_fixed_states"]
        self.n_tot = int(first_obs_const) + n_fixed_states + encoder_output_dim

        N_inn = self.inn_config[0]
        N_changed_elements = self.inn_config[1]
        masks = np.ones((N_inn, self.n_tot))
        indices = np.random.choice(masks.size, N_changed_elements, replace=False)
        masks.flat[indices] = 0
        self.masks = nn.ParameterList(
            [nn.Parameter(torch.Tensor(m), requires_grad=False) for m in masks]
        ).to(self.device)

        self.z0_MC = None
        self.z0 = None
        self.Q_z0 = None
        self.ctrl_MC = None
        self.eta_z0s = 0.01
        self.eta_ctrl = 0.01

    def construct_net(self):
        n = self.net_params["state_dim"]
        m = self.net_params["ctrl_dim"]
        encoder_output_dim = self.net_params["encoder_output_dim"]
        first_obs_const = int(self.net_params["first_obs_const"])
        n_fixed_states = self.net_params["n_fixed_states"]
        self.construct_encoder_()

        # used for linear INN comparison experiments
        from train.linear_INNU import Affine_Coupling

        AFF_s = []
        for i in range(len(self.masks)):
            mask = nn.Parameter(self.masks[i], requires_grad=False)
            fix_idx = torch.nonzero(mask).squeeze()
            change_idx = torch.nonzero(1 - mask).squeeze()
            fix_num = fix_idx.numel()
            change_num = change_idx.numel()
            INN_layers = {
                "tnsl_layers": [change_num]
                + self.inn_layers["tnsl_layers"]
                + [fix_num],
                "cmpd_layers": [fix_num]
                + self.inn_layers["cmpd_layers"]
                + [change_num],
            }
            AFF_s.append(Affine_Coupling(mask, INN_layers))
        self.affine_couplings = nn.ModuleList(AFF_s).to(self.device)

        # used for INN with ReLU comparison experiments
        # from train.INNU import Affine_Coupling
        # net_inn_layers={}
        # net_inn_layers["tnsl_layers"] = [self.n_tot] + self.inn_layers["tnsl_layers"] + [self.n_tot]
        # net_inn_layers["cmpd_layers"] = [self.n_tot] + self.inn_layers["cmpd_layers"] + [self.n_tot]
        # self.affine_couplings = nn.ModuleList(Affine_Coupling(self.masks[i], net_inn_layers) for i in range(len(self.masks))).to(self.device)

        if self.net_params["override_kinematics"]:
            self.koopman_fc_act = nn.Linear(
                m, self.n_tot - (first_obs_const + int(n / 2)), bias=False
            )  # (11,2)'
        else:
            self.koopman_fc_act = nn.Linear(m, self.n_tot - first_obs_const, bias=False)

        self.C = torch.cat(
            (
                torch.zeros((n_fixed_states, first_obs_const)),
                torch.eye(n_fixed_states),
                torch.zeros((n_fixed_states, encoder_output_dim)),
            ),
            1,
        )

    def forward(self, z0, u):
        # data = [x, u]
        _, act_matrix, _ = self.construct_drift_act_matrix_()
        bu = torch.matmul(u, torch.transpose(act_matrix, 0, 1))
        z = z0
        for i in range(len(self.affine_couplings)):
            z = self.affine_couplings[i](z, bu)
        return z

    def inverse(self, zT, u):
        # data = [u, x_prime]
        _, act_matrix, _ = self.construct_drift_act_matrix_()
        bu = torch.matmul(u, torch.transpose(act_matrix, 0, 1))
        z = zT
        for i in range(len(self.affine_couplings) - 1, -1, -1):
            z = self.affine_couplings[i].inverse(z, bu)
        return z

    def construct_drift_act_matrix_(self):
        n = self.net_params["state_dim"]
        m = self.net_params["ctrl_dim"]
        override_kinematics = self.net_params["override_kinematics"]
        first_obs_const = int(self.net_params["first_obs_const"])

        if override_kinematics:
            const_obs_dyn_act = torch.zeros(
                (first_obs_const, m), device=self.koopman_fc_act.weight.device
            )
            kinematics_dyn_act = torch.zeros(
                (int(n / 2), m), device=self.koopman_fc_act.weight.device
            )
            act_matrix = torch.cat(
                (const_obs_dyn_act, kinematics_dyn_act, self.koopman_fc_act.weight), 0
            )
        else:
            const_obs_dyn_act = torch.zeros(
                (first_obs_const, m), device=self.koopman_fc_act.weight.device
            )
            act_matrix = torch.cat((const_obs_dyn_act, self.koopman_fc_act.weight), 0)

        return None, act_matrix, None

    def send_to(self, device):
        hidden_dim = self.net_params["encoder_hidden_depth"]

        if hidden_dim > 0:
            self.encoder_fc_in.to(device)
            for ii in range(hidden_dim - 1):
                self.encoder_fc_hid[ii].to(device)
            self.encoder_fc_out.to(device)
        else:
            self.encoder_fc_out.to(device)

        self.koopman_fc_act.to(device)
        self.C = self.C.to(device)

        try:
            self.loss_scaler_x = self.loss_scaler_x.to(device)
        except:
            pass

    def process(self, data_x, t, data_u=None, downsample_rate=1, train_mode=True):
        n = self.net_params["state_dim"]
        m = self.net_params["ctrl_dim"]
        n_fixed_states = self.net_params["n_fixed_states"]
        n_traj = data_x.shape[0]

        data_scaled_x = self.preprocess_data(data_x, self.standardizer_x)
        data_scaled_u = self.preprocess_data(data_u, self.standardizer_u)
        x = data_scaled_x[:, :-1, :]
        u = data_scaled_u
        x_prime = data_scaled_x[:, 1:, :]

        order = "F"
        n_data_pts = n_traj * (x.shape[1])
        x_flat = x.T.reshape((n, n_data_pts), order=order)
        u_flat = u.T.reshape((m, n_data_pts), order=order)
        x_prime_flat = x_prime.T.reshape((n, n_data_pts), order=order)

        X = np.concatenate((x, u, x_prime), axis=2)
        y = np.concatenate((x, x_prime), axis=2)

        if train_mode:
            self.loss_scaler_x = torch.Tensor(
                np.std(
                    x_prime_flat[:n_fixed_states, :].T - x_flat[:n_fixed_states, :].T,
                    axis=0,
                )
            )  # (6,)
            self.loss_scaler_z = np.std(x_prime_flat.T - x_flat.T)

        return X[::downsample_rate, :], y[::downsample_rate, :]

    def construct_dyn_mat(self):
        override_C = self.net_params["override_C"]

        _, act_matrix, _ = self.construct_drift_act_matrix_()
        self.A = self.Ainv = np.empty((self.n_tot, self.n_tot))
        self.B = act_matrix.data.numpy()

        if not override_C:
            self.C = self.projection_fc.weight.detach().numpy()

    def get_l1_norm_(self):
        return torch.norm(self.koopman_fc_act.weight.view(-1), p=1)

    def Kloss(self, data, Ksteps=100, gamma=0.9):
        # inputs (traj,steps,N)  N = np.concatenate((x, u, x_prime), axis=2)
        # labels = [x, x_prime], penalize when lin_error is not zero,  (traj,steps,2n)  np.concatenate((x, x_prime - x), axis=2)
        n = self.net_params["n_fixed_states"]
        m = self.net_params["ctrl_dim"]
        first_obs_const = int(self.net_params["first_obs_const"])
        criterion = nn.MSELoss()
        alpha = self.net_params["lin_loss_penalty"]

        inputs, _ = data
        inputs = inputs.to(self.device)
        x_data = inputs[:, :, :n]
        x_prime_data = inputs[:, :, n + m :]
        u_data = inputs[:, :, n : n + m]
        beta = 1.0
        beta_sum = 0.0
        f_loss = torch.zeros(1, dtype=torch.float64).to(self.device)
        b_loss = torch.zeros(1, dtype=torch.float64).to(self.device)
        X_f = x_data[:, 0, :]
        zf = torch.cat(
            (
                torch.ones((X_f.shape[0], first_obs_const), device=self.device),
                X_f,
                self.encode_forward_(X_f),
            ),
            1,
        )

        X_b = x_prime_data[:, -1, :]
        zb = torch.cat(
            (
                torch.ones((X_b.shape[0], first_obs_const), device=self.device),
                X_b,
                self.encode_forward_(X_b),
            ),
            1,
        )
        for i in range(Ksteps):
            uf = u_data[:, i, :]
            zf = self.forward(zf, uf)
            x_pred = zf[:, : first_obs_const + n]
            label_xp = x_prime_data[:, i, :]
            pred_loss = criterion(x_pred, label_xp)
            xp_encoded = torch.cat(
                (
                    torch.ones(
                        (label_xp.shape[0], first_obs_const), device=self.device
                    ),
                    label_xp,
                    self.encode_forward_(label_xp),
                ),
                1,
            )
            lin_loss = criterion(xp_encoded, zf)
            f_loss += beta * (pred_loss + alpha * lin_loss)

            ub = u_data[:, Ksteps - 1 - i, :]
            zb = self.inverse(zb, ub)  # [x_proj, x_prev, z_prev]
            x_prev = zb[:, : first_obs_const + n]
            label_x = x_data[:, Ksteps - 1 - i, :]
            prev_loss = criterion(x_prev, label_x)
            xb_encoded = torch.cat(
                (
                    torch.ones((label_x.shape[0], first_obs_const), device=self.device),
                    label_x,
                    self.encode_forward_(label_x),
                ),
                1,
            )
            inv_lin_loss = criterion(zb, xb_encoded)
            b_loss += beta * (prev_loss + alpha * inv_lin_loss)

            beta_sum += beta
            beta *= gamma

        l1_loss = 0.0
        if "l1_reg" in self.net_params and self.net_params["l1_reg"] > 0:
            l1_reg = self.net_params["l1_reg"]
            l1_loss = l1_reg * self.get_l1_norm_()

        tot_loss = (
            f_loss / beta_sum + b_loss / beta_sum + l1_loss
        )  # about x + z + l1 + eye + prev x

        return (tot_loss, f_loss / beta_sum, b_loss / beta_sum, torch.empty(1))
