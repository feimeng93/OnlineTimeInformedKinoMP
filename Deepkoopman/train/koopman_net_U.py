import torch
import torch.nn as nn
import sys
from pathlib import Path
TRAIN_DIR = Path('__file__').absolute().parent
sys.path.append(str(TRAIN_DIR))
from .koopman_net import KoopmanNet
import numpy as np


class KoopmanNetCtrl(KoopmanNet):
    def __init__(self, net_params, standardizer_x=None, standardizer_u=None):
        super(KoopmanNetCtrl, self).__init__(
            net_params, standardizer_x=standardizer_x, standardizer_u=standardizer_u
        )

    def construct_net(self):
        n = self.net_params["state_dim"]
        m = self.net_params["ctrl_dim"]
        encoder_output_dim = self.net_params["encoder_output_dim"]
        first_obs_const = int(self.net_params["first_obs_const"])
        n_fixed_states = self.net_params["n_fixed_states"]
        override_C = self.net_params["override_C"]
        override_kinematics = self.net_params["override_kinematics"]

        if override_C:
            self.n_tot = int(first_obs_const) + n_fixed_states + encoder_output_dim
        else:
            self.n_tot = int(first_obs_const) + encoder_output_dim
            assert (
                override_kinematics is False
            ), "Not overriding C while overriding kinematics not supported"

        # self.C = torch.cat((torch.zeros((n_fixed_states, first_obs_const)), torch.eye(n_fixed_states), torch.zeros((n_fixed_states, encoder_output_dim))), 1)
        self.construct_encoder_()
        if self.net_params["override_kinematics"]:
            self.koopman_fc_drift = nn.Linear(
                self.n_tot, self.n_tot - (first_obs_const + int(n / 2)), bias=False
            )  # data size (14,11)'
            self.koopman_fc_inv_drift = nn.Linear(
                self.n_tot, self.n_tot - (first_obs_const + int(n / 2)), bias=False
            )
            self.koopman_fc_act = nn.Linear(
                m, self.n_tot - (first_obs_const + int(n / 2)), bias=False
            )  # (11,2)'
        else:
            self.koopman_fc_drift = nn.Linear(
                self.n_tot, self.n_tot - first_obs_const, bias=False
            )
            self.koopman_fc_inv_drift = nn.Linear(
                self.n_tot, self.n_tot - first_obs_const, bias=False
            )
            self.koopman_fc_act = nn.Linear(m, self.n_tot - first_obs_const, bias=False)

        # self.C = torch.cat((torch.zeros((n, first_obs_const)), torch.eye(n), torch.zeros((n, encoder_output_dim))), 1)
        self.C = torch.cat(
            (
                torch.zeros((n_fixed_states, first_obs_const)),
                torch.eye(n_fixed_states),
                torch.zeros((n_fixed_states, encoder_output_dim)),
            ),
            1,
        )

        self.parameters_to_prune = [
            (self.koopman_fc_drift, "weight"),
            (self.koopman_fc_act, "weight"),
            (self.koopman_fc_inv_drift, "weight"),
        ]

    def forward(self, data):
        # data = [x, u, x_prime]
        # output = [x_prime_pred, z_prime_pred, z_prime]
        n = self.net_params["state_dim"]
        m = self.net_params["ctrl_dim"]
        first_obs_const = int(self.net_params["first_obs_const"])
        override_C = self.net_params["override_C"]
        n_fixed_states = self.net_params["n_fixed_states"]

        x = data[:, :n]
        u = data[:, n : n + m]
        x_prime = data[:, n + m :]

        # Define linearity networks:
        z = torch.cat(
            (
                torch.ones((x.shape[0], first_obs_const), device=self.device),
                x[:, :n_fixed_states],
                self.encode_forward_(x),
            ),
            1,
        )
        z_prime_diff = (
            self.encode_forward_(x_prime) - z[:, first_obs_const + n_fixed_states :]
        )

        drift_matrix, act_matrix, _ = self.construct_drift_act_matrix_()
        z_prime_diff_pred = torch.matmul(
            z, torch.transpose(drift_matrix, 0, 1)
        ) + torch.matmul(u, torch.transpose(act_matrix, 0, 1))

        # Define prediction network:
        x_proj = torch.matmul(z, torch.transpose(self.C, 0, 1))
        x_prime_diff_pred = torch.matmul(
            z_prime_diff_pred, torch.transpose(self.C, 0, 1)
        )
        z_prime_diff_pred = z_prime_diff_pred[:, first_obs_const + n_fixed_states :]

        return torch.cat(
            (x_proj, x_prime_diff_pred, z_prime_diff_pred, z_prime_diff), 1
        )

    def inverse(self, data):
        # data = [x, u, x_prime]
        n = self.net_params["state_dim"]
        m = self.net_params["ctrl_dim"]
        n_z = self.net_params["encoder_output_dim"]
        first_obs_const = int(self.net_params["first_obs_const"])
        n_fixed_states = self.net_params["n_fixed_states"]

        x = data[:, :n]
        u = data[:, n : n + m]
        x_prime = data[:, n + m :]

        # inverse linearity networks:
        z = torch.cat(
            (
                torch.ones((x_prime.shape[0], first_obs_const), device=self.device),
                x_prime[:, :n_fixed_states],
                self.encode_forward_(x_prime),
            ),
            1,
        )

        _, act_matrix, drift_inv_matrix = self.construct_drift_act_matrix_()
        z_prev = torch.matmul(
            z - torch.matmul(u, torch.transpose(act_matrix, 0, 1)),
            torch.transpose(drift_inv_matrix, 0, 1),
        )

        # Define previous network
        x_proj = torch.matmul(z, torch.transpose(self.C, 0, 1))
        x_prev = torch.matmul(z_prev, torch.transpose(self.C, 0, 1))
        z_prev = z_prev[:, first_obs_const + n_fixed_states :]

        return torch.cat((x_proj, x_prev, z_prev), 1)

    def construct_drift_act_matrix_(self):
        n = self.net_params["state_dim"]
        m = self.net_params["ctrl_dim"]
        override_kinematics = self.net_params["override_kinematics"]
        first_obs_const = int(self.net_params["first_obs_const"])
        dt = self.net_params["dt"]

        if override_kinematics:
            const_obs_dyn_drift = torch.zeros(
                (first_obs_const, self.n_tot),
                device=self.koopman_fc_drift.weight.device,
            )
            kinematics_dyn_drift = torch.zeros(
                (int(n / 2), self.n_tot), device=self.koopman_fc_drift.weight.device
            )
            kinematics_dyn_drift[
                :, first_obs_const + int(n / 2) : first_obs_const + n
            ] = dt * torch.eye(int(n / 2), device=self.koopman_fc_drift.weight.device)
            drift_matrix = torch.cat(
                (
                    const_obs_dyn_drift,
                    kinematics_dyn_drift,
                    self.koopman_fc_drift.weight,
                ),
                0,
            )

            drift_inv_matrix = torch.cat(
                (
                    const_obs_dyn_drift,
                    kinematics_dyn_drift,
                    self.koopman_fc_inv_drift.weight,
                ),
                0,
            )

            const_obs_dyn_act = torch.zeros(
                (first_obs_const, m), device=self.koopman_fc_drift.weight.device
            )
            kinematics_dyn_act = torch.zeros(
                (int(n / 2), m), device=self.koopman_fc_drift.weight.device
            )
            act_matrix = torch.cat(
                (const_obs_dyn_act, kinematics_dyn_act, self.koopman_fc_act.weight), 0
            )
        else:
            const_obs_dyn_drift = torch.zeros(
                (first_obs_const, self.n_tot),
                device=self.koopman_fc_drift.weight.device,
            )
            drift_matrix = torch.cat(
                (const_obs_dyn_drift, self.koopman_fc_drift.weight), 0
            )


            const_obs_dyn_act = torch.zeros(
                (first_obs_const, m), device=self.koopman_fc_drift.weight.device
            )
            act_matrix = torch.cat((const_obs_dyn_act, self.koopman_fc_act.weight), 0)

        return drift_matrix, act_matrix, drift_inv_matrix

    def send_to(self, device):
        hidden_dim = self.net_params["encoder_hidden_depth"]

        if hidden_dim > 0:
            self.encoder_fc_in.to(device)
            for ii in range(hidden_dim - 1):
                self.encoder_fc_hid[ii].to(device)
            self.encoder_fc_out.to(device)
        else:
            self.encoder_fc_out.to(device)

        self.koopman_fc_drift.to(device)
        self.koopman_fc_inv_drift.to(device)
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

        X = np.concatenate((x, u, x_prime), axis=2)
        # y = x_prime_flat.T
        y = np.concatenate((x, x_prime - x), axis=2)

        order = 'F'
        n_data_pts = n_traj * (x.shape[1])
        x_flat = x.T.reshape((n, n_data_pts), order=order)
        u_flat = u.T.reshape((m, n_data_pts), order=order)
        x_prime_flat = x_prime.T.reshape((n, n_data_pts), order=order)


        if train_mode:
            self.loss_scaler_x = torch.Tensor(np.std(x_prime_flat[:n_fixed_states, :].T - x_flat[:n_fixed_states, :].T, axis=0)) #(6,)
            self.loss_scaler_z = np.std(x_prime_flat.T - x_flat.T) #(1,)

        return X[::downsample_rate, :], y[::downsample_rate, :]

    def construct_dyn_mat(self):
        n = self.net_params["state_dim"]
        m = self.net_params["ctrl_dim"]
        encoder_output_dim = self.net_params["encoder_output_dim"]
        first_obs_const = int(self.net_params["first_obs_const"])
        override_kinematics = self.net_params["override_kinematics"]
        override_C = self.net_params["override_C"]

        loss_scaler = np.concatenate(
            (
                np.ones(first_obs_const),
                self.loss_scaler_x.cpu().numpy(),
                self.loss_scaler_z * np.ones(encoder_output_dim),
            )
        )

        drift_matrix, act_matrix, drift_inv_matrix = self.construct_drift_act_matrix_()

        self.A = drift_matrix.data.cpu().numpy()
        self.Ainv = drift_inv_matrix.data.cpu().numpy()
        if override_kinematics:
            # self.A[first_obs_const+int(n/2):,:] *= self.loss_scaler
            self.A[first_obs_const + int(n / 2) :, :] = np.multiply(
                self.A[first_obs_const + int(n / 2) :, :],
                loss_scaler[first_obs_const + int(n / 2) :].reshape(-1, 1),
            )

            self.Ainv[:, first_obs_const + int(n / 2) :] = np.multiply(
                self.Ainv[:, first_obs_const + int(n / 2) :],
                loss_scaler[first_obs_const + int(n / 2) :].reshape(1, -1),
            )

            if self.standardizer_x is not None:
                x_dot_scaling = np.divide(
                    self.standardizer_x.scale_[int(n / 2) :],
                    self.standardizer_x.scale_[: int(n / 2)],
                ).reshape(-1, 1)
                self.A[first_obs_const : first_obs_const + int(n / 2), :] = np.multiply(
                    self.A[first_obs_const : first_obs_const + int(n / 2), :],
                    x_dot_scaling,
                )
                self.Ainv[
                    first_obs_const : first_obs_const + int(n / 2), :
                ] = np.multiply(
                    self.Ainv[first_obs_const : first_obs_const + int(n / 2), :],
                    x_dot_scaling,
                )
        else:
            # self.A[first_obs_const:, :] *= self.loss_scaler
            self.A[first_obs_const:, :] = np.multiply(
                self.A[first_obs_const:, :],
                loss_scaler[first_obs_const:].reshape(-1, 1),
            )
        self.A += np.eye(self.n_tot)

        B_vec = act_matrix.data.cpu().numpy()
        self.B = np.multiply(
                B_vec,
                loss_scaler.reshape(-1, 1),
            )


        if not override_C:
            self.C = self.projection_fc.weight.detach().numpy()

    def get_l1_norm_(self):
        return torch.norm(self.koopman_fc_drift.weight.view(-1), p=1) + torch.norm(self.koopman_fc_inv_drift.weight.view(-1), p=1) + torch.norm(
            self.koopman_fc_act.weight.view(-1), p=1
        )

    def Kloss(self, data, Ksteps=100, gamma=0.9):
        # inputs (traj,steps,N)  N = np.concatenate((x, u, x_prime), axis=2)
        # labels = [x, x_prime], penalize when lin_error is not zero,  (traj,steps,2n)  np.concatenate((x, x_prime - x), axis=2)
        n = self.net_params["n_fixed_states"]
        m = self.net_params["ctrl_dim"]
        n_z = self.net_params["encoder_output_dim"]
        n_override_kinematics = int(n / 2) * int(self.net_params["override_kinematics"])
        criterion = nn.MSELoss()
        alpha = self.net_params["lin_loss_penalty"]
        alpha_0 = self.net_params["eye_loss_penalty"]

        inputs, labels = data
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        beta = 1.0
        beta_sum = 0.0
        f_loss = torch.zeros(1, dtype=torch.float64).to(self.device)
        b_loss = torch.zeros(1, dtype=torch.float64).to(self.device)
        X_f = inputs[:, 0, :]
        X_b = inputs[:, -1, :]
        for i in range(Ksteps - 1):
            pred_outputs = self.forward(X_f) ## [x_proj, x_prime_diff_pred, z_prime_diff_pred, z_prime_diff]
            x_prime_diff_pred = pred_outputs[:, n + n_override_kinematics: 2 * n]
            x_prime_diff = labels[:, i, n + n_override_kinematics: 2 * n]
            z_prime_diff_pred = pred_outputs[:, 2 * n: 2 * n + n_z]
            z_prime_diff = pred_outputs[:, 2 * n + n_z: 2 * n + 2 * n_z]
            X_fp1 = pred_outputs[:, n: 2 * n]+ X_f[:,:n]
            X_f = torch.concat((X_fp1,inputs[:,i+1,n:]),dim=1) #(x, u, x_prime)
            pref_loss = criterion(
                x_prime_diff_pred,
                torch.divide(x_prime_diff, self.loss_scaler_x[n_override_kinematics:n]),
            )
            lin_loss = criterion(z_prime_diff_pred, z_prime_diff / self.loss_scaler_z) / (
                    n_z / n
            )
            f_loss += beta * (pref_loss + alpha * lin_loss)

            prev_outputs = self.inverse(X_b)  # [x_proj, x_prev, z_prev]
            x_prev = prev_outputs[:, n: 2 * n]
            z_prev = prev_outputs[:, 2 * n: 2 * n + n_z]
            prev_loss = criterion(x_prev, inputs[:, -i-1, :n])
            X_b = torch.concat((x_prev, inputs[:,-i-1,n:]), dim=1)
            x_prev_encoded = self.encode_forward_(inputs[:, -i-1, :n])
            inv_lin_loss = criterion(z_prev, x_prev_encoded)
            b_loss += beta * (prev_loss + alpha * inv_lin_loss)
            beta_sum += beta
            beta *= gamma

        l1_loss = 0.0
        if "l1_reg" in self.net_params and self.net_params["l1_reg"] > 0:
            l1_reg = self.net_params["l1_reg"]
            l1_loss = l1_reg * self.get_l1_norm_()

        self.construct_dyn_mat()
        lAw = torch.from_numpy(self.A).to(self.device)
        lAiw = torch.from_numpy(self.Ainv).to(self.device)
        eye = torch.eye(lAw.shape[0], device=self.device)
        eye_loss = torch.linalg.matrix_norm(torch.mm(lAw, lAiw) - eye, ord="fro")

        tot_loss = (
            f_loss / beta_sum
            +b_loss /beta_sum
            + l1_loss
            + alpha_0 * eye_loss
        )  # about x + z + l1 + eye + prev x

        return (
            tot_loss,
            f_loss/beta_sum,
            b_loss/beta_sum,
            alpha_0 * eye_loss,
        )
