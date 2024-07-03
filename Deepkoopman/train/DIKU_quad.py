import sys
import os
import time
import pickle
import argparse
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
DKO_DIR = Path(__file__).absolute().parent.parent
ROOT_DIR = DKO_DIR.parent
sys.path.append(str(DKO_DIR))
sys.path.append(str(ROOT_DIR))
import torch
from copy import copy
import torch.optim as optim
from train.koopman_net_U import KoopmanNetCtrl
from train.koopman_net_IU import KoopmanNetInvCtrl
from Deepkoopman.Utility import data_collecter, set_seed
import dill
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.set_default_dtype(torch.float64)  # use double precision numbers

class KoopDnn:
    """
    Class for neural network-based Koopman methods to learn dynamics models of autonomous and controlled dynamical systems.
    """

    def __init__(self, net, first_obs_const=False, continuous_mdl=False, dt=None):
        self.A = None
        self.Ainv = None
        self.B = None

        self.net = net
        self.optimizer = None
        self.C = None

        self.first_obs_const = first_obs_const
        self.continuous_mdl = continuous_mdl
        self.dt = dt

        self.x_train = None
        self.u_train = None
        self.t_train = None
        self.x_val = None
        self.u_val = None
        self.t_val = None

    def set_datasets(
        self, x_train, t_train, u_train=None, x_val=None, t_val=None, u_val=None
    ):
        self.x_train = x_train
        self.t_train = np.tile(t_train, (self.x_train.shape[0], 1))
        self.u_train = u_train

        self.x_val = x_val
        self.t_val = np.tile(t_val, (self.x_val.shape[0], 1))
        self.u_val = u_val

    def model_pipeline(
        self,
        net_params,
        dkiu,
        logdir,
        logdir_trained,
        print_epoch=False,
        tune_run=False,
        early_stop=False,
        plot_data=False,
    ):
        self.net.net_params = net_params


        X_train, y_train = self.net.process(
            self.x_train, self.t_train, data_u=self.u_train
        )  # (time*traj,state+ctrl+future_state),(state,delta_state)
        X_val, y_val = self.net.process(
            self.x_val, self.t_val, data_u=self.u_val, train_mode=False
        )

        if plot_data:
            self.plot_train_data_(X_train, y_train)

        X_train_t, y_train_t = (
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
        )
        X_val_t, y_val_t = (
            torch.from_numpy(X_val),
            torch.from_numpy(y_val),
        )
        dataset_train = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        dataset_val = torch.utils.data.TensorDataset(X_val_t, y_val_t)
        self.train_model(
            dkiu,
            logdir,
            logdir_trained,
            dataset_train,
            dataset_val,
            print_epoch=print_epoch,
            tune_run=tune_run,
            early_stop=early_stop,
        )

    def train_model(
        self,
        dkiu,
        logdir,
        logdir_trained,
        dataset_train,
        dataset_val,
        print_epoch=True,
        tune_run=False,
        early_stop=False,
        early_stop_crit=1e-3,
        early_stop_max_count=5,
    ):
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda")

        trainloader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.net.net_params["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        valloader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=self.net.net_params["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        val_loss_prev = np.inf
        no_improv_counter = 0
        self.train_loss_hist = []
        self.val_loss_hist = []
        best_loss = np.inf
        writer = SummaryWriter(log_dir=logdir)
   
        if os.path.isfile(logdir + ".pth"):
            print("=> loading checkpoint '{}'".format(logdir + ".pth"))
            checkpoint = torch.load(logdir + ".pth")
            if dkiu:
                self.net.inn_config = checkpoint["INNconfig"]
                self.net.inn_layers = checkpoint["Ilayer"]
                self.net.masks = checkpoint["mask"]
                self.net.construct_net()
                self.set_optimizer_()
                self.net.send_to(device)
                self.net.load_state_dict(checkpoint["model"])
            else:
                self.net.construct_net()
                self.set_optimizer_()
                self.net.send_to(device)    
                self.net.load_state_dict(checkpoint["model"])
                for name, para in self.net.named_parameters():
                    if name.startswith("koopman_fc_act") or name.startswith("encoder_"):
                        para.requires_grad = False
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"]
            # optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            print("=> no checkpoint found at '{}'".format(logdir))
            self.net.construct_net()
            self.set_optimizer_()
            self.net.send_to(device)  
            start_epoch = 0

            if not dkiu:
                trained_file = logdir_trained + ".pth"
                assert os.path.exists(trained_file), "trained file not found"
                dict_trained = torch.load(trained_file)["model"]
                dict_new = self.net.state_dict().copy()
                for name, para in self.net.named_parameters():
                    if name.startswith("koopman_fc_act") or name.startswith("encoder_"):
                        dict_new[name] = dict_trained[name]
                self.net.load_state_dict(dict_new)
                for name, para in self.net.named_parameters():
                    if name.startswith("koopman_fc_act") or name.startswith("encoder_"):
                        para.requires_grad = False
        
        for name, param in self.net.named_parameters():
            print("model:", name, param.requires_grad)
            
        for epoch in range(start_epoch, start_epoch+self.net.net_params["epochs"]):
            running_loss = 0.0
            running_pred_loss = 0.0
            running_prev_loss = 0.0
            running_eye_loss = 0.0
            epoch_steps = 0

            for data in trainloader:
                self.optimizer.zero_grad()
                loss, pred_loss, prev_loss, eye_loss = self.net.Kloss(data)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.detach()
                running_pred_loss += pred_loss.detach()
                running_prev_loss += prev_loss.detach()
                running_eye_loss += eye_loss.detach()
                epoch_steps += 1

            # Validation loss:
            val_loss = 0.0
            val_pred_loss = 0.0
            val_prev_loss = 0.0
            val_eye_loss = 0.0
            val_steps = 0
            for data in valloader:
                with torch.no_grad():
                    loss, pred_loss, prev_loss, eye_loss = self.net.Kloss(data)
                    val_loss += float(loss.detach())
                    val_pred_loss += float(pred_loss.detach())
                    val_prev_loss += float(prev_loss.detach())
                    val_eye_loss += float(eye_loss.detach())
                    val_steps += 1

            # Print epoch loss:
            self.train_loss_hist.append(
                (
                    running_loss / epoch_steps,
                    running_pred_loss / epoch_steps,
                    running_prev_loss / epoch_steps,
                    running_eye_loss / epoch_steps,
                )
            )
            self.val_loss_hist.append(
                (
                    val_loss / val_steps,
                    val_pred_loss / val_steps,
                    val_prev_loss / val_steps,
                    val_eye_loss / val_steps,
                )
            )
            writer.add_scalar("Train/loss", self.train_loss_hist[-1][0], epoch)
            writer.add_scalar("Eval/loss", self.val_loss_hist[-1][0], epoch)
            writer.add_scalar("Eval/Kloss", self.val_loss_hist[-1][1], epoch)
            writer.add_scalar("Eval/inv_Kloss", self.val_loss_hist[-1][2], epoch)

            if print_epoch:
                print(
                    "Epoch %3d: train loss: %.8f, validation loss: %.8f, validation f_loss: %.8f, validation b_loss: %.8f, validation e_loss: %.8f"
                    % (
                        epoch + 1,
                        self.train_loss_hist[-1][0],
                        self.val_loss_hist[-1][0],
                        self.val_loss_hist[-1][1],
                        self.val_loss_hist[-1][2],
                        self.val_loss_hist[-1][3],
                    )
                )

            # Early stop if no improvement:
            if early_stop:
                improvement = (val_loss / val_steps) / val_loss_prev
                if (
                    improvement >= 1 - early_stop_crit
                    and improvement <= 1 + early_stop_crit
                ):
                    no_improv_counter += 1
                else:
                    no_improv_counter = 0

                if no_improv_counter >= early_stop_max_count:
                    print(
                        "Early stopping activated, less than %.4f improvement for the last %2d epochs."
                        % (early_stop_crit, no_improv_counter)
                    )
                    break
                val_loss_prev = val_loss / val_steps

            v_loss = self.val_loss_hist[-1][0]
            if v_loss < best_loss:
                best_loss = copy(v_loss)
                best_state_dict = copy(self.net.state_dict())
                if dkiu:
                    Saved_dict = {
                        "model": best_state_dict,
                        "Ilayer": self.net.inn_layers,
                        "INNconfig": self.net.inn_config,
                        "mask": self.net.masks,
                        "epoch": epoch,
                        "optimizer": self.optimizer.state_dict(),
                        "loss_scaler": [
                            self.net.loss_scaler_x,
                            torch.tensor(self.net.loss_scaler_z),
                        ],
                    }
                else:
                    Saved_dict = {
                        "model": best_state_dict,
                        "epoch": epoch,
                        "optimizer": self.optimizer.state_dict(),
                        "loss_scaler": [
                            self.net.loss_scaler_x,
                            torch.tensor(self.net.loss_scaler_z),
                        ],
                    }
                torch.save(Saved_dict, logdir + ".pth")
                writer.add_scalar("Eval/best_loss", best_loss, epoch)
                print(
                    "{} {}, Epoch:{} Train loss:{}, Eval total-loss:{}, K-loss:{}, inv-Kloss:{}".format(
                        time.asctime( time.localtime(time.time()) ),
                        self.net.net_params["suffix"],
                        epoch,
                        self.train_loss_hist[-1][0],
                        v_loss,
                        self.val_loss_hist[-1][1],
                        self.val_loss_hist[-1][2],
                    )
                )
        print("Finished Training")


    def K_fb_loss(data, sys, Nstate, udim):
        train_traj_num, steps, N = data.shape # Nstates=(xs,us)
        steps = steps - 1
        x_data = data[:, :-1, :Nstate]
        x_prime_data = data[:, 1:, :Nstate]
        u_data = data[:, :-1, Nstate:]

        max_loss_list = []
        mean_loss_list = []
        max_loss_inv_list = []
        mean_loss_inv_list = []
        X_f = x_data[:, 0, :]
        zf = torch.cat(
            (
                torch.ones((X_f.shape[0], sys.first_obs_const), device=sys.net.device),
                X_f,
                sys.net.encode_forward_(X_f),
            ),
            1,
        )
        X_b = x_prime_data[:, -1, :]
        zb = torch.cat(
            (
                torch.ones((X_b.shape[0], sys.first_obs_const), device=sys.net.device),
                X_b,
                sys.net.encode_forward_(X_b),
            ),
            1,
        )
        for i in range(steps):
            uf = u_data[:, i, :]
            zf = sys.net.forward(zf, uf)
            x_pred = zf[:, :sys.first_obs_const + Nstate]
            Err =  x_pred - x_prime_data[:, i, :]
            max_loss_list.append(
                torch.mean(torch.max(torch.abs(Err), axis=0).values).detach().cpu().numpy()
            )
            mean_loss_list.append(
                torch.mean(torch.mean(torch.abs(Err), axis=0)).detach().cpu().numpy()
            )

            ub = u_data[:, steps - 1 - i, :]
            zb = sys.net.inverse(zb, ub)
            x_prev = zb[:, :sys.first_obs_const + Nstate]
            Err1 =  x_prev - x_data[:, steps-1-i, :]
            max_loss_inv_list.insert(0,
                torch.mean(torch.max(torch.abs(Err1), axis=0).values).detach().cpu().numpy()
            )
            mean_loss_inv_list.insert(0,
                torch.mean(torch.mean(torch.abs(Err1), axis=0)).detach().cpu().numpy()
            )
        return (
            np.array(max_loss_list),
            np.array(mean_loss_list),
            np.array(max_loss_inv_list),
            np.array(mean_loss_inv_list),
        )

    def construct_koopman_model(self):
        self.net.send_to("cpu")
        self.construct_dyn_mat_()

    def construct_dyn_mat_(self):
        self.net.construct_dyn_mat()
        self.A = self.net.A
        self.Ainv = self.net.Ainv
        try:
            self.B = self.net.B
        except AttributeError:
            pass
        self.C = self.net.C

    def basis_encode(self, x):
        if self.net.standardizer_x is None:
            x_scaled = np.atleast_2d(x)
        else:
            x_scaled = self.net.standardizer_x.transform(x)

        return self.net.encode(x_scaled)

    def set_optimizer_(self):
        if self.net.net_params["optimizer"] == "sgd":
            lr = self.net.net_params["lr"]
            momentum = self.net.net_params["momentum"]
            self.optimizer_encoder = optim.SGD(
                self.net.opt_parameters_encoder, lr=lr, momentum=momentum
            )
            self.optimizer_dyn_mats = optim.SGD(
                self.net.opt_parameters_dyn_mats, lr=lr, momentum=momentum
            )
        elif self.net.net_params["optimizer"] == "adam":
            lr = self.net.net_params["lr"]
            weight_decay = self.net.net_params["l2_reg"]
            self.optimizer = optim.Adam(
                self.net.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif self.net.net_params["optimizer"] == "adamax":
            lr = self.net.net_params["lr"]
            weight_decay = self.net.net_params["l2_reg"]
            self.optimizer = optim.Adam(
                self.net.parameters(), lr=lr, weight_decay=weight_decay
            )


def main(args):
    set_seed(2023)
    load_tuned_params = False
    Data_collect = data_collecter(args.env_name)
    traindata_dir = str(DKO_DIR / "Data/PlanarQuadrotortrain.npy")
    testdata_dir = str(DKO_DIR / "Data/PlanarQuadrotortest.npy")
    if load_tuned_params:
        print("Loading tuned parameters")
        infile = open(str(DKO_DIR / "train/parameters_IU.pickle") if args.dkiu else str(DKO_DIR / "train/parameters.pickle"), "rb")
        net_params = dill.load(infile)
        infile.close()
        t_eval = Data_collect.t_eval
        n_traj_train = Data_collect.n_traj_train
        n_traj_val = Data_collect.n_traj_val
    else:
        print("Setting parameters")
        net_params = {}
        net_params["env_name"] = args.env_name
        net_params["suffix"] = args.suffix
        net_params["state_dim"] = Data_collect.Nstates
        net_params["ctrl_dim"] = Data_collect.udim
        net_params["encoder_hidden_width"] = 128
        net_params["encoder_hidden_depth"] = 5
        net_params["encoder_output_dim"] = 8
        net_params["optimizer"] = "adam"
        net_params["activation_type"] = "tanh"
        net_params["lr"] = 1e-3  
        net_params["epochs"] = 200000  # Best performance 2000
        net_params["batch_size"] = 1024
        net_params["lin_loss_penalty"] = 0.2  # Best performance until now: 0.2
        net_params["eye_loss_penalty"] = 0.01
        net_params["l2_reg"] = 0.0  # Best performance until now: 0.0
        net_params["l1_reg"] = 0.0
        net_params["n_fixed_states"] = 6
        net_params["first_obs_const"] = False
        net_params["override_kinematics"] = True
        net_params["override_C"] = True
        net_params["dt"] = Data_collect.dt
        net_params["inn_config"] = [args.N_aff, args.N_changed_elements]
        net_params["inn_layers"] = {
            "tnsl_layers": [128,64,32],
            "cmpd_layers": [128,64,32],
        }
        with open(str(DKO_DIR / "train/parameters_IU.pickle") if args.dkiu else str(DKO_DIR / "train/parameters.pickle"), "wb") as handle:
            pickle.dump(net_params, handle, protocol=pickle.HIGHEST_PROTOCOL)


    if os.path.exists(traindata_dir) and os.path.exists(testdata_dir):
        xs_train = np.load(traindata_dir)[:,:,:net_params["n_fixed_states"]]  # (200,301,6), 200 is the number of traj, (301,6) is the states after 300 steps
        us_train = np.load(traindata_dir)[:,:-1,net_params["n_fixed_states"]:]  # (200,300,2), 200 is the number of traj, (300,2) is the control inputs of 300 steps
        xs_val = np.load(testdata_dir)[:, :, :net_params["n_fixed_states"]]
        us_val = np.load(testdata_dir)[:, :-1, net_params["n_fixed_states"]:]
        # xd = np.load('../koopman_core/data/quad/traj_xd.npz')['arr_0']
    else:
        print("Cannot find expert data files")
        train_data = Data_collect.collect_koopman_data(Data_collect.n_traj_train, 100)
        np.save(traindata_dir, train_data)
        xs_train = train_data[:, :, :net_params["n_fixed_states"]]
        us_train = train_data[:, :-1, net_params["n_fixed_states"]:]
        test_data = Data_collect.collect_koopman_data(Data_collect.n_traj_val, 100)
        np.save(testdata_dir, test_data)
        xs_val = test_data[:, :, :net_params["n_fixed_states"]]
        us_val = test_data[:, :-1, net_params["n_fixed_states"]:]

    standardizer_u_kdnn = None
    if not args.dkiu:
        net = KoopmanNetCtrl(net_params, standardizer_u=standardizer_u_kdnn)
        logdir = str(DKO_DIR /
            "Data"
            / args.suffix
            / "KoopmanU_{}layer{}_edim{}_eloss{}_gamma{}".format(
                args.env_name,
                net_params["encoder_hidden_depth"],
                net_params["encoder_output_dim"],
                args.e_loss,
                args.gamma,
            )
        )
        logdir_trained = str(DKO_DIR /
            "Data"
            / args.trained_suffix
            / "KoopmanIU_{}layer{}_edim{}_Naff{}_Nchange{}".format(
                args.env_name,
                net_params["encoder_hidden_depth"],
                net_params["encoder_output_dim"],
                net_params["inn_config"][0],
                net_params["inn_config"][1],
            )
        )
    else:
        net = KoopmanNetInvCtrl(net_params, standardizer_u=standardizer_u_kdnn)
        logdir = str(DKO_DIR /
            "Data"
            / args.suffix
            / "KoopmanIU_{}layer{}_edim{}_Naff{}_Nchange{}".format(
                args.env_name,
                net_params["encoder_hidden_depth"],
                net_params["encoder_output_dim"],
                net_params["inn_config"][0],
                net_params["inn_config"][1],
            )
        )
        logdir_trained = ""
    if not os.path.exists(str(DKO_DIR / "Data" / args.suffix)):
        os.makedirs(str(DKO_DIR /"Data" / args.suffix))
    if not os.path.exists(logdir):
        print(logdir)
        os.makedirs(logdir)
    model_koop_dnn = KoopDnn(net)
    model_koop_dnn.set_datasets(
        xs_train,
        np.tile(Data_collect.t_eval, (Data_collect.n_traj_train, 1)),
        u_train=us_train,
        x_val=xs_val,
        u_val=us_val,
        t_val=np.tile(Data_collect.t_eval, (Data_collect.n_traj_val, 1)),
    )
    model_koop_dnn.model_pipeline(
        net_params, args.dkiu, logdir, logdir_trained, tune_run=True, early_stop=False, plot_data=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="PlanarQuadrotor")
    parser.add_argument("--suffix", type=str, default="Quad0614")
    parser.add_argument("--trained_suffix", type=str, default="Quad0614")
    parser.add_argument("--e_loss", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--dkiu", type=int, default=0)
    parser.add_argument("--N_aff", type=int, default=1)
    parser.add_argument("--N_changed_elements", type=int, default=1)

    args = parser.parse_args()
    main(args)
