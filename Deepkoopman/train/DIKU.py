"""
train KoopmanIU(DKIU) zk+1=INN(zk), z=[x,g_{tht}(x)]
"""
import sys
import os
import time
from pathlib import Path
DKO_DIR = Path(__file__).absolute().parent.parent
ROOT_DIR = DKO_DIR.parent
sys.path.append(str(DKO_DIR))
sys.path.append(str(ROOT_DIR))
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from copy import copy
import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
import argparse
from Deepkoopman.Utility import data_collecter, set_seed


torch.set_default_dtype(torch.float64)  # use double precision numbers


# torch.cuda.set_device(1)
# define network
def gaussian_init_(n_units, std=1):
    sampler = torch.distributions.Normal(
        torch.Tensor([0]), torch.Tensor([std / n_units])
    )
    Omega = sampler.sample((n_units, n_units))[..., 0]
    return Omega


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        # a trajectory (control;state) of K steps (steps,Nstates)
        X = self.data[:, index, :]
        return X

    def __len__(self):
        # traj num
        return self.data.shape[1]


class Network(nn.Module):
    def __init__(self, Nstate, u_dim, encode_layers, inn_layers, inn_config, device):
        super(Network, self).__init__()
        self.device = device
        self.Nkoopman = Nstate + encode_layers[-1]
        self.inn_layers = inn_layers
        N_inn = inn_config[0]
        N_changed_elements = inn_config[1]
        masks = np.ones((N_inn, self.Nkoopman))
        indices = np.random.choice(
            masks.size, N_changed_elements, replace=False
        )
        masks.flat[indices] = 0
        self.u_dim = u_dim
        self.masks = nn.ParameterList(
            [nn.Parameter(torch.Tensor(m), requires_grad=False) for m in masks]
        ).to(device)

        Layers = OrderedDict()
        for layer_i in range(len(encode_layers) - 1):
            Layers["linear_{}".format(layer_i)] = nn.Linear(
                encode_layers[layer_i], encode_layers[layer_i + 1]
            )
            if layer_i != len(encode_layers) - 2:
                Layers["relu_{}".format(layer_i)] = nn.ReLU()
        self.encode_net = nn.Sequential(Layers)

        self.lB = nn.Linear(u_dim, self.Nkoopman, bias=False)

        self.z0_MC = None
        self.z0 = None
        self.Q_z0 = None
        self.ctrl_MC = None
        self.eta_z0s = 0.01
        self.eta_ctrl = 0.01

    def encode_only(self, x):
        return self.encode_net(x)

    def encode(self, x):
        return torch.cat([x, self.encode_net(x)], axis=-1)

    def forward(self, x, us):
        z = x
        u = self.lB(us)
        for i in range(len(self.affine_couplings)):
            z = self.affine_couplings[i](z, u)
        return z

    def inverse(self, xp, us):
        z = xp
        u = self.lB(us)
        for i in range(len(self.affine_couplings) - 1, -1, -1):
            z = self.affine_couplings[i].inverse(z, u)
        return z
    
    def construct_net(self):
        # Used for linear INN comparison experiments
        # from Deepkoopman.train.linear_INNU import Affine_Coupling
        # AFF_s = []
        # for i in range(len(self.masks)):
        #     mask = nn.Parameter(self.masks[i], requires_grad=False)
        #     self.fix_idx = torch.nonzero(mask).squeeze()
        #     self.change_idx = torch.nonzero(1 - mask).squeeze()
        #     self.fix_num = self.fix_idx.numel()
        #     self.change_num = self.change_idx.numel()
        #     INN_layers=       {
        #             "tnsl_layers": [self.change_num] + self.inn_layers["tnsl_layers"] + [self.fix_num],
        #             "cmpd_layers": [self.fix_num] + self.inn_layers["cmpd_layers"] + [self.change_num],
        #         }
        #     AFF_s.append(Affine_Coupling(mask, INN_layers))
        # self.affine_couplings = nn.ModuleList(AFF_s).to(self.device)

        ### Used for INN with ReLU comparison experiments
        from Deepkoopman.train.INNU import Affine_Coupling
        net_inn_layers={}
        net_inn_layers["tnsl_layers"] = [self.Nkoopman] + self.inn_layers["tnsl_layers"] + [self.Nkoopman]
        net_inn_layers["cmpd_layers"] = [self.Nkoopman] + self.inn_layers["cmpd_layers"] + [self.Nkoopman]
        self.affine_couplings = nn.ModuleList(Affine_Coupling(self.masks[i], net_inn_layers) for i in range(len(self.masks))).to(self.device)
    
    def construct_dyn_mat(self):
        if len(self.masks)==1:
            A1 = np.eye(self.fix_num)
            A2 = np.eye(self.change_num)
            for m2 in self.affine_couplings[0].tnsl_net:
                A2 = A2.dot(m2.weight.T.data.numpy())
            for m1 in self.affine_couplings[0].cmpd_net:
                A1 = A1.dot(m1.weight.T.data.numpy())
            ul_submatrix = np.eye(self.fix_num)
            ur_submatrix = A2
            ll_submatrix = A1
            lr_submatrix = np.dot(ur_submatrix, ll_submatrix) + np.eye(self.change_num)
            self.A = np.vstack((np.concatenate((ul_submatrix, ur_submatrix), axis=1), np.concatenate((ll_submatrix, lr_submatrix), axis=1)))
            B1 = self.act_matrix.data.numpy()[:, self.fix_index]
            B2 = self.act_matrix.data.numpy()[:, self.change_index]
            B_l = np.dot(ll_submatrix, B1)+B2
            self.B = np.vstack(B1, B_l)


def K_fb_loss(data, net, Nstate, u_dim):
    steps, train_traj_num, Nstates = data.shape
    Z_current = net.encode(data[0, :, u_dim:])  # [x,g(x)]
    max_loss_list = []
    mean_loss_list = []
    max_loss_inv_list = []
    mean_loss_inv_list = []
    for i in range(steps - 1):
        Z_current = net.forward(Z_current, data[i, :, :u_dim])
        Err = Z_current[:, :Nstate] - data[i + 1, :, u_dim:]
        max_loss_list.append(
            torch.mean(torch.max(torch.abs(Err), axis=0).values).detach().cpu().numpy()
        )
        mean_loss_list.append(
            torch.mean(torch.mean(torch.abs(Err), axis=0)).detach().cpu().numpy()
        )

    Z_current = net.encode(data[-1, :, u_dim:])  # [x,g(x)]
    for j in range(steps - 1, 0, -1):
        Z_current = net.inverse(Z_current, data[j - 1, :, :u_dim])
        Err = Z_current[:, :Nstate] - data[j - 1, :, u_dim:]
        max_loss_inv_list.insert(0,
            torch.mean(torch.max(torch.abs(Err), axis=0).values).detach().cpu().numpy()
        )
        mean_loss_inv_list.insert(0,
            torch.mean(torch.mean(torch.abs(Err), axis=0)).detach().cpu().numpy()
        )
    return (
        np.array(max_loss_list),
        np.array(mean_loss_list),
        np.array(max_loss_inv_list),
        np.array(mean_loss_inv_list),
    )
      

def Klinear_fb_loss(
    data, device, net, mse_loss, u_dim=1, gamma=0.99, Nstate=4, z_loss=0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.DoubleTensor(data).to(device)
    train_traj_num, steps, _ = data.shape[0], data.shape[1], data.shape[2]
    Zf = net.encode(data[:, 0, u_dim:])
    Zb = net.encode(data[:, -1, u_dim:])
    beta = 1.0
    beta_sum = 0.0
    f_loss = torch.zeros(1, dtype=torch.float64).to(device)
    b_loss = torch.zeros(1, dtype=torch.float64).to(device)
    for i in range(steps - 1):
        Zf = net.forward(Zf, data[:, i, :u_dim])
        if not z_loss:
            f_loss += beta * mse_loss(Zf[:, :Nstate], data[:, i + 1, u_dim:])
        else:
            Y = net.encode(data[:, i + 1, u_dim:])
            f_loss += beta * mse_loss(Zf, Y)

        Zb = net.inverse(Zb, data[:, -i - 2, :u_dim])
        if not z_loss:
            b_loss += beta * mse_loss(Zb[:, :Nstate], data[:, -i - 2, u_dim:])
        else:
            b_loss += beta * mse_loss(Zb, net.encode(data[:, -i - 2, u_dim:]))
        beta_sum += beta
        beta *= gamma
    return f_loss / beta_sum, b_loss / beta_sum


def train(
    config,
    env_name,
    suffix,
    logdir,
    z_loss=0,
    gamma=0.5,
):
    set_seed(2023)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dataset
    data_collect = data_collecter(env_name)
    u_dim = data_collect.udim
    Nstate = data_collect.Nstates

    mse_loss = nn.MSELoss()

    # data prepare
    Ktrain_data = np.load(DKO_DIR / "Data/{}train.npy".format(env_name))
    train_dataset = CustomDataset(Ktrain_data)


    Ktest_data = np.load(DKO_DIR / "Data/{}test.npy".format(env_name))
    test_dataset = CustomDataset(Ktest_data)

    trainloader = DataLoader(
        train_dataset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
    )
    valloader = DataLoader(
        test_dataset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
    )

    writer = SummaryWriter(log_dir=logdir)
    if os.path.isfile(logdir + ".pth"):
        print("=> loading checkpoint '{}'".format(logdir + ".pth"))
        checkpoint = torch.load(logdir + ".pth")
        # Network ach parameters
        encoder_layers = checkpoint["Elayer"]
        encode_length = 2*(len(encoder_layers) - 1)
        Nkoopman = Nstate + encoder_layers[-1]
        inn_layers = checkpoint["Ilayer"]
        inn_config = checkpoint["INNconfig"]
        net = Network(Nstate, u_dim, encoder_layers, inn_layers, inn_config, device)
        net.masks = checkpoint["mask"]
        net.construct_net()
        net.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint["epoch"]
        # optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        print("=> no checkpoint found at '{}'".format(logdir))
        start_epoch = 0
        # Network ach parameters
        encoder_layers = (
            [Nstate] + [config["width"]] * config["depth"] + [config["encode_dim"]]
        )
        Nkoopman = Nstate + config["encode_dim"]
        inn_layers = config["inn_layers"]
        inn_config = [config["N_aff"], config["N_changed_elements"]]
        net = Network(Nstate, u_dim, encoder_layers, inn_layers, inn_config, device)
        net.construct_net()
    
    for name, para in net.named_parameters():
        print("model:", name, para.requires_grad)
            
    net.to(device)
    net.double()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=config["lr"])
    # train
    eval_step = 20000
    best_loss = 1000.0
    for epoch in range(start_epoch, eval_step):
        # K loss
        train_steps = 0
        for i, data in enumerate(trainloader, 0):
            train_steps += 1
            optimizer.zero_grad()
            Kloss, inv_Kloss = Klinear_fb_loss(
                data, device, net, mse_loss, u_dim, gamma, Nstate, z_loss
            )
            loss = Kloss + inv_Kloss
            loss.backward()
            optimizer.step()
        writer.add_scalar("Train/loss", loss, epoch)

        val_loss = 0.0
        val_steps = 0
        Klosses = 0.0
        inv_Klosses = 0.0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                val_steps += 1
                Kloss, inv_Kloss = Klinear_fb_loss(
                    data, device, net, mse_loss, u_dim, gamma, Nstate, z_loss
                )
                loss = Kloss + inv_Kloss

                # print statistics
                val_loss += loss.cpu().numpy()
                Klosses += Kloss.cpu().numpy()
                inv_Klosses += inv_Kloss.cpu().numpy()

        val_loss /= val_steps
        Klosses /= val_steps
        inv_Klosses /= val_steps

        writer.add_scalar("Eval/loss", val_loss, epoch)
        writer.add_scalar("Eval/Kloss", Klosses, epoch)
        writer.add_scalar("Eval/inv_Kloss", inv_Klosses, epoch)

        if val_loss < best_loss:
            best_loss = copy(val_loss)
            best_state_dict = copy(net.state_dict())
            Saved_dict = {
                "model": best_state_dict,
                "Elayer": encoder_layers,
                "Ilayer": inn_layers,
                "INNconfig": inn_config,
                "mask": net.masks,
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
            }
            torch.save(Saved_dict, logdir + ".pth")
            writer.add_scalar("Eval/best_loss", best_loss, epoch)
            print(
                "{} {} Epoch:{} Eval total-loss:{}, K-loss:{}, inv-Kloss:{}".format(
                    time.asctime( time.localtime(time.time()) ), suffix, epoch, val_loss, Klosses, inv_Klosses
                )
            )
    print("Finish Training")


def main(args):
    data_collect = data_collecter(args.env_name)
    Nstate = data_collect.Nstates
    u_dim = data_collect.udim
    if not os.path.isfile(DKO_DIR / "Data/{}train.npy".format(args.env_name)):
        Ktrain_samples = 10000
        Ktest_samples = 2500
        Ksteps = 100
        print("Prepare Data")
        Ktrain_data = data_collect.collect_koopman_data(Ktrain_samples, Ksteps)
        print("train data ok!")
        np.save(DKO_DIR / "Data/{}train.npy".format(args.env_name), Ktrain_data)
        Ktest_data = data_collect.collect_koopman_data(Ktest_samples, Ksteps)
        print("test data ok!")
        np.save(DKO_DIR / "Data/{}test.npy".format(args.env_name), Ktest_data)

    config = {
        # encoder arch
        "width": 128,
        "depth": args.layer_depth,
        "encode_dim": args.encode_dim,
        # INN arch
        "N_aff": args.N_aff,
        "N_changed_elements": args.N_changed_elements,
        "inn_layers" : {
        "tnsl_layers": [128,64,32],
        "cmpd_layers": [128,64,32],
        },
        "lr": 0.001,
        "batch_size": 1024,
    }
    print(config["inn_layers"])

    logdir = str(DKO_DIR /
        "Data"
        / args.suffix
        / "KoopmanIU_{}layer{}_edim{}_Naff{}_Nchange{}".format(
            args.env_name,
            config["depth"],
            config["encode_dim"],
            config["N_aff"],
            config["N_changed_elements"],
        )
    )
    if not os.path.exists(str(DKO_DIR / "Data" / args.suffix)):
        os.makedirs(str(DKO_DIR / "Data" / args.suffix))
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    train(config, args.env_name, args.suffix, logdir, z_loss=args.z_loss, gamma=args.gamma)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="CartPole")
    parser.add_argument("--suffix", type=str, default="CartPole0627")
    parser.add_argument("--z_loss", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--layer_depth", type=int, default=3)
    parser.add_argument("--encode_dim", type=int, default=20)
    parser.add_argument("--N_aff", type=int, default=1)
    parser.add_argument("--N_changed_elements", type=int, default=1)
    args = parser.parse_args()
    main(args)
