"""
train KoopmanU(DKUC) z=K_1(z)+K_2(u), z=[x,g_{tht}(x)]
"""
import os
import sys
from pathlib import Path
DKO_DIR = Path(__file__).absolute().parent.parent
ROOT_DIR = DKO_DIR.parent
sys.path.append(str(DKO_DIR))
sys.path.append(str(ROOT_DIR))
import torch
import numpy as np
import torch.nn as nn
import random
from collections import OrderedDict
from copy import copy
import argparse
from Deepkoopman.Utility import data_collecter, set_seed
from torch.utils.tensorboard import SummaryWriter
import time


# torch.cuda.set_device(1)
# define network
def gaussian_init_(n_units, std=1):
    sampler = torch.distributions.Normal(
        torch.Tensor([0]), torch.Tensor([std / n_units])
    )
    Omega = sampler.sample((n_units, n_units))[..., 0]
    return Omega


class Network(nn.Module):
    def __init__(self, encode_layers, Nkoopman, u_dim):
        super(Network, self).__init__()
        Layers = OrderedDict()
        for layer_i in range(len(encode_layers) - 1):
            Layers["linear_{}".format(layer_i)] = nn.Linear(
                encode_layers[layer_i], encode_layers[layer_i + 1]
            )
            if layer_i != len(encode_layers) - 2:
                Layers["relu_{}".format(layer_i)] = nn.ReLU()
        self.encode_net = nn.Sequential(Layers)
        self.Nkoopman = Nkoopman
        self.u_dim = u_dim
        self.lA = nn.Linear(Nkoopman, Nkoopman, bias=False)
        self.lA.weight.data = gaussian_init_(Nkoopman, std=1)
        U, _, V = torch.svd(self.lA.weight.data)
        self.lA.weight.data = torch.mm(U, V.t()) * 0.9
        self.lB = nn.Linear(u_dim, Nkoopman, bias=False)

        self.lA_inv = nn.Linear(Nkoopman, Nkoopman, bias=False)
        self.lA_inv.weight.data = gaussian_init_(Nkoopman, std=1)
        Ui, _, Vi = torch.svd(self.lA_inv.weight.data)
        self.lA_inv.weight.data = torch.mm(U, V.t()) * 0.9

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

    def forward(self, x, u):
        return self.lA(x) + self.lB(u)

    def forward_inv(self, x):
        return self.lA_inv(x)



def K_fb_loss(data, net, Nstate=4, u_dim=1):
    steps, train_traj_num, Nstates = data.shape
    device = torch.device("cpu")
    data = torch.DoubleTensor(data).to(device)
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
        )  # (1,)
        mean_loss_list.append(
            torch.mean(torch.mean(torch.abs(Err), axis=0)).detach().cpu().numpy()
        )

    Z_current = net.encode(data[-1, :, u_dim:])  # [x,g(x)]
    for j in range(steps - 1, 0, -1):
        Z_current = net.forward_inv(Z_current - net.lB(data[j - 1, :, :u_dim]))
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
    data, net, mse_loss, u_dim=1, gamma=0.99, Nstate=4, aug_loss=0, detach=0
):
    steps, train_traj_num, NKoopman = data.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.DoubleTensor(data).to(device)
    Z_current = net.encode(data[0, :, u_dim:])
    beta, beta1 = 1.0, 1.0
    beta_sum, beta1_sum = 0.0, 0.0
    loss = torch.zeros(1, dtype=torch.float64).to(device)
    inv_loss, eye_loss = torch.zeros(1, dtype=torch.float64).to(device), torch.zeros(
        1, dtype=torch.float64
    ).to(device)
    for i in range(steps - 1):
        Z_current = net.forward(Z_current, data[i, :, :u_dim])
        beta_sum += beta
        if not aug_loss:
            loss += beta * mse_loss(Z_current[:, :Nstate], data[i + 1, :, u_dim:])
        else:
            Y = net.encode(data[i + 1, :, u_dim:])
            loss += beta * mse_loss(Z_current, Y)
        beta *= gamma

    Z_current = net.encode(data[-1, :, u_dim:])
    for j in range(steps - 1, 0, -1):
        Z_current = net.forward_inv(Z_current - net.lB(data[j - 1, :, :u_dim]))
        beta1_sum += beta1
        if not aug_loss:
            inv_loss += beta1 * mse_loss(Z_current[:, :Nstate], data[j - 1, :, u_dim:])
        else:
            inv_loss += beta1 * mse_loss(Z_current, net.encode(data[j - 1, :, u_dim:]))
        beta1 *= gamma

    lAw = net.lA.weight.data
    lAiw = net.lA_inv.weight.data
    eye = torch.eye(lAw.shape[0]).to(device)
    eye_loss = torch.linalg.matrix_norm(torch.mm(lAw, lAiw) - eye, ord="fro")

    return loss / beta_sum, inv_loss / beta1_sum, eye_loss


def Stable_loss(net, Nstate):
    x_ref = np.zeros(Nstate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_ref_lift = net.encode_only(torch.DoubleTensor(x_ref).to(device))
    loss = torch.norm(x_ref_lift)
    return loss


def Eig_loss(net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = net.lA.weight
    c = torch.linalg.eigvals(A).abs() - torch.ones(1, dtype=torch.float64).to(device)
    mask = c > 0
    loss = c[mask].sum()
    return loss


def train(
    env_name,
    train_steps=20000,
    suffix="",
    trained_suffix="",
    aug_loss=0,
    encode_dim=12,
    layer_depth=3,
    e_loss=1,
    gamma=0.5,
):
    set_seed(2023)
    Ktrain_samples = 10000
    Ktest_samples = 2500
    Ksteps = 100
    Kbatch_size = 1024
    res = 1
    normal = 1
    # data prepare
    data_collect = data_collecter(env_name)
    u_dim = data_collect.udim
    if os.path.isfile(str(DKO_DIR / "Data/{}test.npy".format(env_name))) and os.path.isfile(str(DKO_DIR / "Data/{}train.npy".format(env_name))):
        Ktest_data = np.load(str(DKO_DIR / "Data/{}test.npy".format(env_name)))
        Ktrain_data = np.load(str(DKO_DIR / "Data/{}train.npy".format(env_name)))
    else:
        Ktest_data = data_collect.collect_koopman_data(Ktest_samples, Ksteps)
        print("test data ok!")
        np.save(str(DKO_DIR / "Data/{}test.npy".format(env_name)), Ktest_data)
        Ktrain_data = data_collect.collect_koopman_data(Ktrain_samples, Ksteps)
        print("train data ok!")
        np.save(str(DKO_DIR / "Data/{}train.npy".format(env_name)), Ktrain_data)
    in_dim = Ktest_data.shape[-1] - u_dim
    Nstate = in_dim
    # layer_depth = 4
    layer_width = 128
    layers = [in_dim] + [layer_width] * layer_depth + [encode_dim]
    Nkoopman = in_dim + encode_dim
    print("layers:", layers)
    net = Network(layers, Nkoopman, u_dim)
    # print(net.named_modules())
    learning_rate = 1e-3
    if torch.cuda.is_available():
        net.cuda()
    net.double()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # train
    eval_step = 100
    best_loss = 1000.0
    best_Kloss = 1000.0
    best_inv_Kloss = 1000.0
    best_eyeloss = 1000.0
    best_state_dict = {}
    logdir = str(DKO_DIR / 
        "Data"
        / suffix
        / "KoopmanU_{}layer{}_edim{}_eloss{}_gamma{}_aloss{}".format(
            env_name, layer_depth, encode_dim, e_loss, gamma, aug_loss
        )
    )
    logdir_trained = str(DKO_DIR /
        "Data"
        / trained_suffix
        / "KoopmanIU_{}layer{}_edim{}_Naff{}_Nchange{}".format(
            env_name,
            layer_depth,
            encode_dim,
            1,
            1,
        )
    )
    if not os.path.exists(str(DKO_DIR / "Data" / suffix)):
        os.makedirs(str(DKO_DIR / "Data" / suffix))
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(log_dir=logdir)
    if os.path.isfile(logdir + ".pth"):
        print("=> loading checkpoint '{}'".format(logdir + ".pth"))
        checkpoint = torch.load(logdir + ".pth")
        net.load_state_dict(checkpoint["model"])
        for name, para in net.named_parameters():
            if name.startswith("lB") or name.startswith("encode_net"):
                para.requires_grad = False
            print("model:", name, para.requires_grad)
        start_step = checkpoint["step"]
    else:
        print("=> no checkpoint found at '{}'".format(logdir))
        start_step=0
        # load trained encoder
        trained_file = logdir_trained + ".pth"
        assert os.path.exists(trained_file), "trained file not found"
        dict_trained = torch.load(trained_file,map_location="cuda:0")["model"]
        dict_new = net.state_dict().copy()
        for name, para in net.named_parameters():
            if name.startswith("lB") or name.startswith("encode_net"):
                dict_new[name] = dict_trained[name]
        net.load_state_dict(dict_new)
        for name, para in net.named_parameters():
            if name.startswith("lB") or name.startswith("encode_net"):
                para.requires_grad = False
            print("model:", name, para.requires_grad)

    start_time = time.process_time()
    for i in range(start_step, train_steps):
        # K loss
        Kindex = list(range(Ktrain_samples))
        random.shuffle(Kindex)
        X = Ktrain_data[:, Kindex[:Kbatch_size], :]  # batch samples of K_steps
        optimizer.zero_grad()
        Kloss, inv_Kloss, eye_loss = Klinear_fb_loss(
            X, net, mse_loss, u_dim, gamma, Nstate, aug_loss
        )
        Eloss = Eig_loss(net)
        loss = (
            Kloss + inv_Kloss + 0.001 * eye_loss + Eloss
            if e_loss
            else Kloss + inv_Kloss + 0.001 * eye_loss
        )
        loss.backward()
        optimizer.step()
        writer.add_scalar("Train/loss", loss, i)
        writer.add_scalar("Train/Kloss", Kloss, i)
        writer.add_scalar("Train/inv_Kloss", inv_Kloss, i)
        writer.add_scalar("Train/eyeloss", eye_loss, i)
        if e_loss:
            writer.add_scalar("Train/Eloss", Eloss, i)
        # print("Step:{} Loss:{}".format(i,loss.detach().cpu().numpy()))
        if (i + 1) % eval_step == 0:
            # K loss
            with torch.no_grad():
                Kloss, inv_Kloss, eye_loss = Klinear_fb_loss(
                    Ktest_data, net, mse_loss, u_dim, gamma, Nstate, aug_loss=0
                )
                Eloss = Eig_loss(net)
                Kloss = Kloss.detach().cpu().numpy()
                inv_Kloss = inv_Kloss.detach().cpu().numpy()
                eye_loss = eye_loss.detach().cpu().numpy()
                Eloss = Eloss.detach().cpu().numpy()
                loss = (
                    Kloss + inv_Kloss + 0.001 * eye_loss + Eloss
                    if e_loss
                    else Kloss + inv_Kloss + 0.001 * eye_loss
                )

                writer.add_scalar("Eval/Kloss", Kloss, i)
                writer.add_scalar("Eval/inv_Kloss", inv_Kloss, i)
                writer.add_scalar("Eval/eyeloss", eye_loss, i)
                if e_loss:
                    writer.add_scalar("Eval/Eloss", Eloss, i)
                if (
                    loss < best_loss
                ):
                    best_Kloss, best_inv_Kloss, best_eyeloss = (
                        copy(Kloss),
                        copy(inv_Kloss),
                        copy(eye_loss),
                    )
                    best_state_dict = copy(net.state_dict())
                    Saved_dict = {"model": best_state_dict, "layer": layers,"step": i}
                    torch.save(Saved_dict, logdir + ".pth")
                    print(
                        "{} {} Step:{} Eval K-loss:{} inv-Kloss:{} eye-loss:{}".format(
                            time.asctime( time.localtime(time.time()) ), suffix, i, Kloss, inv_Kloss, 0.001*eye_loss
                        )
                    )
                # print("-------------END-------------")
        writer.add_scalar("Eval/best_Kloss", best_Kloss, i)
        writer.add_scalar("Eval/best_inv_Kloss", best_inv_Kloss, i)
        writer.add_scalar("Eval/best_eyeloss", best_eyeloss, i)
    print("END-best_Kloss{}".format(best_Kloss))
    print("END-best_inv_loss{}".format(best_inv_Kloss))


def main():
    train(
        args.env,
        suffix=args.suffix,
        trained_suffix=args.trained_suffix,
        aug_loss=args.aug_loss,
        encode_dim=args.encode_dim,
        layer_depth=args.layer_depth,
        e_loss=args.e_loss,
        gamma=args.gamma,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole")
    parser.add_argument("--suffix", type=str, default="CartPole0627")
    parser.add_argument("--trained_suffix", type=str, default="CartPole0627")
    parser.add_argument("--aug_loss", type=int, default=0)
    parser.add_argument("--e_loss", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--encode_dim", type=int, default=20)
    parser.add_argument("--layer_depth", type=int, default=3)
    args = parser.parse_args()
    main()
