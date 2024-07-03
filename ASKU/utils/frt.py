import numpy as np
import torch
from . import ell_proj_np
import cvxpy as cp

import torch
import torch


def sample_pts_unit_ball(dim, NB_pts):
    """
    Uniformly samples points in a d-dimensional sphere (in a ball)
    Points characterized by    ||x||_2 < 1
    arguments:  dim    - nb of dimensions
                NB_pts - nb of points
    output:     pts    - points sampled uniformly in ball [xdim x NB_pts]
    Reference: http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
    """

    us = np.random.normal(0, 1, (dim, NB_pts))
    norms = np.linalg.norm(us, 2, axis=0)
    rs = np.random.random(NB_pts) ** (1.0 / dim)
    pts = rs * us / norms

    return pts


def sample_pts_in_ellipsoid(mu, Q, NB_pts):
    """
    Uniformly samples points in an ellipsoid, specified as
            (xi-mu)^T Q^{-1} (xi-mu) <= 1
    arguments: mu - mean [dim]
                Q - Q [dim x dim]
    output:     pts - points sampled uniformly in ellipsoid [xdim x NB_pts]
    """
    xs = sample_pts_unit_ball(mu.shape[0], NB_pts)

    E = np.linalg.cholesky(Q)
    ys = (np.array(E @ xs).T + mu).T

    return ys


def Zs_dparams_MC(Zs, Zs_dZ, forward):
    """
    Returns the Jacobian matrices of the state TRAJECTORY
    w.r.t. to all parameters
    Inputs:  Xs       : (N_MC, N , n_x)
             Xs_dx    : (N_MC, N , n_x, n_x)
    Outputs: Xs_dX0   : (N_MC, N, n_x, n_x)

    """
    N_MC, T, NKoopman = Zs.shape[0], Zs.shape[1], Zs.shape[2]

    Zs_dZ0s = np.zeros((N_MC, T, NKoopman, NKoopman))
    if forward:
        Zs_dZ0s[:, 0, :, :] = np.repeat(np.eye(NKoopman)[None, :], N_MC, axis=0)
        for j in range(T - 1):
            # Jacobians w.r.t. Initial conditions
            Zs_dZ0s[:, j + 1, :, :] = np.einsum(
                "Mxy,Myz->Mxz", Zs_dZ[:, j, :, :], Zs_dZ0s[:, j, :, :]
            )
    else:
        Zs_dZ0s[:, -1, :, :] = np.repeat(np.eye(NKoopman)[None, :], N_MC, axis=0)
        for j in range(T - 1, 0, -1):
            # Jacobians w.r.t. Initial conditions
            Zs_dZ0s[:, j - 1, :, :] = np.einsum(
                "Mxy,Myz->Mxz", Zs_dZ[:, j - 1, :, :], Zs_dZ0s[:, j, :, :]
            )
    return Zs_dZ0s


def Xs_dparams_MC(Xs, Xs_dX, forward):
    """
    Returns the Jacobian matrices of the state TRAJECTORY
    w.r.t. to all parameters
    Inputs:  Xs       : (N_MC, N , n_x)
             Xs_dx    : (N_MC, N , n_x, n_x)
    Outputs: Xs_dX0   : (N_MC, N, n_x, n_x)

    """
    N_MC, T, Nstates = Xs.shape[0], Xs.shape[1], Xs.shape[2]

    Xs_dX0MC = np.zeros((N_MC, T, Nstates, Nstates))
    if forward:
        Xs_dX0MC[:, 0, :, :] = np.repeat(np.eye(Nstates)[None, :], N_MC, axis=0)
        for j in range(T - 1):
            # Jacobians w.r.t. Initial conditions
            Xs_dX0MC[:, j + 1, :, :] = np.einsum(
                "Mxy,Myz->Mxz", Xs_dX[:, j, :, :], Xs_dX0MC[:, j, :, :]
            )
    else:
        Xs_dX0MC[:, -1, :, :] = np.repeat(np.eye(Nstates)[None, :], N_MC, axis=0)
        for j in range(T - 1, 0, -1):
            # Jacobians w.r.t. Initial conditions
            Xs_dX0MC[:, j - 1, :, :] = np.einsum(
                "Mxy,Myz->Mxz", Xs_dX[:, j - 1, :, :], Xs_dX0MC[:, j, :, :]
            )
    return Xs_dX0MC




def adv_sample_params(org_sys, sys, Zs, Zs_dz, Zs_du, forward, device):
    """
    resamples parameters  self.X0s
          using           Zs       : (N_MC, T, NKoopman)
                          Zs_dz    : (N_MC, T, NKoopman, NKoopman)
    """
    N_MC, T, z_dim = Zs.shape[0], Zs.shape[1], Zs.shape[2]
    Nstates = org_sys.Nstates
    X0s = sys.z0_MC.cpu().numpy()[:, :Nstates]
    Us = sys.ctrl_MC.cpu().numpy()
    x0 = sys.z0.cpu().numpy()[:Nstates]
    Xs = Zs.copy()[:, :, :Nstates]
    Xs_dx = Zs_dz.copy()[:, :, :Nstates, :Nstates]
    Xs_du = Zs_du.copy()[:, :, :Nstates, :]

    Cs = np.mean(Zs, 0)[:, :Nstates]  # (T, z_dim)
    Qs = np.zeros((T, Nstates, Nstates))
    for t in range(1, T):
        Qs[t, :, :] = np.linalg.inv(np.cov(Xs[:, t, :].T))

    # compute cost gradient
    Jdists_dXs = np.einsum("txy,Mty->Mtx", 2 * Qs, Xs - Cs)

    # compute trajectory gradient w.r.t. parameters
    Xs_dX0 = Xs_dparams_MC(Xs, Xs_dx, forward)

    # compute cost gradient w.r.t params (average over horizon)
    Jdists_dX0MC = np.mean(np.einsum("MTx,MTxy->MTy", Jdists_dXs, Xs_dX0), axis=1)
    Jdists_dus = np.mean(
        np.einsum("MTx,MTxu->MTu", Jdists_dXs[:, 1:, :], Xs_du), axis=1
    )

    # gradient ascent
    X0s += sys.eta_z0s * Jdists_dX0MC
    Us += sys.eta_ctrl * Jdists_dus[:, None, :]

    # # project parameters
    x0s = np.repeat(x0[None, :], N_MC, axis=0)
    Q0_inv = np.linalg.inv(org_sys.Q)
    Q0s = np.repeat(Q0_inv[None, :, :], N_MC, axis=0)

    x0s_deltas = ell_proj_np.proj_ell(Q0s, X0s - x0s, eps=0.1)

    X0s += x0s_deltas

    X0s = np.clip(X0s, org_sys.low, -org_sys.low)
    Us = np.clip(Us, org_sys.umin, org_sys.umax)

    sys.z0_MC = sys.encode(torch.from_numpy(X0s).to(device))
    sys.ctrl_MC = torch.DoubleTensor(Us).to(device)
