import random
from scipy.integrate import odeint
import scipy.linalg
import pybullet as pb
import pybullet_data
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import sys
import torch


sys.path.append("")


def set_seed(seed=0):
    """Set one seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


# data collect


class data_collecter:
    def __init__(self, env_name) -> None:
        self.env_name = env_name
        if self.env_name.startswith("Poly"):
            self.env = Poly_System()
            self.Nstates = self.env.Nstates
            self.umax = self.env.umax
            self.umin = self.env.umin
            self.udim = 1
        elif self.env_name.startswith("Linear"):
            self.env = Linear2D()
            self.Nstates = self.env.Nstates
            self.umax = self.env.umax
            self.umin = self.env.umin
            self.udim = 1
        elif self.env_name.startswith("Nonlinear"):
            self.env = Nonlinear3D()
            self.Nstates = self.env.Nstates
            self.umax = self.env.umax
            self.umin = self.env.umin
            self.udim = 1
        elif self.env_name.startswith("DampingPendulum"):
            self.env = SinglePendulum()
            self.Nstates = self.env.Nstates
            self.umax = self.env.umax
            self.umin = self.env.umin
            self.udim = 1
        elif self.env_name.startswith("TwoLinkRobot"):
            self.env = TwoLinkRobot()
            self.Nstates = self.env.Nstates
            self.umax = self.env.umax
            self.umin = self.env.umin
            self.udim = 2
        elif self.env_name.startswith("CartPole"):
            self.env = CartPoleEnv()
            self.udim = self.env.action_space.shape[0]
            self.Nstates = self.env.observation_space.shape[0]
            self.umin = self.env.action_space.low
            self.umax = self.env.action_space.high
        elif self.env_name.endswith("Franka"):
            from franka.franka_env import FrankaEnv

            self.env = FrankaEnv(render=False)
            self.Nstates = 17
            self.uval = 0.12
            self.udim = 7
            self.reset_joint_state = np.array(self.env.reset_joint_state)
        elif self.env_name.endswith("FrankaForce"):
            from franka.franka_env_force import FrankaEnv

            self.env = FrankaEnv(render=False)
            self.Nstates = 17
            self.uval = 20
            self.udim = 7
            self.reset_joint_state = np.array(self.env.reset_joint_state)
        elif self.env_name.startswith("PlanarQuadrotor"):
            from Deepkoopman.dynamics import (
                LinearLiftedDynamics,
                PlanarQuadrotorForceInput,
            )

            self.env = PlanarQuadrotorForceInput()
            # Data collection parameters:
            traj_length_dc = 1.0  # Trajectory length, data collection
            self.n_pred_dc = int(
                traj_length_dc / self.env.dt
            )  # Number of time steps, data collection
            self.t_eval = self.env.dt * np.arange(
                self.n_pred_dc + 1
            )  # Simulation time points
            self.n_traj_train = (
                5000  # Number of trajectories to execute, data collection
            )
            self.n_traj_val = int(0.25 * self.n_traj_train)
            self.xmax = np.array(
                [2, 2, np.pi / 3, 2.0, 2.0, 2.0]
            )  # State constraints, trajectory generation
            self.xmin = -self.xmax
            self.umax = np.array(
                [2 * self.env.hover_thrust, 2 * self.env.hover_thrust]
            )  # Actuation constraint, trajectory generation
            self.umin = np.array([0.0, 0.0])
            self.bounds = [[self.xmin, self.xmax], [self.umin, self.umax]]
            self.x0_max = np.array(
                [self.xmax[0], self.xmax[1], self.xmax[2], 1.0, 1.0, 1.0]
            )  # Initial value limits
            self.Nstates = self.env.n
            self.udim = self.env.m
            q_dc, r_dc = 5e2, 1  # State and actuation penalty values, data collection
            Q_dc = q_dc * np.identity(
                self.Nstates
            )  # State penalty matrix, data collection
            R_dc = r_dc * np.identity(
                self.udim
            )  # Actuation penalty matrix, data collection
            P_dc = scipy.linalg.solve_continuous_are(
                self.env.A_nom, self.env.B_nom, Q_dc, R_dc
            )  # Algebraic Ricatti equation solution, data collection
            K_dc = (
                np.linalg.inv(R_dc) @ self.env.B_nom.T @ P_dc
            )  # LQR feedback gain matrix, data collection
            self.K_dc_p = K_dc[
                :, : int(self.Nstates / 2)
            ]  # Proportional control gains, data collection
            self.K_dc_d = K_dc[
                :, int(self.Nstates / 2) :
            ]  # Derivative control gains, data collection
            self.nominal_sys = LinearLiftedDynamics(
                self.env.A_nom, None, self.env.B_nom, np.eye(self.Nstates), lambda x: x
            )

            self.noise_var = (
                5.0  # Exploration noise to perturb controller, data collection
            )

            self.QN_trajgen = scipy.sparse.diags(
                [5e1, 5e1, 5e1, 1e1, 1e1, 1e1]
            )  # Final state penalty matrix, trajectory generation
            self.R_trajgen = scipy.sparse.eye(
                self.udim
            )  # Actuation penalty matrix, trajectory generation
            self.env.observation_space = spaces.Box(
                self.xmax, self.xmin, dtype=np.float64
            )
        else:
            self.env = gym.make(env_name)
            self.udim = self.env.action_space.shape[0]
            self.Nstates = self.env.observation_space.shape[0]
            self.umin = self.env.action_space.low
            self.umax = self.env.action_space.high
        self.observation_space = self.env.observation_space
        # self.env.reset()
        self.dt = self.env.dt

    def collect_koopman_data(self, traj_num, steps):
        train_data = np.empty((steps + 1, traj_num, self.Nstates + self.udim))
        if self.env_name.startswith("Franka"):
            for traj_i in range(traj_num):
                noise = (np.random.rand(7) - 0.5) * 2 * 0.2
                joint_init = self.reset_joint_state + noise
                joint_init = np.clip(
                    joint_init, self.env.joint_low, self.env.joint_high
                )
                s0 = self.env.reset_state(joint_init)
                s0 = FrankaObs(s0)
                u10 = (np.random.rand(7) - 0.5) * 2 * self.uval
                train_data[0, traj_i, :] = np.concatenate(
                    [u10.reshape(-1), s0.reshape(-1)], axis=0
                ).reshape(-1)
                for i in range(1, steps + 1):
                    s0 = self.env.step(u10)
                    s0 = FrankaObs(s0)
                    u10 = (np.random.rand(7) - 0.5) * 2 * self.uval
                    train_data[i, traj_i, :] = np.concatenate(
                        [u10.reshape(-1), s0.reshape(-1)], axis=0
                    ).reshape(-1)
        elif self.env_name.startswith("PlanarQuad"):
            from Deepkoopman.controllers import (
                MPCController,
                PDController,
                PerturbedController,
            )
            from Deepkoopman.dynamics import QuadrotorPdOutput

            xd = np.empty((traj_num, steps + 1, self.Nstates))
            xs = np.empty((traj_num, steps + 1, self.Nstates))
            us = np.empty((traj_num, steps, self.udim))

            for ii in range(traj_num):
                x0 = np.asarray(
                    [random.uniform(l, u) for l, u in zip(-self.x0_max, self.x0_max)]
                )
                set_pt_dc = np.asarray(
                    [random.uniform(l, u) for l, u in zip(-self.x0_max, self.x0_max)]
                )
                mpc_trajgen = MPCController(
                    self.nominal_sys,
                    steps,
                    self.dt,
                    self.umin,
                    self.umax,
                    self.xmin,
                    self.xmax,
                    self.QN_trajgen,
                    self.R_trajgen,
                    self.QN_trajgen,
                    set_pt_dc,
                )
                mpc_trajgen.eval(x0, 0)
                xd[ii, :, :] = mpc_trajgen.parse_result().T
                while abs(x0[0]) + abs(x0[1]) < 1 or np.any(np.isnan(xd[ii, :, :])):
                    x0 = np.asarray(
                        [
                            random.uniform(l, u)
                            for l, u in zip(-self.x0_max, self.x0_max)
                        ]
                    )
                    set_pt_dc = np.asarray(
                        [
                            random.uniform(l, u)
                            for l, u in zip(-self.x0_max, self.x0_max)
                        ]
                    )
                    mpc_trajgen = MPCController(
                        self.nominal_sys,
                        steps,
                        self.dt,
                        self.umin - self.env.hover_thrust,
                        self.umax - self.env.hover_thrust,
                        self.xmin,
                        self.xmax,
                        self.QN_trajgen,
                        self.R_trajgen,
                        self.QN_trajgen,
                        set_pt_dc,
                    )
                    mpc_trajgen.eval(x0, 0)
                    xd[ii, :, :] = mpc_trajgen.parse_result().T

                output = QuadrotorPdOutput(
                    self.env, xd[ii, :, :], self.t_eval, self.Nstates, self.udim
                )
                pd_controller = PDController(output, self.K_dc_p, self.K_dc_d)
                perturbed_pd_controller = PerturbedController(
                    self.env,
                    pd_controller,
                    self.noise_var,
                    const_offset=self.env.hover_thrust,
                )
                xs[ii, :, :], us[ii, :, :] = self.env.simulate(
                    x0, perturbed_pd_controller, self.t_eval
                )
            train_data = np.concatenate(
                [xs, np.concatenate([us, np.zeros((traj_num, 1, self.udim))], axis=1)],
                axis=2,
            )
        else:
            for traj_i in range(traj_num):
                s0 = self.env.reset()
                # s0 = self.random_state()
                u10 = np.random.uniform(self.umin, self.umax)
                # self.env.reset_state(s0)
                train_data[0, traj_i, :] = np.concatenate(
                    [u10.reshape(-1), s0.reshape(-1)], axis=0
                ).reshape(-1)
                for i in range(1, steps + 1):
                    s0, r, done, _ = self.env.step(u10)
                    u10 = np.random.uniform(self.umin, self.umax)
                    train_data[i, traj_i, :] = np.concatenate(
                        [u10.reshape(-1), s0.reshape(-1)], axis=0
                    ).reshape(-1)
        return train_data

    def load_data(self, data_dir):
        pass


class Nonlinear3D:
    def __init__(self) -> None:
        self.dt = 0.05
        self.s0 = np.array([1, 0.5, -0.2])
        self.Nstates = 3
        self.u_dim = 1
        self.umin = np.array([-0.2])
        self.umax = np.array([0.2])
        self.low = np.array([-2.0, -2.0, -2.0], dtype=np.float32)
        self.observation_space = spaces.Box(self.low, -self.low, dtype=np.float32)
        self.x0_MC = None
        self.Q= None

    def reset(self):
        x1 = random.uniform(0.5, 1.5)
        x2 = random.uniform(0.0, 1.0)
        x3 = random.uniform(-0.5, 1.0)
        self.s0 = np.array([x1, x2, x3])
        return self.s0

    def reset_state(self, s):
        self.s0 = s
        return self.s0

    def dynamics(self, y, t, u):
        x1, x2, x3 = y
        # f = asarray([dtheta, -g/l * sin(theta) +  u*cos(theta)/(m*l)])
        f = np.asarray([x3, -x1 + x1**3 / 6 - x3, u])
        return f

    def step(self, u):
        u = np.array(u).reshape(1)
        sn = odeint(self.dynamics, self.s0, [0, self.dt], args=(u[0],))
        self.s0 = sn[-1, :]
        r = 0
        done = False

        return self.s0, r, done, {}


class Linear2D:
    def __init__(self) -> None:
        self.dt = 0.05
        self.s0 = np.array([-2, 1])
        self.Nstates = 2
        self.u_dim = 1
        self.umin = np.array([-0.5])
        self.umax = np.array([0.5])
        self.low = np.array([-3.0, -3.0], dtype=np.float32)
        self.observation_space = spaces.Box(self.low, -self.low, dtype=np.float32)
        self.Q = None
        self.x0_MC = None

    def reset(self):
        x1 = random.uniform(-3.0, 3.0)
        x2 = random.uniform(-3.0, 3.0)
        self.s0 = np.array([x1, x2])
        return self.s0

    def reset_state(self, s):
        self.s0 = s
        return self.s0

    def dynamics(self, y, t, u):
        x1, x2 = y
        # f = asarray([dtheta, -g/l * sin(theta) +  u*cos(theta)/(m*l)])
        f = np.array([0.5 * x2, -0.1 * x1 + 0.2 * x2 + u])
        return f

    def step(self, u):
        u = np.array(u).reshape(1)
        # sn = odeint(self.dynamics, self.s0, [0, self.dt], args=(u[0],))
        dx1 = 0.5 * self.s0[1]
        dx2 = -0.1 * self.s0[0] + 0.2 * self.s0[1] + u[0]
        dx = np.array([dx1, dx2])
        self.s0 = np.add(self.s0, dx * self.dt)
        r = 0
        done = False

        return self.s0, r, done, {}

    def step_inv(self, u):
        u = np.array(u).reshape(1)
        sn = self.s0 - self.dt * (
            np.dot(np.array([[0, 0.5], [-0.1, 0.2]]), self.s0)
            + np.dot(np.array([[0], [1]]), u)
        )
        self.s0 = sn
        r = 0
        done = False

        return self.s0, r, done, {}


class Poly_System:
    def __init__(self) -> None:
        self.dt = 0.01
        self.s0 = np.array([1.2, 1.1])
        self.Nstates = 2
        self.umin = np.array([0])
        self.umax = np.array([0.2])
        low = np.array([-2.0, -2.0], dtype=np.float32)
        self.observation_space = spaces.Box(low, -low, dtype=np.float32)
        self.x0_MC = None

    def reset(self):
        x1 = random.uniform(0, 1.5)
        x2 = random.uniform(0, 1.5)
        self.s0 = np.array([x1, x2])
        return self.s0

    def reset_state(self, s):
        self.s0 = s
        return self.s0

    def dynamics(self, y, t, u):
        x1, x2 = y
        # f = asarray([dtheta, -g/l * sin(theta) +  u*cos(theta)/(m*l)])
        f = np.asarray([(u + 3) * x1 * (1 - x2), (u + 3) * x2 * (x1 - 1)])
        return f

    def step(self, u):
        u = np.array(u).reshape(1)
        sn = odeint(self.dynamics, self.s0, [0, self.dt], args=(u[0],))
        self.s0 = sn[-1, :]
        r = 0
        done = False

        return self.s0, r, done, {}


class TwoLinkRobot:
    def __init__(self) -> None:
        self.g = 9.8
        self.l1 = 1.0
        self.l2 = 1.0
        self.m1 = 1.0
        self.m2 = 1.0
        self.dt = 0.01
        self.s0 = np.zeros(4)
        self.Nstates = 4
        self.umin = np.array([-6, -6])
        self.umax = np.array([6, 6])
        self.b = 2
        self.low = np.array([-np.pi, -np.pi, -8, -8], dtype=np.float32)
        self.observation_space = spaces.Box(self.low, -self.low, dtype=np.float32)
        self.Q = None
        self.x0_MC = None

    def reset(self):
        th0 = random.uniform(-0.1 * np.pi, 0.1 * np.pi)
        dth0 = random.uniform(-1, 1)
        th1 = random.uniform(-0.1 * np.pi, 0.1 * np.pi)
        dth1 = random.uniform(-1, 1)
        self.s0 = np.array([th0, th1, dth0, dth1])
        return self.s0

    def reset_state(self, s):
        self.s0 = s
        return self.s0

    def dynamics(self, y, t, u):
        th1, th2, dth1, dth2 = y
        u = np.array(u).reshape(2, 1)
        f = np.zeros(4)
        g = self.g
        l1 = self.l1
        l2 = self.l2
        m1 = self.m1
        m2 = self.m2
        c2 = np.cos(th2)
        s2 = np.sin(th2)
        M = np.zeros((2, 2))
        M[0, 0] = m1 * l1**2 + m2 * (l1**2 + 2 * l1 * l2 * c2 + l2**2)
        M[0, 1] = m2 * (l1 * l2 * c2 + l2**2)
        M[1, 0] = m2 * (l1 * l2 * c2 + l2**2)
        M[1, 1] = m2 * l2**2
        C = np.zeros((2, 1))
        C[0, 0] = -m2 * l1 * l2 * s2 * (2 * dth1 * dth2 + dth2**2)
        C[1, 0] = m2 * l1 * l2 * dth1**2 * s2
        G = np.zeros((2, 1))
        G[0, 0] = (m1 + m2) * l1 * g * np.cos(th1) + m2 * g * l2 * np.cos(th1 + th2)
        G[1, 0] = m2 * g * l2 * np.cos(th1 + th2)
        Minv = scipy.linalg.pinv(M)
        ddth = np.dot(Minv, (u - C - G)).reshape(-1)
        f[0] = dth1
        f[1] = dth2
        f[2] = ddth[0]
        f[3] = ddth[1]
        return f

    def step(self, u):
        sn = odeint(self.dynamics, self.s0, [0, self.dt], args=(u,))
        self.s0 = sn[-1, :]
        r = 0
        done = False
        return self.s0, r, done, {}


class SinglePendulum:
    def __init__(self) -> None:
        self.g = 9.8
        self.l = 1.0
        self.m = 1.0
        self.dt = 0.02
        self.s0 = np.zeros(2)
        self.Nstates = 2
        self.umin = np.array([-8])
        self.umax = np.array([8])
        self.b = 1
        self.low = np.array([-np.pi, -8], dtype=np.float32)
        self.observation_space = spaces.Box(self.low, -self.low, dtype=np.float32)
        self.Q = None
        self.x0_MC = None

    def reset(self):
        th0 = random.uniform(-0.1 * np.pi, 0.1 * np.pi)
        dth0 = random.uniform(-1, 1)
        self.s0 = np.array([th0, dth0])
        return self.s0

    def reset_state(self, s):
        self.s0 = s
        return self.s0

    def single_pendulum(self, y, t, u):
        theta, dtheta = y
        # f = asarray([dtheta, -g/l * sin(theta) +  u*cos(theta)/(m*l)])
        f = np.asarray(
            [
                dtheta,
                -self.g / self.l * np.sin(theta)
                - self.b * self.l * dtheta / self.m
                + np.cos(theta) * u / (self.m * self.l),
            ]
        )
        return f

    def step(self, u):
        u = np.array(u).reshape(1)
        sn = odeint(self.single_pendulum, self.s0, [0, self.dt], args=(u[0],))
        self.s0 = sn[-1, :]
        r = 0
        done = False

        return self.s0, r, done, {}


def FrankaObs(o):
    return np.concatenate((o[:3], o[7:]), axis=0)


"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""


class CartPoleEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right

        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):
        self.g = 9.8
        self.M = 10
        self.m = 5
        self.total_mass = self.M + self.m
        self.L = 2.5  # actually half the pole's length
        self.I = 10
        self.polemass_length = self.m * self.L
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"
        self.max_torque = 300
        self.dt = self.tau
        self.Q = None
        self.x0_MC = None
        self.umin = np.array([-self.max_torque])
        self.umax = np.array([self.max_torque])
        self.Nstates = 4

        # Angle at which to fail the episode
        # self.theta_threshold_radians = 12 * 2 * math.pi / 360
        # self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        self.low = np.array(
            [
                -30,  # 4.8
                -10,
                -np.pi,  # 0.418 rad
                -2,
            ],
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(self.low, -self.low, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # err_msg = "%r (%s) invalid" % (action, type(action))
        # assert self.action_space.contains(action), err_msg

        STATE_X = 0
        STATE_V = 1 
        STATE_THETA = 2
        STATE_W = 3
        deriv = np.zeros_like(self.state)
        _v = self.state[STATE_V]
        _w = self.state[STATE_W]
        _theta = self.state[STATE_THETA]
        mass_term = (self.M + self.m) * (self.I + self.m * self.L * self.L) - self.m * self.m * self.L * self.L * np.cos(_theta) * np.cos(_theta)
    
        deriv[STATE_X] = _v
        deriv[STATE_THETA] = _w
        mass_term = 1.0 / mass_term
        deriv[STATE_V] = ((self.I + self.m * self.L * self.L) * (action + self.m * self.L * _w * _w * np.sin(_theta)) + self.m * self.m * self.L * self.L * np.cos(_theta) * np.sin(_theta) * self.g) * mass_term
        deriv[STATE_W] = ((-self.m * self.L * np.cos(_theta)) * (action + self.m * self.L * _w * _w * np.sin(_theta)) + (self.M + self.m) * (-self.m * self.g * self.L * np.sin(_theta))) * mass_term


        # self.state = (x, x_dot[0], theta, theta_dot[0])
        self.state += self.dt * deriv


        done = bool(
            self.state[0] < -30
            or self.state[0] > 30
            or self.state[2] < -np.pi
            or self.state[2] > np.pi
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)

    def reset_state(self, s):
        self.state = s
        return np.array(self.state, dtype=np.float32)

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


from matplotlib.pyplot import (
    figure,
    grid,
    legend,
    plot,
    show,
    subplot,
    suptitle,
    title,
    savefig,
    ylim,
    ylabel,
    xlabel,
)
from numpy import array, gradient, zeros, tile
import numpy as np
import torch
from torch.nn.utils import prune


def differentiate_vec(xs, ts):
    """differentiate_vec Numerically differencitate a vector

    Arguments:
        xs {numpy array [Nt,Ns]} -- state as a block matrix
        ts {numpy array [Nt,]} -- time vecotr

    Keyword Arguments:
        L {integer} -- differenciation order, only L=3 (default: {3})

    Returns:
        numpy array [Nt,Ns] -- numerical derivative
    """
    assert xs.shape[0] == ts.shape[0]
    return array(
        [differentiate(xs[:, ii], ts) for ii in range(xs.shape[1])]
    ).transpose()


def fit_standardizer(data, standardizer, flattened=False):
    if flattened:
        data_flat = data
    else:
        n_traj, traj_length, n = data.shape
        data_flat = data.T.reshape((n, n_traj * traj_length), order="F").T

    standardizer.fit(data_flat)

    return standardizer


def plot_trajectory(X, X_d, U, U_nom, t, display=True, save=False, filename=""):
    """Plots the position, velocity and control input

    # Inputs:
    - state X, numpy 2d array [number of time steps 'N', number of states 'n']
    - desired state X_d, numpy 2d array [number of time steps 'N', number of states 'n']
    - control input U, numpy 2d array [number of time steps, number of inputs]
    - nominal control input U_nom, numpy 2d array [number of time steps, number of inputs]
    - time t, numpy 1d array [number of time steps 'N']
    """
    figure()
    subplot(2, 1, 1)
    plot(t, X[:, 0], linewidth=2, label="$x$")
    plot(t, X[:, 2], linewidth=2, label="$\\dot{x}$")
    plot(t, X_d[:, 0], "--", linewidth=2, label="$x_d$")
    plot(t, X_d[:, 2], "--", linewidth=2, label="$\\dot{x}_d$")
    title("Trajectory Tracking with PD controller")
    legend(fontsize=12)
    grid()
    subplot(2, 1, 2)
    plot(t[:-1], U[:, 0], label="$u$")
    plot(t[:-1], U_nom[:, 0], label="$u_{nom}$")
    legend(fontsize=12)
    grid()
    if display:
        show()
    if save:
        savefig(filename)


def plot_trajectory_ep(
    X, X_d, U, U_nom, t, display=True, save=False, filename="", episode=0
):
    # Plot the first simulated trajectory
    figure(figsize=(4.7, 5.5))
    subplot(3, 1, 1)
    title("Trajectory tracking with MPC, episode " + str(episode))
    plot(t, X[0, :], linewidth=2, label="$x$")
    plot(t, X[2, :], linewidth=2, label="$\\dot{x}$")
    plot(t, X_d[0, :], "--", linewidth=2, label="$x_d$")
    plot(t, X_d[2, :], "--", linewidth=2, label="$\\dot{x}_d$")
    legend(fontsize=10, loc="lower right", ncol=4)
    ylim((-4.5, 2.5))
    ylabel("$x$, $\\dot{x}$")
    grid()
    subplot(3, 1, 2)
    plot(t, X[1, :], linewidth=2, label="$\\theta$")
    plot(t, X[3, :], linewidth=2, label="$\\dot{\\theta}$")
    plot(t, X_d[1, :], "--", linewidth=2, label="$\\theta_d$")
    plot(t, X_d[3, :], "--", linewidth=2, label="$\\dot{\\theta}_d$")
    legend(fontsize=10, loc="lower right", ncol=4)
    ylim((-2.25, 1.25))
    ylabel("$\\theta$, $\\dot{\\theta}$")
    grid()

    subplot(3, 1, 3)
    plot(t[:-1], U[0, :], label="$u$")
    plot(t[:-1], U_nom[0, :], label="$u_{nom}$")
    legend(fontsize=10, loc="upper right", ncol=2)
    ylabel("u")
    xlabel("Time (sec)")
    grid()
    if save:
        savefig(filename)
    if display:
        show()


def rbf(X, C, type="gauss", eps=1.0):
    """rbf Radial Basis Function

    Arguments:
        X {numpy array [Ns,Nz]} -- state
        C {numpy array [Ns,Nc]} -- centers.

    Keyword Arguments:
        type {str} -- RBF type (default: {'gauss'})
        eps {float} -- epsilon for gauss (default: {1.})

    Returns:
        numpy array [] -- [description]
    """
    N = X.shape[1]
    n = X.shape[0]
    Cbig = C
    Y = zeros((C.shape[1], N))
    for ii in range(C.shape[1]):
        C = Cbig[:, ii]
        C = tile(C.reshape((C.size, 1)), (1, N))
        r_sq = np.sum((X - C) ** 2, axis=0)
        if type == "gauss":
            y = np.exp(-(eps**2) * r_sq)

        Y[ii, :] = y

    return Y


def calc_koopman_modes(A, output, x_0, t_eval):
    d_w, w = np.linalg.eig(A.T)
    d, v = np.linalg.eig(A)

    sort_ind_w = np.argsort(np.abs(d_w))
    w = w[:, sort_ind_w]
    d_w = d_w[sort_ind_w]

    sort_ind_v = np.argsort(np.abs(d))
    v = v[:, sort_ind_v]
    d = d[sort_ind_v]

    non_zero_cols = np.where(np.diag(np.dot(w.T, v)) > 0)
    w = w[:, non_zero_cols].squeeze()
    v = v[:, non_zero_cols].squeeze()
    d = d[non_zero_cols].squeeze()

    eigfuncs = lambda x, t: np.divide(
        np.dot(w.T, output(x, t)), np.diag(np.dot(w.T, v))
    )
    eigvals = np.exp(d)

    koop_mode = lambda t: [
        eigvals[ii] ** t * eigfuncs(x_0, t)[ii] * v[:, ii] for ii in range(d.size)
    ]
    xs_koop = array(
        [koop_mode(t) for t in t_eval]
    )  # Evolution of each mode [n_time, n_modes, n_outputs]

    return xs_koop, v, w, d


def calc_reduced_mdl(model):
    A = model.A
    C = model.C
    useful_rows = np.argwhere(np.abs(C) > 0)
    useful_rows = np.unique(useful_rows[:, 1])
    useful_inds = np.argwhere(np.abs(A[useful_rows, :]) > 0)
    useful_cols = np.unique(useful_inds[:, 1])
    useful_coords = np.unique(np.concatenate((useful_rows, useful_cols)))

    A_red = model.A[useful_coords, :]
    A_red = A_red[:, useful_coords]
    if model.B is not None:
        B_red = model.B[useful_coords, :]
    else:
        B_red = None
    C_red = C[:, useful_coords]

    return A_red, B_red, C_red, useful_coords


class ThresholdPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, threshold):
        self.threshold = threshold

    def compute_mask(self, tensor, default_mask):
        return torch.abs(tensor) > self.threshold
