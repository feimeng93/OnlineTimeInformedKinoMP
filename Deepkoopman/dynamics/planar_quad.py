from . import ConfigurationDynamics, RoboticDynamics
from numpy import (
    pi,
    append,
    arange,
    dot,
    array,
    concatenate,
    cos,
    reshape,
    sin,
    zeros,
    interp,
    eye,
    gradient,
)
from matplotlib.pyplot import figure
from gym import spaces


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


def differentiate(xs, ts):
    """differentiate     Compute the discrete derivative of a Python function
    f on [a,b] using n intervals. Internal points apply
    a centered difference, while end points apply a one-sided
    difference. Vectorized version.


    Arguments:
        xs {numpy array [Nt,]} -- state as a vector
        ts {numpy array [Nt,]} -- time vecotr

    Returns:
        numpy array [Nt,] -- numerical derivative
    """

    dt = ts[1] - ts[0]
    dx = gradient(xs, dt, edge_order=2)
    return dx


def default_fig(fig, ax):
    if fig is None:
        fig = figure(figsize=(6, 6), tight_layout=True)

    if ax is None:
        ax = fig.add_subplot(1, 1, 1)

    return fig, ax


class PlanarQuadrotorForceInput(RoboticDynamics):
    def __init__(self, m=2.0, J=1.0, b=0.2, g=9.81):
        RoboticDynamics.__init__(self, 3, 2)
        # Linearized system specification:
        self.n, self.m = 6, 2  # Number of states, number of control inputs
        self.A_nom = array(
            [
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                ],  # Linearization of the true system around the origin
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, -g, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        self.B_nom = array(
            [
                [0.0, 0.0],  # Linearization of the true system around the origin
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0 / m, 1.0 / m],
                [-b / J, b / J],
            ]
        )

        self.hover_thrust = m * g / m  # planar quadrotor system parameters
        self.dt = 1.0e-2  # Time step length
        self.params = m, J, b, g
        self.standardizer_u = None
        self.standardizer_x = None
        self.observation_space = None
        self.low = -array([2, 2, pi / 3, 2.0, 2.0, 2.0])
        self.umax = array(
                [2 * self.hover_thrust, 2 * self.hover_thrust]
            )
        self.umin = array([0.0, 0.0])
        self.Nstates= self.n
        
        

    def reset(self):
        pass

    def D(self, q):
        m, J, b, _ = self.params
        return array([[m, 0, 0], [0, m, 0], [0, 0, J / b]])

    def C(self, q, q_dot):
        return zeros((3, 3))

    def U(self, q):
        m, _, _, g = self.params
        _, z, _ = q
        return m * g * z

    def G(self, q):
        m, _, _, g = self.params
        return array([0, m * g, 0])

    def B(self, q):
        _, _, theta = q
        return array([[-sin(theta), -sin(theta)], [cos(theta), cos(theta)], [-1, 1]])

    def plot_coordinates(self, ts, qs, fig=None, ax=None, labels=None):
        if fig is None:
            fig = figure(figsize=(6, 6), tight_layout=True)

        if ax is None:
            ax = fig.add_subplot(1, 1, 1, projection="3d")

        xs, zs, thetas = qs.T

        ax.set_title("Coordinates", fontsize=16)
        ax.set_xlabel("$x$ (m)", fontsize=16)
        ax.set_ylabel("$\\theta$ (rad)", fontsize=16)
        ax.set_zlabel("$z$ (m)", fontsize=16)
        ax.plot(xs, thetas, zs, linewidth=3)

        return fig, ax

    def plot_states(self, ts, xs, fig=None, ax=None, labels=None):
        fig, ax = default_fig(fig, ax)

        ax.set_title("States", fontsize=16)
        ax.set_xlabel("$q$", fontsize=16)
        ax.set_ylabel("$\\dot{q}$", fontsize=16)
        ax.plot(xs[:, 0], xs[:, 3], linewidth=3, label="$x$ (m)")
        ax.plot(xs[:, 1], xs[:, 4], linewidth=3, label="$z$ (m)")
        ax.plot(xs[:, 2], xs[:, 5], linewidth=3, label="$\\theta$ (rad)")
        ax.legend(fontsize=16)

        return fig, ax

    def plot_actions(self, ts, us, fig=None, ax=None, labels=None):
        fig, ax = default_fig(fig, ax)

        if labels is None:
            labels = ["$f$ (N)", "$\\tau$ (N $\\cdot$ m)"]

        ax.set_title("Actions", fontsize=16)
        ax.set_xlabel(labels[0], fontsize=16)
        ax.set_ylabel(labels[1], fontsize=16)
        ax.plot(*us.T, linewidth=3)

        return fig, ax

    def plot_tangents(self, ts, xs, fig=None, ax=None, skip=1):
        fig, ax = default_fig(fig, ax)

        ax.set_title("Tangent Vectors", fontsize=16)
        ax.set_xlabel("$x$ (m)", fontsize=16)
        ax.set_ylabel("$z$ (m)", fontsize=16)
        ax.plot(*xs[:, :2].T, linewidth=3)
        ax.quiver(*xs[::skip, :2].T, *xs[::skip, 3:5].T, angles="xy")

        return fig, ax

    def plot_physical(self, ts, xs, fig=None, ax=None, skip=1):
        fig, ax = default_fig(fig, ax)

        xs, zs, thetas = xs[:, :3].T
        dirs = array([sin(thetas), cos(thetas)])[:, ::skip]

        ax.set_title("Physical Space", fontsize=16)
        ax.set_xlabel("$x$ (m)", fontsize=16)
        ax.set_ylabel("$z$ (m)", fontsize=16)
        ax.quiver(xs[::skip], zs[::skip], *dirs, angles="xy")
        ax.plot(xs, zs, linewidth=3)
        ax.axis("equal")

        return fig, ax

    def plot(self, xs, us, ts, fig=None, action_labels=None, skip=1):
        if fig is None:
            fig = figure(figsize=(12, 6), tight_layout=True)

        physical_ax = fig.add_subplot(1, 2, 1)
        fig, physical_ax = self.plot_physical(ts, xs, fig, physical_ax, skip)

        action_ax = fig.add_subplot(1, 2, 2)
        fig, action_ax = self.plot_actions(ts, us, fig, action_ax, action_labels)

        return fig, (physical_ax, action_ax)


class PlanarQuadrotorForceInputDiscrete(PlanarQuadrotorForceInput):
    def __init__(self, mass=2.0, inertia=1.0, prop_arm=0.2, g=9.81, dt=1e-2):
        PlanarQuadrotorForceInput.__init__(self, mass, inertia, prop_arm, g=g)
        self.dt = dt

    def eval_dot(self, x, u, t):
        return x + self.dt * self.drift(x, t) + self.dt * dot(self.act(x, t), u)

    def get_linearization(self, x0, x1, u0, t):
        m, J, b, g = self.params
        A_lin = eye(self.n) + self.dt * array(
            [
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [
                    0,
                    0,
                    -(1 / m) * cos(x0[2]) * u0[0] - (1 / m) * cos(x0[2]) * u0[1],
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    -(1 / m) * sin(x0[2]) * u0[0] - (1 / m) * sin(x0[2]) * u0[1],
                    0,
                    0,
                    0,
                ],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        B_lin = self.dt * array(
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [-(1 / m) * sin(x0[2]), -(1 / m) * sin(x0[2])],
                [(1 / m) * cos(x0[2]), (1 / m) * cos(x0[2])],
                [-b / J, b / J],
            ]
        )

        if x1 is None:
            x1 = A_lin @ x0 + B_lin @ u0

        f_d = self.eval_dot(x0, u0, t)
        r_lin = f_d - x1

        return A_lin, B_lin, r_lin


class QuadrotorPdOutput(ConfigurationDynamics):
    def __init__(self, dynamics, xd, t_d, n, m):
        ConfigurationDynamics.__init__(self, dynamics, 1)
        self.xd = xd
        self.t_d = t_d
        self.xd_dot = differentiate_vec(self.xd, self.t_d)
        self.n = n
        self.m = m

    def proportional(self, x, t):
        q, q_dot = x[: int(self.n / 2)], x[int(self.n / 2) :]
        return self.y(q) - self.y_d(t)

    def derivative(self, x, t):
        q, q_dot = x[: int(self.n / 2)], x[int(self.n / 2) :]
        return self.dydq(q) @ q_dot - self.y_d_dot(t)

    def y(self, q):
        return q

    def dydq(self, q):
        return eye(int(self.n / 2))

    def d2ydq2(self, q):
        return zeros((int(self.n / 2), int(self.n / 2), int(self.n / 2)))

    def y_d(self, t):
        return self.desired_state_(t)[: int(self.n / 2)]

    def y_d_dot(self, t):
        return self.desired_state_(t)[int(self.n / 2) :]

    def y_d_ddot(self, t):
        return self.desired_state_dot_(t)[int(self.n / 2) :]

    def desired_state_(self, t):
        return [
            interp(t, self.t_d.flatten(), self.xd[:, ii].flatten())
            for ii in range(self.xd.shape[1])
        ]

    def desired_state_dot_(self, t):
        return [
            interp(t, self.t_d.flatten(), self.xd_dot[:, ii].flatten())
            for ii in range(self.xd_dot.shape[1])
        ]
