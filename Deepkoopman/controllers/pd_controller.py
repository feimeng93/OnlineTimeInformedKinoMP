import sys
sys.path.append("..")
from numpy import dot
from .controller import Controller
from Deepkoopman.dynamics import ConfigurationDynamics


class KoopPdOutput(ConfigurationDynamics):
    def __init__(self, dynamics, xd, n, m):
        ConfigurationDynamics.__init__(self, dynamics, 1)
        self.xd = xd
        self.n = n
        self.m = m

    def proportional(self, x, t):
        q = x[: int(self.n / 2)]
        q_d = self.xd[: int(self.n / 2)]

        return q - q_d

    def derivative(self, x, t):
        q_dot = x[int(self.n / 2) :]
        q_dot_d = self.xd[int(self.n / 2) :]

        return q_dot - q_dot_d


class PDController(Controller):
    """Class for proportional-derivative policies."""

    def __init__(self, pd_dynamics, K_p, K_d):
        """Create a PDController object.

        Policy is u = -K_p * e_p - K_d * e_d, where e_p and e_d are propotional
        and derivative components of error.

        Inputs:
        Proportional-derivative dynamics, pd_dynamics: PDDynamics
        Proportional gain matrix, K_p: numpy array
        Derivative gain matrix, K_d: numpy array
        """

        Controller.__init__(self, pd_dynamics)
        self.K_p = K_p
        self.K_d = K_d

    def eval(self, x, t):
        e_p = self.dynamics.proportional(x, t)
        e_d = self.dynamics.derivative(x, t)
        return -dot(self.K_p, e_p) - dot(self.K_d, e_d)
