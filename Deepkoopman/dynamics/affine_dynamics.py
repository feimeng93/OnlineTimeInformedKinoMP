from numpy import dot, array
import torch
from .dynamics import Dynamics


class AffineDynamics(Dynamics):
    """Abstract class for dynamics of the form x_dot = f(x, t) + g(x, t) * u.

    Override eval, drift, act.
    """

    def drift(self, x, t):
        """Compute drift vector f(x, t).

        Inputs:
        State, x: numpy array
        Time, t: float

        Outputs:
        Drift vector: numpy array
        """

        pass

    def act(self, x, t):
        """Compute actuation matrix g(x, t).

        Inputs:
        State, x: numpy array
        Time, t: float

        Outputs:
        Actuation matrix: numpy array
        """

        pass

    def eval_dot(self, x, u, t):
        return self.drift(x, t) + dot(self.act(x, t), u)

    def batch_eval_dot(self, x, u, t):
        return self.batch_drift(x, t) + torch.matmul(
            self.batch_act(x, t), u.unsqueeze(-1)
        ).squeeze(-1)
