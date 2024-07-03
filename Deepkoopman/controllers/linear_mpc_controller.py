import cvxpy as cvx
import time
from numba import jit
from controller import Controller
import numpy as np
import time 
from scipy import sparse


@jit(nopython=True)
def _update_current_sol(z_init, dz_flat, u_init, du_flat, u_init_flat, nx, nu, N):
    cur_z = z_init + dz_flat.reshape(N + 1, nx)
    cur_u = u_init + du_flat.reshape(N, nu)
    u_init_flat = u_init_flat + du_flat

    return cur_z, cur_u, u_init_flat


@jit(nopython=True)
def _update_objective(C, Q, QN, R, xr, x_init, u_init, const_offset, nx, nu, N):
    """
    Construct MPC objective function
    :return:
    """
    res = np.empty(nx*(N+1)+nu*N)
    res[nx*N:nx*(N+1)] = C.T @ QN @ (x_init[:, -1] - xr)
    for ii in range(N):
        res[ii*nx:(ii+1)*nx] = C.T @ Q @ (x_init[:, ii] - xr)
        res[(N+1)*nx + nu*ii:(N+1)*nx + nu*(ii+1)] = R @ (u_init[ii, :] + const_offset)

    return res


class LinearMPCController(Controller):

    def __init__(self, n, m, n_lift, n_pred, linear_dynamics, xmin, xmax, umin, umax, Q, Q_n, R, set_pt, const_offset=0.):

        super(LinearMPCController, self).__init__(linear_dynamics)
        self.n = n
        self.m = m
        self.n_lift = n_lift
        self.n_pred = n_pred
        self.linear_dynamics = linear_dynamics
        self.xmin = xmin
        self.xmax = xmax
        self.umin = umin
        self.umax = umax
        self.Q = Q
        self.Q_n = Q_n
        self.R = R
        self.set_pt = set_pt
        self.const_offset=const_offset

        self.mpc_prob = None
        self.x_init = None
        self.comp_time = []

    def construct_controller(self):
        u = cvx.Variable((self.m, self.n_pred))
        x = cvx.Variable((self.n, self.n_pred+1))
        self.x_init = cvx.Parameter(self.n)
        objective = 0
        constraints = [x[:,0] == self.x_init]

        for k in range(self.n_pred):
            objective += cvx.quad_form(x[:,k] - self.set_pt, self.Q) + cvx.quad_form(u[:,k]+self.const_offset, self.R)
            constraints += [x[:,k+1] == self.linear_dynamics.A @ x[:,k] + self.linear_dynamics.B @ u[:,k]]
            constraints += [self.xmin <= x[:,k], x[:,k] <= self.xmax]
            constraints += [self.umin <= u[:,k], u[:,k] <= self.umax]

        objective += cvx.quad_form(x[:,self.n_pred] - self.set_pt, self.Q_n)
        self.mpc_prob = cvx.Problem(cvx.Minimize(objective), constraints)    

    def update_initial_guess_(self):
        """
        Update the intial guess of the solution (z_init, u_init)
        :return:
        """
        z_last = self.cur_z[-1, :]
        u_new = self.cur_u[-1, :]
        z_new = self.dynamics_object.eval_dot(z_last, u_new, None)

        self.z_init[:-1, :] = self.cur_z[1:, :]
        self.z_init[-1, :] = z_new

        self.u_init[:-1, :] = self.cur_u[1:, :]
        self.u_init[-1, :] = u_new
        self.u_init_flat[:-self.nu] = self.u_init_flat[self.nu:]
        self.u_init_flat[-self.nu:] = u_new

        self.x_init = self.linear_dynamics.C @ self.z_init.T
        self.x_init_flat = self.x_init.flatten(order='F')

    def eval(self, x, t):
        # TODO: Add support for update of reference trajectory

        self.x_init.value = x
        time_eval0 = time.time()
        self.mpc_prob.solve(solver=cvx.OSQP, warm_start=True)
        self.comp_time.append(time.time()-time_eval0)
        assert self.mpc_prob.status == 'optimal', 'MPC not solved to optimality'
        u0 = self.mpc_prob.variables()[1].value[:,0]
        return u0
    
    def solve_mpc_(self):
        """
        Solve the MPC sub-problem
        :return:
        """
        self.prob.update(q=self._osqp_q, Ax=self._osqp_A_data, l=self._osqp_l, u=self._osqp_u)
        self.res = self.prob.solve()
        self.dz_flat = self.res.x[:(self.N + 1) * self.nx]
        self.du_flat = self.res.x[(self.N + 1) * self.nx:(self.N + 1) * self.nx + self.nu * self.N]


    def eval2(self, x, t):
        """
        Run single iteration of SQP-algorithm to get control signal in closed-loop control
        :param x: (np.array) Current state
        :param t: (float) Current time (for time-dependent dynamics)
        :return: u: (np.array) Current control input
        """
        t0 = time.time()
        z = self.dynamics_object.lift(x.reshape((1, -1)), None).squeeze()   # Not compiled with Numba
        self.update_initial_guess_()                                        # Not compiled with Numba
        self.update_objective_()                                            # Compiled with Numba
        A_lst, B_lst = self.update_linearization_()                         # Not compiled with Numba
        self.update_constraint_matrix_data_(A_lst, B_lst)                   # Not compiled with Numba
        self.update_constraint_vecs_(z, t)                                  # Compiled with Numba
        t_prep = time.time() - t0

        self.solve_mpc_()
        self.cur_z, self.cur_u, self.u_init_flat = _update_current_sol(self.z_init, self.dz_flat, self.u_init,
                                                                        self.du_flat, self.u_init_flat, self.nx,
                                                                        self.nu, self.N)  # Compiled with Numba
        self.comp_time.append(time.time() - t0)
        self.prep_time.append(t_prep)
        self.qp_time.append(self.comp_time[-1] - t_prep)

        if self.dynamics_object.standardizer_u is None:
            return self.cur_u[0, :]
        else:
            return self.dynamics_object.standardizer_u.inverse_transform(self.cur_u[0, :])

    def construct_objective_(self):
        """
        Construct MPC objective function
        :return:
        """
        # Quadratic objective:

        if not self.add_slack:
            self._osqp_P = sparse.block_diag([sparse.kron(sparse.eye(self.N), self.C.T @ self.Q @ self.C),
                                                self.C.T @ self.QN @ self.C,
                                                sparse.kron(sparse.eye(self.N), self.R)], format='csc')

        else:
            self._osqp_P = sparse.block_diag([sparse.kron(sparse.eye(self.N), self.C.T @ self.Q @ self.C),
                                                self.C.T @ self.QN @ self.C,
                                                sparse.kron(sparse.eye(self.N), self.R),
                                                self.Q_slack], format='csc')

        # Linear objective:
        if not self.add_slack:
            self._osqp_q = np.hstack(
                [(self.C.T @ self.Q @ (self.C @ self.z_init[:-1, :].T - self.xr.reshape(-1, 1))).flatten(order='F'),
                    self.C.T @ self.QN @ (self.C @ self.z_init[-1, :] - self.xr),
                    (self.R @ (self.u_init.T + self.const_offset)).flatten(order='F')])

        else:
            self._osqp_q = np.hstack(
                [(self.C.T @ self.Q @ (self.C @ self.z_init[:-1, :].T - self.xr.reshape(-1, 1))).flatten(order='F'),
                    self.C.T @ self.QN @ (self.C @ self.z_init[-1, :] - self.xr),
                    (self.R @ (self.u_init.T + self.const_offset)).flatten(order='F'),
                    np.zeros(self.ns * (self.N))])

        
    def update_objective_(self):
        """
        Construct MPC objective function
        :return:
        """
        self._osqp_q[:self.nx * (self.N + 1) + self.nu * self.N] = \
            _update_objective(self.C, self.Q, self.QN, self.R, self.xr, self.x_init, self.u_init, self.const_offset.squeeze(),
                                self.nx, self.nu, self.N)
    
    def get_state_prediction(self):
        """
        Get the state prediction from the MPC problem
        :return: Z (np.array) current state prediction
        """
        return self.cur_z

    def get_control_prediction(self):
        """
        Get the control prediction from the MPC problem
        :return: U (np.array) current control prediction
        """
        return self.cur_u
    
    def solve_to_convergence(self, z, t, z_init_0, u_init_0, eps=1e-3, max_iter=1):
        """
        Run SQP-algorithm to convergence
        :param z: (np.array) Initial value of z
        :param t: (float) Initial value of t (for time-dependent dynamics)
        :param z_init_0: (np.array) Initial guess of z-solution
        :param u_init_0: (np.array) Initial guess of u-solution
        :param eps: (float) Stop criterion, normed difference of the control input sequence
        :param max_iter: (int) Maximum SQP-iterations to run
        :return:
        """
        iter = 0
        self.cur_z = z_init_0
        if self.dynamics_object.standardizer_u is None:
            self.cur_u = u_init_0
        else:
            self.cur_u = self.dynamics_object.standardizer_u.inverse_transform(u_init_0)
        u_prev = np.zeros_like(u_init_0)

        while (iter == 0 or np.linalg.norm(u_prev - self.cur_u) / np.linalg.norm(u_prev) > eps) and iter < max_iter:
            t0 = time.time()
            u_prev = self.cur_u.copy()
            self.z_init = self.cur_z.copy()
            self.x_init = (self.C @ self.z_init.T)
            self.u_init = self.cur_u.copy()

            # Update equality constraint matrices:
            A_lst, B_lst = self.update_linearization_()

            # Solve MPC Instance
            self.update_objective_()
            self.construct_constraint_vecs_(z, None)
            self.update_constraint_matrix_data_(A_lst, B_lst)
            t_prep = time.time() - t0

            self.solve_mpc_()
            dz = self.dz_flat.reshape(self.N + 1, self.nx)
            du = self.du_flat.reshape(self.N, self.nu)

            alpha = 1
            self.cur_z = self.z_init + alpha * dz
            self.cur_u = self.u_init + alpha * du
            self.u_init_flat = self.u_init_flat + alpha * self.du_flat

            iter += 1
            self.comp_time.append(time.time() - t0)
            self.prep_time.append(t_prep)
            self.qp_time.append(self.comp_time[-1] - t_prep)
            self.x_iter.append(self.cur_z.copy().T)
            self.u_iter.append(self.cur_u.copy().T)
    



