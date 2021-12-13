import numpy as np
import matplotlib.pyplot as plt
from random import uniform
from kalman_aux import *

class KalmanFilter:
    
    def __init__(self, A, B, C, Q, R, n, times):
        """
        Init function for Kalman Filter
        """
        self.A = A                                                                                      # A, hidden process for state matrix
        self.B = B                                                                                      # B, hidden process for state matrix
        self.C = C                                                                                      # C, observable process for state matrix

        self.Q = Q                                                                                      # Q, covariance matrix related to w_t (defined in kalman_aux.py)
        self.R = R                                                                                      # R, covariance matrix related to v_t (defined in kalman_aux.py)

        self.times = times                                                                              # save times for simulation
        self.P = np.diag(np.full(self.A.shape[0], sigma_w))                                             # P, process covariance matrix initial estimation
        
        self.z_t = np.zeros((2,))                                                                       # init observation process
        self.x_t = np.zeros((4,))                                                                       # uniform initial
        
        self.u_t = np.array([u_f(i) for i in range(times)])                                             # input function
        
        self.w_t = np.random.normal(mu, sigma, (100, 4))                                                # gaussian noise R^2
        self.v_t = np.random.normal(mu, sigma, (100, 2))                                                # gaussian noise R^4
        
        self.n = n                                                                                      # number of experiments
        self.e_t = np.zeros((times, n))                                                                 # init relative error
        #self.e_t = np.zeros((times))                                                                    # init relative error
        self.K_t = np.zeros((A.shape[0], C.shape[0]))                                                   # init K Gain

    def kf_system(self, t):
        """
        Computes dynamic system defined by
            x_{t+1} = Ax_t + w_t
                z_t = Cx_t + v_t
        """
        self.z_t = self.C @ self.x_t + self.v_t[t]                                                      # update observation process
        self.x_t = self.A @ self.x_t + self.B * self.u_t[t] + self.w_t[t]                               # update states

    def kf_gain(self, P_bar):
        """
        Computes the Kalman Gain given by
            K_t(C \bar P_t C' + R) = \bar P_t C'
        """
        self.K_t = P_bar @ self.C.T @ np.linalg.inv(self.C @ P_bar @ self.C.T + self.R)                 # update kalman gain

    def kf_update_matrix(self, x_bar, P_bar):
        """
        Updates the estimate and the covariance
        """
        x_hat = self.x_t + self.K_t @ (self.z_t - self.C @ self.x_t)                                    # update the estimate
        P_t = np.identity(self.C.shape[1]) - self.K_t @ self.C                                          # update the covariance

        return x_hat, P_t

    def kf_compute_priors(self, x_hat, P_t, t):
        """
        Computes the priors (predict)
        """
        x_bar = self.A @ x_hat + self.B * self.u_t[t]                                                   # compute the priors (state matrix)
        P_bar = self.A @ P_t @ self.A.T + self.Q                                                        # compute the priors (process covariance matrix)

        return x_bar, P_bar

    def kf_relative_error(self, x_hat, i, t):
        """
        Computes relative error
        """    
        self.e_t[t, i] = (np.linalg.norm(x_hat - self.x_t) ** 2) / (np.linalg.norm(self.x_t) ** 2)      #  relative error
        #self.e_t[t] = (np.linalg.norm(x_hat - self.x_t) ** 2) / (np.linalg.norm(self.x_t) ** 2)         #  relative error
            
    def kf_run(self):
        """
        KF simulation
        """
        # initial values
        x_bar = self.x_t
        P_bar = self.P
        #x_pred = np.zeros((self.n, self.times, A.shape[0]))

        # n experiments
        for i in range(self.n):
            # simulation times = 0,...,99
            for t in range(self.times):
                self.kf_system(t)
                self.kf_gain(P_bar)
                x_hat, P_t = self.kf_update_matrix(x_bar, P_bar)
                x_bar, P_bar = self.kf_compute_priors(x_hat, P_t, t)
                self.kf_relative_error(x_hat, i, t)
                #self.kf_relative_error(x_hat, t)
                #x_pred[i, t] = x_hat
                #print('x_hat ..... ', x_hat)

        #print('aaa --- ', x_pred.shape)
        #print('bbb ---', np.mean(x_pred, axis=0).shape)
        #print('ccc ---', self.e_t.shape)
        mean_err = np.mean(self.e_t, axis=1)
        std_err = np.std(self.e_t, axis=1) / self.n  
        
        return self.e_t, mean_err, std_err

a_1 = [0.2, 0.1, 0., -0.1]
a_2 = [0.99, 0.1, 0., -0.1]
a_3 = [1., 0.1, 0., -0.1]
a_4 = [0.2, 0.1, 0., -1.]
a_values = [a_1, a_2, a_3, a_4]

B = np.ones(4).T
C = np.array([[1., 0., 0., 0.],
              [0., 1., 0., 0.]])

n = 10
times = 100
mu, sigma = 0., 1.

KF_outputs = []
errors, stds, acc = [], [], []
for matrix in a_values:
    A = np.array(mk_mat(matrix))
    KF = KalmanFilter(A, B, C, Q, R, n, times)
    err_kf, mean_err, std_err = KF.kf_run()
    KF_outputs.append(KF)
    errors.append(mean_err)
    stds.append(std_err)
    acc.append(np.cumsum(mean_err))

fig, ax = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(f'Relative error of Kalman Filter for {n} experiments')
ax = np.array(ax).flatten()
for i in range(4):
    ax[i].plot(np.arange(times), errors[i], label=f"Mean of relative error")
    ax[i].fill_between(np.arange(times), np.maximum(errors[i] - stds[i], 0), errors[i] + stds[i], color = 'purple', alpha = 0.3, label = r"$\pm$ Std")
    ax[i].set_title(f'$A_{i}$')
    ax[i].set_xlabel("Time")
fig.tight_layout()
plt.legend()
plt.show()

fig, ax = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(f'Accumulate relative error of Kalman Filter for {n} experiments')
ax = np.array(ax).flatten()
for i in range(4):
    ax[i].plot(np.arange(times), acc[i], label=f"Mean of relative error")
    ax[i].set_title(f'$A_{i}$')
    ax[i].set_xlabel("Time")
fig.tight_layout()
plt.legend()
plt.show()
