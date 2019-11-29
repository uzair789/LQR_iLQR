"""LQR, iLQR and MPC."""

import numpy as np
import scipy.linalg as scp
import copy
import gym

from deeprl_hw6.arm_env import TwoLinkArmEnv

def simulate_dynamics(env, x, u, dt=1e-5):
    """Step simulator to see how state changes.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.
    u: np.array
      The command to test. When approximating B you will need to
      perturb this.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    xdot: np.array
      This is the **CHANGE** in x. i.e. (x[1] - x[0]) / dt
      If you return x you will need to solve a different equation in
      your LQR controller.
    """

    env.state = x
    next_state, _, _, _ = env.step(u, dt)
    change = (next_state - x) / dt
    #return np.array(next_state)
    return np.array(change)
    #return np.zeros(x.shape)


def approximate_A(env, x, u, delta=1e-5, dt=1e-5):
    """Approximate A matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. You will need to perturb this.
    u: np.array
      The command to test.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    A: np.array
      The A matrix for the dynamics at state x and command u.
    """
    A = np.zeros((x.shape[0], x.shape[0]))
    for i,x_dim in enumerate(x):
        x_copy1 = x.copy()
        x_copy2 = x.copy()
        x_copy1[i] = x_copy1[i] + delta
        x_pos = simulate_dynamics(env, x_copy1, u)
        x_copy2[i] = x_copy2[i] - delta
        x_neg = simulate_dynamics(env, x_copy2, u)

        print(x_pos, type(x_pos))
        print('x_neg', x_neg, type(x_neg))

        A[:, i] = (x_pos - x_neg) / (2 * delta)

    return A
    #return np.zeros((x.shape[0], x.shape[0]))


def approximate_B(env, x, u, delta=1e-5, dt=1e-5):
    """Approximate B matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test.
    u: np.array
      The command to test. You will ned to perturb this.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    B: np.array
      The B matrix for the dynamics at state x and command u.
    """
    B = np.zeros((x.shape[0], u.shape[0]))
    for i, u_dim in enumerate(u):
        u_copy1 = u.copy()
        u_copy2 = u.copy()
        u_copy1[i] = u_copy1[i] + delta
        x_pos = simulate_dynamics(env, x, u_copy1) 
        u_copy2[i] = u_copy2[i] - delta
        x_neg = simulate_dynamics(env, x, u_copy2)
        B[:, i] = (x_pos - x_neg) / (2 * delta)

    return B
    #return np.zeros((x.shape[0], u.shape[0]))


def calc_lqr_input(env, sim_env):
    """Calculate the optimal control input for the given state.

    If you are following the API and simulate dynamics is returning
    xdot, then you should use the scipy.linalg.solve_continuous_are
    function to solve the Ricatti equations.

    Parameters
    ----------
    env: gym.core.Env
      This is the true environment you will execute the computed
      commands on. Use this environment to get the Q and R values as
      well as the state.
    sim_env: gym.core.Env
      A copy of the env class. Use this to simulate the dynamics when
      doing finite differences.

    x, u : current state and action (added by me)


    Returns
    -------
    u: np.array
      The command to execute at this point.
    """
    u = env.action_space.sample()
    u = np.array([0, 0])
    x = env.state
    A = approximate_A(sim_env, x, u)
    B = approximate_B(sim_env, x, u)
    Q = env.Q
    R = env.R

    P = scp.solve_continuous_are(A, B, Q, R)

    print('A --', A.shape)
    print('R --', R.shape)
    print('B --', B.shape)
    print('P --', P.shape)
    print('x --', x.shape)


    action_u = - np.linalg.inv(R).dot(B.T).dot(P).dot( x.reshape(-1,1)) # check the multiplication. element wise or dot product

    return action_u
    #return np.ones((2,))



if __name__ == '__main__':
    #loop over states
    #for each state compute the u and apply the action
    env = gym.make('TwoLinkArm-v0')
    sim_env = copy.deepcopy(env)
    
    x = env.reset()
    done = False
    while not done:
        u = calc_lqr_input(env, sim_env)
        next_x, r, done, _ = env.step(x , u)
        x = next_x
        print(r) 


