"""LQR, iLQR and MPC."""

import numpy as np
import gym
from  deeprl_hw6.arm_env import *


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
    xdot,reward,is_done,_ = env.step(u,dt)
    #print(xdot,reward,id_done)
    return xdot/dt
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
    
    for i in range(x.shape[0]):
      curr_x = x.copy()
      curr_x[i] = curr_x[i]+delta
      xplus = simulate_dynamics(env,curr_x,u,dt)
      curr_x = x.copy()
      curr_x[i] = curr_x[i]-delta
      xminus = simulate_dynamics(env,curr_x,dt)
      diff = (xplus - xminus) / (2*delta)
      A[:,i] = diff
    return A


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
    for i in range(x.shape[0]):
      curr_act = u.copy()
      curr_act[i] = curr_act[i]+delta
      xplus = simulate_dynamics(env, x, curr_act, dt)
      curr_act = u.copy()
      curr_act[i] = curr_act[i]-delta
      xminus = simulate_dynamics(env, x, curr_act, dt)
      diff = (xplus - xminus)/(2*delta)
      B[:,i] = diff
    return B


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

    Returns
    -------
    u: np.array
      The command to execute at this point.
    """
    x = env.state
    u = np.zeros((2,))
    A = approximate_A(sim_env,x,u)
    B = approximate_B(sim_env,x,u)
    Q = env.Q
    R = env.R
    K = scipy.linalg.solve_continuous_are(A,B,Q,R)
    #xdot = simulate_dynamics(sim_env,x,u)*env.dt
    xdot = x - env.goal()
    prod = np.dot(K,xdot)
    U = -np.dot(np.dot(np.linalg.inv(R),B.T),prod)
    return np.ones((2,))

if __name__ == '__main__':

  env = gym.make('TwoLinkArm-v0')
  sim_env = gym.make('TwoLinkArm-v0')

  for steps in range(1000):
    for t in range(100):
      u = calc_lqr_input(env,sim_env)
      env.step(u)
      env.render()