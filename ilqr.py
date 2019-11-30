"""LQR, iLQR and MPC."""

#from deeprl_hw3.controllers import approximate_A, approximate_B
import gym
import os
import matplotlib.pyplot as plt
import scipy.linalg as scp
import numpy as np
import scipy.linalg
import copy

from deeprl_hw6.arm_env import TwoLinkArmEnv

def simulate_dynamics_next(env, x, u):
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

    Returns
    -------
    next_x: np.array
    """
    env.state = copy.deepcopy(x)
    next_state = env.step(u)
    return np.array(next_state)
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
        x_neg = simulate_dynamics_next(env, x_copy2, u)
        A[:, i] = (x_pos - x_neg) / (2 * delta)
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
    for i, u_dim in enumerate(u):
        u_copy1 = u.copy()
        u_copy2 = u.copy()
        u_copy1[i] = u_copy1[i] + delta
        x_pos = simulate_dynamics(env, x, u_copy1) 
        u_copy2[i] = u_copy2[i] - delta
        x_neg = simulate_dynamics_next(env, x, u_copy2)
        B[:, i] = (x_pos - x_neg) / (2 * delta)
    return B
def cost_inter(env, x, u):
    """intermediate cost function

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

    Returns
    -------
    l, l_x, l_xx, l_u, l_uu, l_ux. The first term is the loss, where the remaining terms are derivatives respect to the
    corresponding variables, ex: (1) l_x is the first order derivative d l/d x (2) l_xx is the second order derivative
    d^2 l/d x^2
    """
    d = {}

    l = np.linalg.norm(u)
 
    

    d{ 'l_x' : l_x,
       'l_xx' : l_xx,
       'l_u' : l_u,
       'l_uu' : l_uu,
       'l_ux' : l_ux }



    return l, d


def cost_final(env, x):
    """cost function of the last step

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.

    Returns
    -------
    l, l_x, l_xx The first term is the loss, where the remaining terms are derivatives respect to the
    corresponding variables
    """
    return None


def simulate(env, x0, U):
    return None


def calc_ilqr_input(env, sim_env, tN=50, max_iter=1e6):
    """Calculate the optimal control input for the given state.


    Parameters
    ----------
    env: gym.core.Env
      This is the true environment you will execute the computed
      commands on. Use this environment to get the Q and R values as
      well as the state.
    sim_env: gym.core.Env
      A copy of the env class. Use this to simulate the dynamics when
      doing finite differences.
    tN: number of control steps you are going to execute
    max_itr: max iterations for optmization

    Returns
    -------
    U: np.array
      The SEQUENCE of commands to execute. The size should be (tN, #parameters)
    """
    # generate a control sequence U
    U = np.zeros([tN, 2])
    x0 = env.state

    # compute the forward roll out
    traj_cost, traj_derivatives = forward(sim_env, x0, U)



    exit()


    # backward
    k, K = backward(traj_derivatives)

    #update the control_sequences

    return np.zeros((50, 2))

def forward(sim_env, x0, U):
    """ Forward rollout

    Parameters
    ----------
    sim_env : env object
    x0: 
      The first state of the trajectory
    U: np.array
      The SEQUENCE of commands to execute [tN, params]

    Returns
    -------
    traj_costs: np.array
        Array of costs for each x, u pair at each time step in the forward roll out
    traj_derivatives: List[dict]
        List of dicts of all derivatives at each timestep

    """

    traj_costs = []
    traj_derivatives = []

    x = x0
    for i, u in enumerate(U):
        inter_cost, inter_derivatives = cost_inter(sim_env, x, u)
        traj_costs.append(inter_cost)
        traj_derivatives.append(inter_derivatives)
        next_x = simulate_dynamics_next(sim_env, x, u)
        x = next_x

        #compute the final state cost
        if i == len(U) - 1:
            final_cost, final_derivatives = cost_final(sim_env, x)
            traj_costs.append(final_cost)
            traj_derivatives.append(final_derivatives)

    return traj_costs, traj_derivatives




PATH = './ilqr_plots'
def plot_graph(data, title, xlabel, ylabel):
        plt.figure(figsize=(12,5))
        plt.title(title)
        for i in range(data.shape[1]):
            plt.plot(data[:, i])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(os.path.join(PATH, title+'.png'))


if __name__=='__main__':
    env = gym.make('TwoLinkArm-v0')
    sim_env = copy.deepcopy(env)

    done = False

    q = []
    qdot = []
   
    q.append(env.position)
    qdot.append(env.velocity)

    rewards = []

    #get the optimal U's for this env (test roll out)
    U_optimal = calc_ilqr_input(env, sim_env)

    for i in range(len(U_optimal)):
        next_x, r, done, _ = env.step(U_optimal[i])
        if done:
            print('CRASHED!!')
            break

        rewards.append(r)
        q.append(env.position)
        qdot.append(env.velocity)

    

    actions, q, qdot, rewards = np.array(U_optimal), np.array(q, ndmin=2), np.array(qdot, ndmin=2), np.array(rewards, ndmin=2)
    plot_graph(actions, 'U_control', 'episode steps', 'control')
    plot_graph(q, 'positions', 'episode steps', 'q')
    plot_graph(qdot, 'velocity', 'episode steps', 'qdot')
    plot_graph(rewards.T, 'rewards', 'episode steps', 'rewards')







