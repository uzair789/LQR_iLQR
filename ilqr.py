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

    
    #print('--->>>')
    #print(x)
    #print(u)
    
    env.state = copy.deepcopy(x)
    next_state, _, _, _  = env.step(u)
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
        x_pos = simulate_dynamics_next(env, x_copy1, u)
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
        x_pos = simulate_dynamics_next(env, x, u_copy1) 
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

    inter_scale = 1#0.001
    l = np.square(np.linalg.norm(u))

    x = x.reshape(-1, 1)
    u = u.reshape(-1, 1)


    x_dims = x.shape[0]
    u_dims = u.shape[0]

    l_x = np.zeros(x.shape)
    l_xx = np.zeros([x_dims, x_dims])

    l_u = 2 * u
    l_uu = 2 * np.eye(u_dims)

    l_ux = np.zeros([u_dims, x_dims])
    ''' 
    print('x', x.shape) 
    print('u', u.shape) 
 
    print('l_x', l_x.shape)
    print('l_xx', l_xx.shape)
    print('l_u', l_u.shape)
    print('l_uu', l_uu.shape)
    print('l_ux', l_ux.shape)
    '''
    d =  { 'l_x' : l_x*inter_scale,
       'l_xx' : l_xx*inter_scale,
       'l_u' : l_u*inter_scale,
       'l_uu' : l_uu*inter_scale,
       'l_ux' : l_ux*inter_scale }



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
    scaling = 10000

    x = x.reshape(-1,1)
    x_dims = x.shape[0]

    g = copy.deepcopy(env.goal)
    g = g.reshape(-1,1)
    l = scaling * np.square(np.linalg.norm(x - g))

    l_x = scaling * 2 * (x - g)
    l_xx = scaling * 2 * np.eye(x_dims)

    '''
    print('-- in  final state')
    print('x', x.shape)
    print('g', g.shape)
    print('l_x', l_x.shape)
    print('l_xx', l_xx.shape)
    '''
    d = {'l_x':l_x,
         'l_xx': l_xx }
    


    return l, d


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
    x0 = copy.deepcopy(env.state)
    all_costs  = []# np.zeros([int(max_iter), 1])
    cost_thresh = np.inf
    alpha = 1
    lamb = 1
    for r in range(int(max_iter)):

        X_new = []
        # compute the forward roll out
        #x0 = copy.deepcopy(env.reset())
        traj_states, traj_costs, traj_derivatives = forward(sim_env, x0, U)

        summed_cost = np.sum(traj_costs)
        print('ITER - ', r, ' | costs = ', summed_cost, '| alpha = ', alpha, '| lamb = ', lamb)

        #print(all_costs.shape)
        all_costs.append(summed_cost)

        #print('traj_states', traj_states.shape)
        #print('traj_costs', traj_costs.shape)
        #print('traj_derivatives', traj_derivatives.shape)

        # backward
        k, K = backward(traj_derivatives, lamb)

        #print('k',k.shape)
        #print('K', K.shape)
        #print('U', U.shape)
        #print('states', traj_states.shape)
        #update the control_sequences
        U_new = []
        curr = copy.deepcopy(x0)
        #curr = env.reset()

        '''
        print(curr.shape)
        print(traj_states[:,1].shape)
        print(K[0,:,:].shape)
        print(k[0, :])
        print(U[0, :])
        a = np.dot(K[0,  :, :], curr - traj_states[:, 0])
        print(a.shape)
        '''
        cost = 0
        for t in  range(tN):
            u_new = U[t,:] +   alpha * (k[t, :] + np.dot(K[t,  :, :], curr - traj_states[:, t]))
            #print('---->>>>', u_new.shape)
            #print(r, t, u_new)
            next_s = simulate_dynamics_next(sim_env, curr, u_new)

            c_i, _ = cost_inter(sim_env, curr, u_new)
            cost += c_i
            curr = copy.deepcopy(next_s)
            U_new.append(u_new)
            if t==tN-1:
                c_f, _ = cost_final(sim_env, curr)
                cost += c_f
 
        print('current_cost = ', cost, ' | cost_thresh = ', cost_thresh)
        #print('diff between  U',  np.linalg.norm(U_new - U))
     

        if cost < cost_thresh:
           U =  copy.deepcopy(np.array(U_new))
           cost_thresh = cost
           #lamb = lamb /2.0
        else:
           if r%10 == 0:
               print('clip alpha')
               alpha = max(0.000000001, alpha/2.0)

        
        # test if the  control  takes  state close  to goal
        U_test = copy.deepcopy(U_new)
        state = copy.deepcopy(x0)
        
        for tt, u_test  in enumerate(U_test):
             next_state  = simulate_dynamics_next(env, state, u_test)
             state = copy.deepcopy(next_state)
        
        if  np.linalg.norm(next_state - env.goal) < 0.005:
               break    
        
         
        print('dist from goal', np.linalg.norm(next_state - env.goal))






    #print('updated U', U.shape) 
    all_costs = np.array(all_costs).reshape(-1, 1)
    plot_graph(all_costs, 'Costs', 'iterations', 'costs')
     
    return U


def inv_stable(M, lamb=1):
    """Inverts matrix M in a numerically stable manner.

    This involves looking at the eigenvector (i.e., spectral) decomposition of the
    matrix, and (1) removing any eigenvectors with non-positive eingenvalues, and
    (2) adding a constant to all eigenvalues.
    """
    M_evals, M_evecs = np.linalg.eig(M)
    M_evals[M_evals < 0] = 0.0
    M_evals += lamb
    M_inv = np.dot(M_evecs,
                   np.dot(np.diag(1.0 / M_evals), M_evecs.T))
    return M_inv



def backward(traj_derivatives, lamb):
    """This function performs the backward pass to compute the Qs at each time step.

    Arguments:
    ---------

    Returns:
    -------
    the gains-- k and K 
    """
    final_grads = traj_derivatives.pop()

    #print(final_grads)

    k_list =  []
    K_list = []

    V_x = final_grads['l_x']
    V_xx = final_grads['l_xx']

    # now going backwards ([49 --> 0])
    for t in range(len(traj_derivatives)-1, -1, -1):
        l_x = traj_derivatives[t]['l_x']
        f_x = traj_derivatives[t]['f_x']
        l_u = traj_derivatives[t]['l_u']
        f_u = traj_derivatives[t]['f_u']
        l_xx = traj_derivatives[t]['l_xx']
        l_ux = traj_derivatives[t]['l_ux']
        l_uu = traj_derivatives[t]['l_uu']
        '''
        print(t, '---')
        print('l_x', l_x.shape)
        print('f_x', f_x.shape)
        print('l_u', l_u.shape)
        print('f_u', f_u.shape)
        print('V_x', V_x.shape)
        print('V_xx', V_xx.shape)
        print('l_xx', l_xx.shape)
        print('l_ux', l_ux.shape)
        print('l_uu', l_uu.shape)
        '''      

        Q_x = l_x + np.dot(f_x.T, V_x) 
        Q_u = l_u + np.dot(f_u.T, V_x)
        Q_xx = l_xx + f_x.T.dot(V_xx).dot(f_x) #f_xx is 0
        Q_ux = l_ux + f_u.T.dot(V_xx).dot(f_x)
        Q_uu = l_uu + f_u.T.dot(V_xx).dot(f_u)

        '''
        print('Q_x', Q_x.shape)
        print('Q_u', Q_u.shape)
        print('Q_xx', Q_xx.shape)
        print('Q_ux', Q_ux.shape)
        print('Q_uu', Q_uu.shape)
        '''
        # compute the gains
        #Q_uu_inv = np.linalg.pinv(Q_uu + lambd *  np.eye(Q_uu.shape[0]))
        #Q_uu_inv = np.linalg.inv(Q_uu)

        Q_uu_inv = inv_stable(Q_uu, lamb=lamb)

        k = - Q_uu_inv.dot(Q_u)
        K = - Q_uu_inv.dot(Q_ux)
        '''
        print('k', k, k.shape )
        print('K', K, K.shape )
        '''
        V_x = Q_x - K.T.dot(Q_uu).dot(k)
        V_xx = Q_xx - K.T.dot(Q_uu).dot(K)

        '''
        print('V_x updated', V_x.shape)
        print('V_xx updated', V_xx.shape)
        '''
        k_list.append(k)
        K_list.append(K)


    # reverse the gains so that the list contains k and K from 0 to T-1
    k_list.reverse()
    K_list.reverse()

    k_list = np.array(k_list).squeeze()
    K_list = np.array(K_list)

    '''
    print('k_list', k_list.shape)
    print('K_list', K_list.shape)
    '''

    return k_list, K_list

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
    states = np.zeros([x0.shape[0], U.shape[0]+1])
    traj_costs = []
    traj_derivatives = []

    x = copy.deepcopy(x0)
    states[:, 0] = copy.deepcopy(x)
    for i, u in enumerate(U):
        inter_cost, inter_derivatives = cost_inter(sim_env, x, u)
        traj_costs.append(inter_cost)

        #Add the df/dx and df/du also along with the derivatives of the cost
        df_x = approximate_A(sim_env, x, u)
        df_u = approximate_B(sim_env, x, u)
        inter_derivatives.update({'f_x':df_x, 'f_u':df_u})
        #print('----')
        #for key in inter_derivatives:
        #     print(i, key)
        traj_derivatives.append(inter_derivatives)
        next_x = simulate_dynamics_next(sim_env, x, u)
        x = copy.deepcopy(next_x)
        states[:, i+1] = copy.deepcopy(x)

        #compute the final state cost
        if i == len(U) - 1:
            final_cost, final_derivatives = cost_final(sim_env, x)
            traj_costs.append(final_cost)
            traj_derivatives.append(final_derivatives)


    #print('len costs per trajectory', len(traj_costs))
    
    #print('len derivatives per trajectory', len(traj_derivatives))
    return states, traj_costs, traj_derivatives


MAX_ITERS = 5000
TN = 700

PATH = './ilqr_plots_iters'+str(MAX_ITERS)+'_Tn'+str(TN)
os.mkdir(PATH)
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


    env.reset()
    sim_env.reset() 
    done = False

    q = []
    qdot = []
   
    q.append(env.position)
    qdot.append(env.velocity)

    rewards = []

    #get the optimal U's for this env (test roll out)
    U_optimal = calc_ilqr_input(env, sim_env, tN = TN, max_iter=MAX_ITERS)

    env.reset()
    for i in range(len(U_optimal)):
        next_x, r, done, _ = env.step(U_optimal[i])
        #env.render()
        if done:
            print('CRASHED!!')
            break

        rewards.append(r)
        q.append(env.position)
        qdot.append(env.velocity)

    

    actions, q, qdot, rewards = np.array(U_optimal), np.array(q, ndmin=2), np.array(qdot, ndmin=2), np.array(rewards, ndmin=2)
    print('Final Rewards = ', np.sum(rewards))
    plot_graph(actions, 'U_control', 'episode steps', 'control')
    plot_graph(q, 'positions', 'episode steps', 'q')
    plot_graph(qdot, 'velocity', 'episode steps', 'qdot')
    plot_graph(rewards.T, 'rewards', 'episode steps', 'rewards')







