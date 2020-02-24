import numpy as np
import os
import time

import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import random

p = 0.1
gamma = 0.9
convergence_criterion = 10e-10
action_mask = {'up': [[-1, -1, -1], [-1, 0, 1]],
               'down': [[1, 1, 1], [-1, 0, 1]],
               'left': [[1, 0, -1], [-1, -1, -1]],
               'right': [[-1, 0, 1], [1, 1, 1]],
               'idle': [[0, 0, 0], [0, 0, 0]]}

def load_maze():
    fn = sys.argv[1]
    if os.path.exists(fn):
        print("Maze file exists!\n")
        print("Filename: {}".format(os.path.basename(fn)))
    with open(sys.argv[1], 'r') as f:
        content = f.read()
    print(content)
    print('\nProcess Maze to 2D numpy array...')
    lines = []
    with open(sys.argv[1], 'r') as f:
        for line in f:
            li = line.strip()
            if not li.startswith("#"):
                lines.append([letter for letter in li if letter != ' '])
    global maze
    global start
    global goal
    maze = np.array(lines, dtype=str)
    print("Maze is of datatype {} and has shape {}\n".format(type(maze), maze.shape))
    start = (np.where(maze == 'S')[0][0], np.where(maze == 'S')[1][0])
    goal = (np.where(maze == 'G')[0][0], np.where(maze == 'S')[1][0])

def get_lookup_table(current_state):
    lookup_table = []
    if maze[current_state] == '1': return lookup_table
    for action, mask in action_mask.items():
        try_action = (current_state[0] + mask[0][1], current_state[1] + mask[1][1])
        if 0 <= try_action[0] < maze.shape[0] and 0 <= try_action[1] < maze.shape[1]:
            if maze[try_action] != '1' :
                lookup_table.append(action)
        else: continue
    return lookup_table

def get_action_space(current_state, action):
    action_space = {'idle': [], 'up': [], 'down': [], 'left': [], 'right': []}
    # if we are on the wall returning nothing
    if maze[current_state] == '1':
        return []
    if action == 'idle':
        action_space[action].append([1, current_state])
        return action_space[action]
    try_action = [(current_state[0]+action_mask[action][0][i],
                  current_state[1]+action_mask[action][1][i]) for i in range(3)]
    if 0 <= try_action[1][0] < maze.shape[0] and 0 <= try_action[1][1] < maze.shape[1]:
        if maze[try_action[1]] == '1':
            return action_space[action]
        action_space[action].append([1, (current_state[0]+action_mask[action][0][1],
                                         current_state[1]+action_mask[action][1][1])])
        if 0 <= try_action[0][0] < maze.shape[0] and 0 <= try_action[0][1] < maze.shape[1] \
                and maze[try_action[0]] != '1':
            action_space[action][-1][0] = 1-p
            action_space[action].append([p, (current_state[0] + action_mask[action][0][0],
                                             current_state[1] + action_mask[action][1][0])])
        elif 0 <= try_action[2][0] < maze.shape[0] and 0 <= try_action[2][1] < maze.shape[1] \
                and maze[try_action[2]] != '1':
            action_space[action][-1][0] = 1 - 2*p
            action_space[action].append([p, (current_state[0] + action_mask[action][0][2],
                                             current_state[1] + action_mask[action][1][2])])
        else:
            return action_space[action]
    return action_space[action]

def g1(i, u):
    """
    cost function 1
    :param i: tupel of (row_idx, col_idx)
    :param u: decided action as string ('idle', 'up', 'down', 'left', 'right')
    :return: cost
    """
    if maze[i] == 'G': # check for goal
        return -1
    elif maze[i] == 'T': #check for trap
        return 50
    else:
        return 0 # everything else is 'boring'

def g2(i, u):
    if maze[i] == 'G':# and u == 'idle':
        return 0
    elif maze[i] == 'T':
        return 51
    return 1

def state_to_index(state):
    """
    returns an index of the value function vector
    :param state: a tuple (i,j) where i is the the row and j is the column. Counting starts at 0
    :return: index of the value function vector in which the value should be written/changed
    """
    rows, cols = maze.shape
    c_row, c_col = state
    return c_row*cols + c_col


def index_to_state(index):
    """
    @param index: Scalar index in range(#fields in the maze)
    @return: Tuple of the form (row_idx, col_idx) that corresponds to the 1D-index in the 2D Maze array
    """
    rows, cols = maze.shape
    c_row = int(index/cols)
    c_col = index - (c_row*cols)
    return (c_row, c_col)


def opt_bellman(i, g, V):
    u_x = get_lookup_table(i)
    min_val = None; min_u = None
    for u in u_x:
        probs_list = get_action_space(i, u)
        val = 0
        for p, j in probs_list:
            val += p * (g(i,u) + gamma * V[state_to_index(j)])
        if min_val is None or val < min_val:
            min_val = val; min_u = u
    return min_val, min_u

def val_bellman(i, g, V, pi):
    u = pi[state_to_index(i)]
    probs_list = get_action_space(i, u)
    val = 0; min_val = None
    if len(probs_list) == 0: return min_val
    for p, j in probs_list:
        val += p * (g(i,u) + gamma*V[state_to_index(j)])
    if min_val is None or val < min_val:
        min_val = val
    return min_val

def value_iteration(g):
    converged = False
    V = np.empty(maze.shape[0]*maze.shape[1])
    pi = []
    num_iters = 0
    while not converged:
        V_new = np.empty(maze.shape[0]*maze.shape[1])
        pi_ = []
        for i in range(V.shape[0]):
            V_new[i], min_u = opt_bellman(index_to_state(i), g, V)
            pi_.append(min_u)
        if np.nanmax(np.abs(V-V_new)) < convergence_criterion:
            converged = True
        V = V_new
        pi = pi_
        num_iters += 1
    print('converged after %d iterations' %(num_iters-1))
    np.set_printoptions(precision=3)
    return V, pi

def update_policy(i, g, V):
    # argmin E[g(i,u) + gamma * V(j)]
    u_x = get_lookup_table(i)
    min_val = None; min_u = None
    for u in u_x:
        probs_list = get_action_space(i, u)
        val = 0
        for p, j in probs_list:
            val += p * (g(i,u) + gamma * V[state_to_index(j)])
        if min_val is None or val < min_val:
            min_val = val; min_u = u
    return min_u

def policy_iteration(g, optimistic=False, m=None):
    converged = False
    V = np.empty(maze.shape[0]*maze.shape[1])
    pi = []
    # set pi random
    pi = init_pi(V)
    num_iters = 0
    while not converged:
        V_outside = V
        converged_to_vpi = False
        num_iters_inner = 0
        while not converged_to_vpi:
            # compute V_pi_k
            V_new = np.empty(V.shape)*1e6
            for i in range(V.shape[0]):
                V_new[i] = val_bellman(index_to_state(i), g, V, pi)
            if optimistic:
                if num_iters_inner >= m: converged_to_vpi = True
            elif np.nanmax(np.abs(V-V_new)) < convergence_criterion:
                converged_to_vpi = True
            V = V_new
            num_iters_inner += 1
        print('inner convergence after %d iterations' % (num_iters_inner-1))
        np.set_printoptions(precision=3)

        # update policy
        pi_new = []
        for i in range(len(pi)):
            pi_new.append(update_policy(index_to_state(i), g, V))
        pi = pi_new

        if np.nanmax(np.abs(V_outside-V)) < convergence_criterion:
            converged = True
        num_iters += 1
    print('outer convergence after %d iterations' %(num_iters-1))
    np.set_printoptions(precision=3)
    return V, pi

def get_Value_maze(values):
    val_maze = np.zeros(maze.shape)
    for i in range(values.shape[0]):
        val_maze[index_to_state(i)] = values[i]
    #print(val_maze)
    return val_maze

def heat_map(name, V, pi):
    plt.figure()
    scale_arrow = 0.4
    val_maze = get_Value_maze(V)
    val_max = np.nanmax(val_maze)
    with np.errstate(invalid='ignore'):
        ax = sns.heatmap(val_maze, vmin=np.nanmin(val_maze), vmax=np.nanmax(val_maze[val_maze<val_max]), cmap='Blues')
        sns.heatmap(val_maze, ax=ax, mask=(val_maze < (val_max-1)), annot=True, cbar=False, cmap='Reds')
    ax.set_aspect("equal")
    ax.set_facecolor('xkcd:black')
    for row in range(val_maze.shape[0]):
        for col in range(val_maze.shape[1]):
            if np.isnan(val_maze[row, col]) == True:
                continue
            x = col + 0.5
            y = row + 0.5
            if pi[state_to_index((row, col))] == 'idle':
                plt.plot(x, y, marker='.', color='black')
                continue
            if pi[state_to_index((row, col))] is not None:
                dx = action_mask[pi[state_to_index((row, col))]][1][1] * scale_arrow
                dy = action_mask[pi[state_to_index((row, col))]][0][1] * scale_arrow
                plt.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc='k', ec='k')
    plt.savefig(name+'.png')

def init_pi(V):
    pi = []
    for i in range(V.shape[0]):
        u_x = get_lookup_table(index_to_state(i))
        if len(u_x) == 0:
            pi.append(None)
            continue
        pi.append(random.choice(u_x))
    return pi

def try_gammas():
    global gamma
    gamma = 0.99
    _, pi_ref = value_iteration(g1)
    counts = []
    for gam in np.linspace(0.01, 0.99, 15):
        print(gam)
        gamma = gam
        _, pi_current =  value_iteration(g1)
        counts.append(compare_lists(pi_current, pi_ref))
    print(counts)
    plt.figure()
    plt.plot(np.linspace(0.01, 0.99, 15), counts)
    plt.xlabel('gamma')
    plt.ylabel('number of differences')
    plt.savefig('gamma_test')

def compare_lists(v1, v2):
    count = 0
    if len(v1) != len(v2):
        print('Vectors of unequal shape {} and {}'.format(len(v1), len(v2)))
    for i in range(len(v1)):
        if v1[i] != v2[i]:
            count += 1
    return count

def compare_runtimes():
    # compare runtimes
    global gamma
    gamma = 0.99
    start = time.time()
    V, pi = value_iteration(g1)
    end = time.time()
    print('the measurement for value_iter: ', (end-start))

    start = time.time()
    V, pi = policy_iteration(g1)
    end = time.time()
    print('the measurement for policy_iter: ', (end - start))

    start = time.time()
    V, pi = policy_iteration(g1, True, 50)
    end = time.time()
    print('the measurement for opt_policy_iter: ', (end - start))

def investigate_opi():
    ms = [i for i in range(1,100,2)]
    ms.append(100)
    runtimes = []
    for m in ms:
        start = time.time()
        V, pi = policy_iteration(g1, True, m)
        end = time.time()
        runtimes.append(end-start)
    plt.figure()
    plt.plot(ms, runtimes)
    plt.xlabel('m')
    plt.ylabel('runtime')
    plt.savefig('runtime_test_for_different_ms')


def main():
    load_maze()
    # value iteration
    folder = 'plots/'

    name = 'value_iter_g1_'+str(gamma)
    print(name+': ')
    V, pi = value_iteration(g1)
    heat_map(folder+name, V, pi)
    print('------------------------------------')

    name = 'value_iter_g2_'+str(gamma)
    print(name+': ')
    V, pi = value_iteration(g2)
    heat_map(folder+name, V, pi)
    print('------------------------------------')


    # policy iteration
    name = 'policy_iter_g1_'+str(gamma)
    print(name+':')
    V2, pi = policy_iteration(g1)
    heat_map(folder+name, V, pi)
    print('------------------------------------')

    name = 'policy_iter_g2_'+str(gamma)
    print(name)
    V, pi = policy_iteration(g2)
    heat_map(folder+name, V, pi)
    print('------------------------------------')



    # optimistic policy iteration
    name = 'opt_policy_iter_g1_'+str(gamma)
    print(name)
    V, pi = policy_iteration(g1, True, 50)
    heat_map(folder+name, V, pi)
    print('------------------------------------')

    name = 'opt_policy_iter_g2_'+str(gamma)
    print(name)
    V, pi = policy_iteration(g2, True, 50)
    heat_map(folder+name, V, pi)
    print('------------------------------------')

    # gamma test
    try_gammas()
    compare_runtimes()
    investigate_opi()

if __name__ == '__main__':
    main()
