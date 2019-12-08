import numpy as np
import matplotlib.pyplot as plt

Q = 3
S = 3
T = 2

N = 10

### samples a new job from an uniform distribution
def new_job():
    return np.random.random_integers(0,T)

### initializes empty queues and samples a new job to be processed
def init_state(Q,S,T):
    t = new_job()
    init_state = [[[0 for s in range(S)] for q in range(Q)], t]
    return init_state

### prints out the current state of the queue
def print_queue(current_state):
    print('New Job to process: {}'.format(current_state[1]))
    for i in range(Q):
        print('{}: '.format(i+1), end=' ')
        for j in range(S):
            if current_state[0][i][j] == 0:
                print('-', end=' ')
            else:
                print('{}'.format(current_state[0][i][j]), end=' ')
        print('')

### pops the current job from the list and samples a new one to be processed.
def process_job(current_state):
    job_to_process = current_state.pop()
    next_job = new_job()
    current_state.append(next_job)
    return job_to_process, current_state

def action_space(current_state):
    U = [0]
    if current_state[1] == 0:
        U.append(q for q in range())
        return U

    for q_nr, queue in enumerate(current_state[0]):
        if queue[-1] != 0:
            continue
        U.append(q_nr+1)
    return U

def init_uncertainty():
    W = [0]
    for w in range(T):
        W.append(np.random.random_sample())
    return W

def get_ind_leftmost(current_state):
    queues = current_state[0]
    ind = []
    for q in range(Q):
        ind_list = np.nonzero(queues[q])
        print(ind_list[0])
        if len(ind_list[0]) == 0:
            ind.append(-1)
        else:
            ind.append(np.amax(ind_list[0]))
    return ind


def system_dynamics(current_state, u, W):
    queues = current_state[0]
    leftmost = get_ind_leftmost(current_state)
    for q in range(Q):
        if leftmost[q] == -1:
            continue
        sample = np.random.random_sample()
        if sample >= W[q]:
            current_state[0][q].pop(0)
            current_state[0][q].append(0)
    if u != 0:
        ind = queues[q].index(next(filter(lambda x: x == 0, queues[q])))
        current_state[0][u-1][ind] = current_state[1]
        job, _ = process_job(current_state)
        current_state[1] = job



def get_fill_level(current_state):
    fill_level = 0
    for i in range(len(current_state[0])):
        if current_state[0] != 0:
            fill_level += 1
    return fill_level

def cost_to_go(current_state, u, W):
    cost_to_go = 0
    if u == 0:
        cost_to_go += 5
    cost_to_go += get_fill_level(current_state)
    return cost_to_go

def cost(current_state, u, W, k):
    if k==N:
        cost = get_fill_level(current_state)
        return cost




if __name__ == '__main__':
    current_state = [[[0,0,0],[1,2,1],[1,0,0]],2]
    print(current_state)
    print_queue(current_state)
    print(get_ind_leftmost(current_state))