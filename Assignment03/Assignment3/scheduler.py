import itertools
from queue import Queue
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt


def main():
    Q = 3
    S = 3
    T = 2
    N = 10

    system = System(Q, S, T)
    V, pis = system.dp(N)
    print('V:')
    print(V)
    print('----\n pis: ')
    print(pis)

    print('\n\n -----------------')
    print('running some tests')
    state = system.create_state_from_tuple(((2), (0, 0, 0), (0, 0, 0), (0, 0, 0)))
    # state = system.create_state_from_tuple(((1),(1,1,1),(1,1,1),(1,1,1)))
    print(state)
    print('probs of workers: ', 1 - system.w)
    print(pis[0, :])
    best_action = pis[0, system.get_index(state)]
    print(best_action)


    random_val = rand.randint(0, 10125)
    start_state_tuple = system.all_possibilities()[random_val]
    x = system.create_state_from_tuple(start_state_tuple)
    costs = np.empty((10, 250))
    for k in range(10):
        for i in range(250):
            u = pis[k, system.get_index(x)]
            costs[k, i] = system.g(x, u, system.w)
            x = system.f(x, u, system.w)
        plt.plot(costs[k, :])


# for i in pis[0,:]:
# print(i)

class System:
    def __init__(self, Q, S, T):
        self.w = rand.uniform(0, 1, size=Q)
        print('Probabilities for the workers: ')
        print(self.w)
        self.S = S
        self.T = T
        self.Q = Q

    def dp(self, N):
        state_space = self.all_possibilities()
        V = np.empty((N + 1, len(state_space)))
        pis = np.empty((N, len(state_space))).astype('int')
        # last stage:
        for idx, x in enumerate(state_space):
            x = self.create_state_from_tuple(x)
            V[N, idx] = self.g_N(x)

        for stage in range(N - 1, -1, -1):
            print('stage: ', stage)
            # iterative step
            for idx, x in enumerate(state_space):
                # state space should be in the same order as
                # the indices
                x = self.create_state_from_tuple(x)
                u_set = self.retrieve_actions(x)
                min_val = -1
                min_u = -1  # worst case stay idle
                for u in u_set:
                    # print('u: ', u)
                    states, probs = x.calculate_following_states(u, self.w, self.T)
                    #for state in states: print(state)
                    val = 0
                    for idx_2, state in enumerate(states):
                        val += (self.g(x, u, self.w)+V[stage + 1, self.get_index(state)]) * probs[idx_2]
                    # val = self.g(x,u,self.w)+V[stage+1,self.get_index(self.f(x.copy(),min_u,self.w))]
                    if val < min_val or min_val == -1:
                        # print('min_u before: ', min_u)
                        min_u = u;
                        min_val = val
                    # print('min_u: ', min_u)
                pis[stage, idx] = min_u
                V[stage, idx] = min_val  # + V[stage+1,self.get_index(self.f(x,min_u,self.w))]
        return V, pis

    def f(self, x, u, w):
        for idx, qs in enumerate(x.qstates):
            rn = w[idx]
            if rn < self.calc_probabilities()[idx] and not qs.q.empty():
                qs.solve_task()

        if x.new_job == 0:
            x.new_job = rand.randint(0, self.T)
            return

        if u > 0 and x.new_job != 0:
            x.qstates[u - 1].add_task(x.new_job)
            x.new_job = rand.randint(0, self.T)

        return x

    def g(self, x, u, w):
        cost = 0
        for qs in x.qstates:
            cost += qs.q.qsize()
        if cost > 0 and u == 0:
            cost += 5
        return cost

    def g_N(self, x):
        cost = 0
        for qs in x.qstates:
            cost += qs.q.qsize()
        return cost

    def map_one_q(self, t):
        l = len(t)
        index = 0
        for i in range(l):
            index += t[i] ** (l - i)
        return index

    def get_index(self, x):
        index = x.new_job * 15 ** self.Q
        # now iterate over the queues and mulitply 15**smth
        for i in range(self.Q):
            index += 15 ** (self.Q - 1 - i) * self.map_one_q(x.qstates[i].q.queue)
        return index

    def create_one_q(self, T):
        qstates = []
        for i in range(T + 1):
            end1 = T + 1 if i != 0 else 1
            for j in range(end1):
                end2 = T + 1 if j != 0 else 1
                for k in range(end2):
                    qstates.append((i, j, k))
        return qstates

    def all_possibilities(self):
        qstates = self.create_one_q(self.T)
        my_list = [[i for i in range(self.T + 1)]]
        my_list += [qstates for i in range(self.Q)]
        return list(itertools.product(*my_list))

    def create_state_from_tuple(self, tupl):
        qs = []
        for i in range(1, len(tupl)):
            qs.append(QueueState(len(tupl[i]), tupl[i]))
        return State(qs, tupl[0])

    def calc_probabilities(self):
        w = rand.uniform(0, 1, size=self.Q)
        return w

    def retrieve_actions(self, x):
        # now retrieve a set with all allowed actions for certain state x
        s = set()
        s.add(0)
        if x.new_job == 0: return s
        for idx, qs in enumerate(x.qstates):
            if not qs.q.full():
                s.add(idx + 1)
        return s


class State:
    def __init__(self, qstates, new_job):
        self.qstates = qstates
        self.new_job = new_job

    def append_qstate(qstate):
        self.qstates.append(qstate)

    def copy(self):
        qstates = [state.copy() for state in self.qstates]
        return State(qstates, self.new_job)

    def calculate_following_states(self, u, worker_threshold, T):
        job = self.new_job
        probs = []
        states = []
        old_queues = [q.copy() for q in self.qstates]
        combinations = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
        for comb in combinations:
            prob = 1
            #queues = [None for i in range(len(old_queues))]
            queues = [q.copy() for q in old_queues]
            for queue_idx, task in enumerate(comb):
                if old_queues[queue_idx].q.empty():
                    #queues[queue_idx] = old_queues[queue_idx].copy()
                    if task==1: continue
                elif task == 0:
                    prob *= worker_threshold[queue_idx]
                    #queues[queue_idx] = old_queues[queue_idx].copy()
                elif task == 1:
                    prob *= (1 - worker_threshold[queue_idx])
                    #old_queues[queue_idx].copy().solve_task()
                    queues[queue_idx].solve_task()
            if job == 0:
                prob *= 1 / (T + 1)
                for new_job in range(T + 1):
                    states.append(State(queues, new_job))
                    probs.append(prob)
            elif u == 0 and job != 0:
                new_job = job
                states.append(State(queues, new_job))
                probs.append(prob)
            else:
                queues[u - 1].add_task(job)
                prob *= 1 / (T + 1)
                for new_job in range(T + 1):
                    states.append(State(queues, new_job))
                    probs.append(prob)
        return states, probs

    def __str__(self):
        string = ""
        string += "New Job to process: " + str(self.new_job) + "\n"
        for idx, s in enumerate(self.qstates):
            string += str(idx + 1) + ": " + str(s)
        return string


class QueueState:
    def __init__(self, S, tupl=None):
        self.size = S
        self.q = Queue(maxsize=S)
        if tupl is not None:
            for i in tupl:
                if i == 0:
                    continue
                self.add_task(i)

    def add_task(self, t):
        if not self.q.full():
            self.q.put(t)
        else:
            print("Queue is already full")

    def copy(self):
        return QueueState(self.size, self.q.queue)

    def solve_task(self):
        return self.q.get()
        '''
        if not(self.q.empty()):
            self.q.get()
        else:
            print('Trying to solve task in emtpy queue')
        '''

    def __str__(self):
        string = ""
        idx = -1
        for idx, i in enumerate(self.q.queue):
            string += str(i)
            string += " "
        for j in range(idx + 1, self.size):
            string += "- "
        string += "\n"
        return string


if __name__ == "__main__":
    main()

