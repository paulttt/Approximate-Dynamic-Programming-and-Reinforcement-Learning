import numpy as np
import os
import sys


p = 0.1
action_mask = {'up': [[-1, -1, -1], [-1, 0, 1]],
               'down': [[1, 1, 1], [-1, 0, 1]],
               'left': [[1, 0, -1], [-1, -1, -1]],
               'right': [[-1, 0, 1], [1, 1, 1]],
               'idle': [[0, 0, 0], [0, 0, 0]]}


def main():
    maze, start, goal = load_maze()
    #print(maze, start, goal)
    print(get_action_space(maze, start, 'up'))
    print(get_lookup_table(maze, start))


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
    maze = np.array(lines, dtype=str)
    print("Maze is of datatype {} and has shape {}\n".format(type(maze), maze.shape))
    start_idx = (np.where(maze == 'S')[0][0], np.where(maze == 'S')[1][0])
    goal_idx = (np.where(maze == 'G')[0][0], np.where(maze == 'S')[1][0])
    return maze, start_idx, goal_idx


def get_lookup_table(maze, current_state):
    lookup_table = []
    for action, mask in action_mask.items():
        try_action = (current_state[0] + mask[0][1], current_state[1] + mask[1][1])
        if maze[try_action] != '1':
            lookup_table.append(action)
    return lookup_table


def get_action_space(maze, current_state, action):
    action_space = {'idle': [], 'up': [], 'down': [], 'left': [], 'right': []}

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

'''
def cost_to_go(maze, current_state, action):
    action_space = get_action_space(maze, current_state, action)
    cost = 0
    for action in action_space:
        if maze[action[1]] == 'T':
            cost += 50
'''


if __name__ == '__main__':
    main()
