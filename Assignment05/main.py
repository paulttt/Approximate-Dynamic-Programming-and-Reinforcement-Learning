import numpy as np
import os
import sys


p = 0.1
action_mask = {'up': [[-1, -1, -1], [-1, 0, 1]],
               'down': [[1, 1, 1], [-1, 0, 1]],
               'left': [[1, 0, -1], [-1, -1, -1]],
               'right': [[-1, 0, 1], [1, 1, 1]],
               'idle': [0, 0]}


def main():
    maze, start, goal = load_maze()
    print(maze, start, goal)
    print(get_action_space(maze, start))


def load_maze():
    fn = sys.argv[1]
    if os.path.exists(fn):
        print("Maze file exists!\n")
        print("Filename: {}".format(os.path.basename(fn)))
    with open(sys.argv[1], 'r') as f:
        content = f.read()
    print(content)
    print('\nProcess Maze to 2D numpy array...')
    l = []
    with open(sys.argv[1], 'r') as f:
        for line in f:
            li = line.strip()
            if not li.startswith("#"):
                l.append([letter for letter in li if letter != ' '])
    maze = np.array(l, dtype=str)
    print("Maze is of datatype {} and has shape {}\n".format(type(maze), maze.shape))
    start_idx = (np.where(maze == 'S')[0][0], np.where(maze == 'S')[1][0])
    goal_idx = (np.where(maze == 'G')[0][0], np.where(maze == 'S')[1][0])
    return maze, start_idx, goal_idx


def get_action_space(maze, current_state, action):
    action_space = {'idle': [], 'up': [], 'down': [], 'left': [], 'right': []}

    if action == 'idle':
        action_space['idle'].append([1, current_state])
    try_action = (current_state[0]+action_mask[action][0][1], [current_state[1]+action_mask[action][1][1]])
    if (0 <= try_action[0] < maze.shape[0] and 0 <= try_action[1] < maze.shape[1]):
        if maze[try_action] == '1':
            break
        action_space[action].append([p, (current_state[0]+action_mask)])
        try_action_left = (current_state[0] + action_mask[action][0][0],
                           [current_state[1] + action_mask[action][1][0]])
        try_action_right = (current_state[0] + action_mask[action][0][2],
                            [current_state[1] + action_mask[action][1][2]])
        if (0 <= try_action[0] < maze.shape[0] and 0 <= try_action[1] < maze.shape[1]):
            if maze[try_action] == '1':
                break
    return action_space[action]

    for i in range(len(action_mask[0])):
        try_action = (current_state[0]+action_mask[0][i], [current_state[1]+action_mask[1][i]])
        if maze[try_action] == '1':
            continue
        elif not (0 > try_action[0] > maze.shape[0] or 0 > try_action[1] > maze.shape[1]):
            continue
        elif maze[try_action] == 'T':
            print("Be aware, the trap is near!")
        else:
            if action_mask[0][i] == -1:
                action_space['up'].append([])
    return action_space


#def get_prob(maze, current_state, action):


if __name__ == '__main__':
    main()
